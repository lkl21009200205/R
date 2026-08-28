from random import random

import numpy as np
import torch
from numpy.ma import copy
import torch.nn.functional as F
import copy
import random

# Generate representative models.
def generate_representative_models(model_storage):
    num_models = len(model_storage)
    represent_model_storage = []

    for i in range(num_models):
        # 1. Create a new model by copying the current model structure and initial parameters.
        represent_model = type(model_storage[i])()  # Create an empty model of the same type.
        represent_model.load_state_dict(model_storage[i].state_dict())  # Initialize parameters.

        # 2. Set weights: current model gets 50%, and other models split the remaining 50% equally.
        weights = [0.0] * num_models
        weights[i] = 0.5  # Self weight is 50%.

        remaining_weight = 0.5  # Remaining 50% weight.
        for j in range(num_models):
            if j != i:  # Weight for other models.
                weights[j] = remaining_weight / (num_models - 1)  # Split the remaining weight equally across other models.

        # 3. Get the new model's state_dict for storing fused parameters.
        blended_state_dict = represent_model.state_dict()

        # 4. Fuse parameters layer by layer.
        for name, param in blended_state_dict.items():
            if not param.is_floating_point():
                blended_state_dict[name] = model_storage[i].state_dict()[name].clone()
                continue
            blended_param = torch.zeros_like(param)
            for j in range(num_models):
                # Accumulate parameters from all models with weights.
                blended_param += weights[j] * model_storage[j].state_dict()[name]
            blended_state_dict[name] = blended_param

        # 5. Update representative model parameters.
        represent_model.load_state_dict(blended_state_dict)
        represent_model_storage.append(represent_model)

        print(f"Generated representative model {i+1}/{num_models} (base model={i}, self weight=50%)")

    return


# Generate representative models.
def create_representative_models(local_models):
    """
    Efficient representative model generation.

    Optimizations:
    1. Cache all model states in advance.
    2. Compute fusion in batches.
    """
    num_models = len(local_models)
    self_weight = 0.5

    if num_models == 0:
        return []

    if not (0 <= self_weight <= 1):
        raise ValueError(f"self_weight must be between 0 and 1, got: {self_weight}")

    # Compute weights.
    other_weight = (1.0 - self_weight) / (num_models - 1) if num_models > 1 else 0.0

    # 1. Preload all model states.
    print("Loading all model states...")
    model_states = [model.state_dict() for model in local_models]

    # 2. Get the parameter name list.
    param_names = list(model_states[0].keys())

    # 3. Create representative models.
    representative_models = []

    for i in range(num_models):
        print(f"\nFusing representative model {i}...")

        # Deep-copy one model as the template.
        rep_model = copy.deepcopy(local_models[i])
        fused_state = {}

        # Fuse each parameter.
        for name in param_names:
            if not model_states[i][name].is_floating_point():
                fused_state[name] = model_states[i][name].clone()
                continue
            # Contribution from the current model.
            fused_param = self_weight * model_states[i][name]

            # Contributions from other models.
            for j in range(num_models):
                if j != i:
                    fused_param += other_weight * model_states[j][name]

            fused_state[name] = fused_param

        # Load fused parameters.
        rep_model.load_state_dict(fused_state)
        rep_model.eval()

        representative_models.append(rep_model)

    return representative_models


def create_representative_models_fuse_late_layers(
    local_models,
    *,
    self_weight: float = 0.5,
    late_layer_prefixes=("layer4", "fc"),
):
    """Generate representative models by fusing only late layers, recommended for ResNet-18 layer4 + fc.

    Design goals:
    - Preserve each client's early-layer feature extraction ability by avoiding full-parameter averaging.
    - Form consensus only in more discrimination-sensitive late layers, which better supports malicious-update detection and validation.

    Args:
        local_models: local models participating in this round after training.
        self_weight: self weight in [0,1]; remaining weight is split equally across other clients.
        late_layer_prefixes: parameter-name prefixes to fuse.
            - For torchvision ResNet-18, the default is ("layer4", "fc").
            - To include layer3 as a late layer, pass ("layer3", "layer4", "fc").

    Returns:
        representative_models: representative model list with the same length as local_models.
    """

    num_models = len(local_models)
    if num_models == 0:
        return []
    if not (0.0 <= float(self_weight) <= 1.0):
        raise ValueError(f"self_weight must be between 0 and 1, got: {self_weight}")
    if num_models == 1:
        # No fusion is needed when there is only one client.
        rep_model = copy.deepcopy(local_models[0])
        rep_model.eval()
        return [rep_model]

    other_weight = (1.0 - float(self_weight)) / (num_models - 1)
    prefixes = tuple(str(p) for p in late_layer_prefixes)

    print("Loading all model states for late-layer fusion...")
    model_states = [model.state_dict() for model in local_models]
    param_names = list(model_states[0].keys())

    def _is_late_param(name: str) -> bool:
        return any(name.startswith(prefix) for prefix in prefixes)

    representative_models = []
    for i in range(num_models):
        print(f"\nFusing representative model {i} (late layers only: {prefixes}) ...")

        rep_model = copy.deepcopy(local_models[i])
        fused_state = {}

        # Fuse only late-layer parameters; keep the remaining parameters from the client itself.
        for name in param_names:
            if _is_late_param(name) and model_states[i][name].is_floating_point():
                fused_param = float(self_weight) * model_states[i][name]
                for j in range(num_models):
                    if j != i:
                        fused_param += other_weight * model_states[j][name]
                fused_state[name] = fused_param
            else:
                fused_state[name] = model_states[i][name]

        rep_model.load_state_dict(fused_state)
        rep_model.eval()
        representative_models.append(rep_model)

    return representative_models


def _extract_late_layer_float_tensors_from_state(
    state_dict,
    *,
    late_layer_prefixes=("layer4", "fc"),
):
    """Extract floating-point tensors with the specified late-layer prefixes from a state_dict for cross-round difference analysis.

    Returns: dict[name] = cpu_float_tensor_clone
    - Keep only floating-point tensors so integer running-stat entries do not break differencing.
    - Copy everything to CPU to reduce GPU memory usage and simplify cross-round storage.
    """

    prefixes = tuple(str(p) for p in late_layer_prefixes)
    # late_state stores late-layer parameter snapshots, not differences, for next-round delta-w computation.
    late_state = {}
    for name, tensor in state_dict.items():
        if not any(name.startswith(p) for p in prefixes):
            continue
        if not torch.is_tensor(tensor):
            continue
        if not tensor.is_floating_point():
            continue
        # Store uniformly on CPU:
        # 1) Avoid filling GPU memory with cross-round snapshots.
        # 2) Next round only needs numerical differences/similarity, for which CPU is sufficient.
        late_state[name] = tensor.detach().to("cpu").float().clone()
    return late_state


def update_prev_late_layer_snapshots(
    current_local_models,
    client_ids,
    prev_late_state_storage,
    *,
    late_layer_prefixes=("layer4", "fc"),
    verbose=True,
):
    """Write late-layer parameter snapshots for this round's participating clients into prev_late_state_storage.

    prev_late_state_storage is designed as a list:
        - index = client_id
        - content = dict(parameter name -> CPU floating-point tensor)

    This ensures cross-round alignment by client_id even when candidates change randomly each round.
    """

    if prev_late_state_storage is None:
        raise ValueError("prev_late_state_storage cannot be empty; initialize it in main.py as an array with length conf['no_models']")
    if len(current_local_models) != len(client_ids):
        raise ValueError("current_local_models and client_ids must have the same length")

    # updated is used only for printing how many client_id snapshots were updated this round.
    updated = 0
    for model, cid in zip(current_local_models, client_ids):
        if cid is None:
            continue
        snap = _extract_late_layer_float_tensors_from_state(
            model.state_dict(),
            late_layer_prefixes=late_layer_prefixes,
        )
        if 0 <= int(cid) < len(prev_late_state_storage):
            # Core behavior: prev_late_state_storage is cross-round memory, written by client_id.
            prev_late_state_storage[int(cid)] = snap
            updated += 1

    if verbose:
        prefixes = tuple(str(p) for p in late_layer_prefixes)
        print(f"\n[Structural Anomaly Detection] Updated previous-round snapshots for {updated} clients")
        print("[Structural Anomaly Detection] prev_late_state_storage stores: prev_late_state_storage[client_id] = {late-layer parameter name -> CPU float tensor}")
        print(f"[Structural Anomaly Detection] Late-layer prefixes: {prefixes}")

    return prev_late_state_storage


def compute_structural_anomaly_scores_cross_round(
    current_local_models,
    client_ids,
    prev_late_state_storage,
    *,
    round_index: int,
    late_layer_prefixes=("layer4", "fc"),
    verbose=True,
):
    """Compute structural anomaly scores from late-layer parameter changes, delta-w, for the same client across rounds.

    Core idea, concise but suitable for the paper:
    - For each client i, take the late-layer snapshots w_i^{t-1} and w_i^t, then compute delta_w_i = w_i^t - w_i^{t-1}.
    - Build the group center delta_w_bar from this round's participating clients, using the mean center here.
    - Two evidence terms:
        1) Direction consistency: cos(delta_w_i, delta_w_bar).
        2) Magnitude anomaly: deviation of ||delta_w_i|| from the group median/MAD.
    - Output struct_score_i in [0,1], where larger means more structurally normal.

    Constraint:
    - Round 0, the first round, treats structural differences as normal by default, sets all struct_score values to 1.0, and writes snapshots for the next round.
    """

    num_models = len(current_local_models)
    if num_models == 0:
        return [], prev_late_state_storage, {}
    if len(client_ids) != num_models:
        raise ValueError("current_local_models and client_ids must have the same length")
    if prev_late_state_storage is None:
        raise ValueError("prev_late_state_storage cannot be empty")

    prefixes = tuple(str(p) for p in late_layer_prefixes)

    # round_index<=0, the first round:
    # - There is no previous round for comparison, so all client structural differences default to normal (1.0).
    # - Still write this round's late-layer parameter snapshots into prev_late_state_storage for the next round.
    if int(round_index) <= 0:
        struct_scores = [1.0] * num_models
        prev_late_state_storage = update_prev_late_layer_snapshots(
            current_local_models,
            client_ids,
            prev_late_state_storage,
            late_layer_prefixes=late_layer_prefixes,
            verbose=verbose,
        )
        detailed = {
            "round_index": int(round_index),
            "late_layer_prefixes": prefixes,
            "struct_scores": struct_scores,
            "note": "round_index<=0: structural differences default to normal (1.0); snapshots are stored for the next round only",
        }
        return struct_scores, prev_late_state_storage, detailed

    # 1) Extract this round's late-layer states, i.e. snapshots w_i^t.
    current_late_states = []
    for model in current_local_models:
        current_late_states.append(
            _extract_late_layer_float_tensors_from_state(
                model.state_dict(),
                late_layer_prefixes=late_layer_prefixes,
            )
        )

    # 2) Build delta_w_i = w_i^t - w_i^{t-1}.
    #    Only compute it for clients with a snapshot from the previous round (valid_mask=True).
    delta_states = [None] * num_models
    valid_mask = [False] * num_models
    for i, cid in enumerate(client_ids):
        if cid is None:
            continue
        cid_int = int(cid)
        if not (0 <= cid_int < len(prev_late_state_storage)):
            continue
        prev_state = prev_late_state_storage[cid_int]
        if prev_state is None:
            continue

        curr_state = current_late_states[i]
        # shared_names are late-layer parameter names present in both current and previous states, so they can be safely differenced.
        shared_names = [n for n in curr_state.keys() if n in prev_state]
        if not shared_names:
            continue

        d = {}
        for n in shared_names:
            d[n] = curr_state[n] - prev_state[n]
        delta_states[i] = d
        valid_mask[i] = True

    # If no client can compute delta_w this round, for example because the participating clients were absent last round,
    # default structural scores to 1.0 while still updating snapshots for later rounds.
    if not any(valid_mask):
        struct_scores = [1.0] * num_models
        prev_late_state_storage = update_prev_late_layer_snapshots(
            current_local_models,
            client_ids,
            prev_late_state_storage,
            late_layer_prefixes=late_layer_prefixes,
            verbose=verbose,
        )
        detailed = {
            "round_index": int(round_index),
            "late_layer_prefixes": prefixes,
            "struct_scores": struct_scores,
            "note": "No usable historical snapshots, possibly due to candidate changes or first participation: structural scores default to 1.0 this round, and snapshots are updated",
        }
        return struct_scores, prev_late_state_storage, detailed

    # 3) Compute the group center delta_w_bar.
    # Use element-wise parameter means as the center instead of concatenating every parameter into one long vector;
    # this is more memory-efficient and more interpretable.
    center_delta = {}
    # Use any valid parameter set as the reference.
    base_i = next(i for i, ok in enumerate(valid_mask) if ok)
    base_names = list(delta_states[base_i].keys())

    for name in base_names:
        # Only include valid clients that actually contain this parameter in the mean for better robustness.
        tensors = []
        for i in range(num_models):
            if not valid_mask[i]:
                continue
            if name not in delta_states[i]:
                continue
            tensors.append(delta_states[i][name])
        if tensors:
            center_delta[name] = torch.stack(tensors, dim=0).mean(dim=0)

    # 4) Compute structural metrics for each client.
    # - cos(delta_w_i, delta_w_bar): whether the direction matches the group; more consistent is more normal.
    # - ||delta_w_i||: whether the magnitude is anomalous; too large or too small is suspicious.
    eps = 1e-12
    center_norm_sq = 0.0
    for name, t in center_delta.items():
        center_norm_sq += float((t * t).sum().item())
    center_norm = float(np.sqrt(max(center_norm_sq, eps)))

    norms = [0.0] * num_models
    cosines = [0.0] * num_models
    for i in range(num_models):
        if not valid_mask[i]:
            continue
        dot = 0.0
        norm_sq = 0.0
        for name, t in delta_states[i].items():
            norm_sq += float((t * t).sum().item())
            if name in center_delta:
                dot += float((t * center_delta[name]).sum().item())
        nrm = float(np.sqrt(max(norm_sq, eps)))
        norms[i] = nrm
        cosines[i] = float(dot / (nrm * center_norm + eps))

    # Magnitude anomaly: estimate group scale with median + MAD (median absolute deviation).
    # MAD is more robust to extremes than mean/variance and is suitable when malicious clients may exist.
    valid_norms = [norms[i] for i in range(num_models) if valid_mask[i]]
    med = float(np.median(valid_norms))
    mad = float(np.median([abs(x - med) for x in valid_norms]))
    mad = max(mad, 1e-6)

    struct_scores = [1.0] * num_models
    for i in range(num_models):
        if not valid_mask[i]:
            struct_scores[i] = 1.0
            continue

        # Map cosine from [-1,1] to [0,1].
        cos01 = max(0.0, min(1.0, (cosines[i] + 1.0) * 0.5))
        # Larger z means greater deviation from the group median; exp(-z) maps it to (0,1].
        z = abs(norms[i] - med) / mad
        norm_score = float(np.exp(-z))
        norm_score = max(0.0, min(1.0, norm_score))

        # Structural score: direction (50%) + magnitude (50%).
        struct_scores[i] = 0.5 * cos01 + 0.5 * norm_score

    if verbose:
        print(f"\n[Structural Anomaly Detection] round={int(round_index)} late-layer delta-w structural scores (larger is more normal)")
        print(f"[Structural Anomaly Detection] Late-layer prefixes used: {prefixes}")
        print(f"[Structural Anomaly Detection] Magnitude statistics: median(||delta_w||)={med:.6f}, MAD={mad:.6f}")
        for i in range(num_models):
            if valid_mask[i]:
                print(
                    f"  idx={i} client_id={client_ids[i]} struct={struct_scores[i]:.4f} "
                    f"cos={cosines[i]:.4f} norm={norms[i]:.6f}"
                )
            else:
                print(f"  idx={i} client_id={client_ids[i]} struct=1.0000 (no historical snapshot)")

    # 5) Store this round's late-layer snapshots for next-round comparison regardless of structural scores.
    prev_late_state_storage = update_prev_late_layer_snapshots(
        current_local_models,
        client_ids,
        prev_late_state_storage,
        late_layer_prefixes=late_layer_prefixes,
        verbose=verbose,
    )

    detailed = {
        "round_index": int(round_index),
        "late_layer_prefixes": prefixes,
        "valid_mask": valid_mask,
        "norms": norms,
        "cosines": cosines,
        "struct_scores": struct_scores,
        "median_norm": med,
        "mad_norm": mad,
        "note": "struct_score = 0.5*scaled_cos + 0.5*exp(-|norm-med|/MAD)",
    }
    return struct_scores, prev_late_state_storage, detailed


def validate_models_lipc_ds_with_structure_anomaly(
    represent_models,
    validation_models,
    *,
    current_local_models,
    client_ids,
    prev_late_state_storage,
    round_index: int,
    late_layer_prefixes=("layer4", "fc"),
    weight_ds: float = 0.5,
    weight_struct: float = 0.5,
    verbose=True,
):
    """Final score = 0.5 * D-S score + 0.5 * structural anomaly score.

    - D-S score: reuse validate_models_lipc_ds calculation, based on accuracy + loss evidence.
    - Structural anomaly score: direction consistency + magnitude anomaly of cross-round late-layer delta-w.
    """

    if abs(float(weight_ds) + float(weight_struct) - 1.0) > 1e-8:
        s = float(weight_ds) + float(weight_struct)
        weight_ds = float(weight_ds) / s
        weight_struct = float(weight_struct) / s

    # Compute structural anomaly scores first; they need cross-round storage and return the updated prev_late_state_storage.
    struct_scores, prev_late_state_storage, struct_detail = compute_structural_anomaly_scores_cross_round(
        current_local_models,
        client_ids,
        prev_late_state_storage,
        round_index=int(round_index),
        late_layer_prefixes=late_layer_prefixes,
        verbose=verbose,
    )

    # Then compute the D-S evidence fusion score, reusing the original 624-864 logic.
    ds_scores = validate_models_lipc_ds(represent_models, validation_models)

    if len(ds_scores) != len(struct_scores):
        raise ValueError("ds_scores and struct_scores must have the same length")

    # Final score: linear fusion of the two terms, defaulting to 0.5 / 0.5.
    final_scores = [
        float(weight_ds) * float(ds_scores[i]) + float(weight_struct) * float(struct_scores[i])
        for i in range(len(ds_scores))
    ]

    if verbose:
        print(f"\n[Final Score] final = {weight_ds:.2f}*DS + {weight_struct:.2f}*Struct")
        for i in range(len(final_scores)):
            print(
                f"  idx={i} final={final_scores[i]:.4f} (DS={ds_scores[i]:.4f}, Struct={struct_scores[i]:.4f})"
            )

    detailed = {
        "round_index": int(round_index),
        "weight_ds": float(weight_ds),
        "weight_struct": float(weight_struct),
        "ds_scores": ds_scores,
        "struct": struct_detail,
        "final_scores": final_scores,
    }
    return final_scores, prev_late_state_storage, detailed


# # Use half of the local clients as the validation dataset.
# def split_validation_models(clients):
#     num_models = len(clients)
#     num_validation = num_models // 2  # Strictly take the first half.
#
#     validation_models = []
#     for i in range(num_validation):
#         original_model = clients[i]
#
#         # Core method: fully copy the model through state_dict.
#         new_state_dict = {
#             name: param.clone()  # Explicitly clone tensors to ensure independent memory.
#             for name, param in original_model.state_dict().items()
#         }
#
#         # Create a new model without calling __init__.
#         new_model = original_model.__class__.__new__(original_model.__class__)
#         if hasattr(new_model, '_apply'):
#             new_model._apply(lambda t: t)  # Trigger basic initialization.
#
#         # Load the copied parameters.
#         new_model.load_state_dict(new_state_dict, strict=True)
#         validation_models.append(new_model)
#
#     print(f"Generated validation models (total {num_validation})")
#     return validation_models

# Build the validation dataset.
def split_validation_models(model_storage):
    num_models = len(model_storage)
    num_validation = num_models // 2  # Strictly take the first half.

    validation_models = []
    for i in range(num_validation):
        validation_models.append(model_storage[i])

    print(f"Randomly selected validation datasets (total {num_validation})")
    return validation_models

# Model-validation LIPC loss deviation.
def validate_models_loss(represent_models, validation_models):
    num_models = len(represent_models)
    representative_loss = [None] * num_models

    for i in range(num_models):
        model = represent_models[i]  # Representative model.
        model.eval()  # Switch to evaluation mode.
        total_loss = 0.0
        total_samples = 0  # Used to compute average loss.

        with torch.no_grad():  # Do not compute gradients, saving memory.
            for batch_id, batch in enumerate(validation_models[0].train_loader):
                data, target = batch

                if torch.cuda.is_available():
                    data = data.cuda()
                    target = target.cuda()

                output = model(data)  # Compute output with the representative model.
                batch_loss = torch.nn.functional.cross_entropy(output, target, reduction='sum').item()
                total_loss += batch_loss
                total_samples += len(data)  # Count total samples.

        # Compute average loss, optionally.
        representative_loss[i] = total_loss / total_samples if total_samples > 0 else 0.0

    print("---------Representative model LIPC metric calculation complete-------------")

    for i, loss in enumerate(representative_loss):
        print(f"Model {i} LIPC: {loss}")

    return representative_loss

# Model-validation LIPC accuracy.
def validate_models_acc(represent_models, validation_models):
    """
    Compute the LIPC metric, accuracy, for representative models on the validation set.

    Args:
        represent_models: representative model list.
        validation_models: validation model list, each containing train_loader.

    Returns:
        List of LIPC accuracy values for each representative model.
    """
    num_models = len(represent_models)
    representative_lipc = [0.0] * num_models  # Store LIPC metrics, accuracy.

    for i in range(num_models):
        model = represent_models[i]  # Representative model.
        model.eval()  # Switch to evaluation mode.

        total_correct = 0
        total_samples = 0

        with torch.no_grad():  # Do not compute gradients, saving memory.
            # Iterate through the train_loader for all validation models.
            for val_model in validation_models:
                for batch_id, batch in enumerate(val_model.train_loader):
                    data, target = batch


                    if torch.cuda.is_available():
                        data = data.cuda()
                        target = target.cuda()

                    output = model(data)  # Compute output with the representative model.

                    # Get prediction results.
                    pred = output.data.max(1)[1]  # Get the index with maximum probability, i.e. predicted class.

                    # Count correct predictions.
                    correct = pred.eq(target.data.view_as(pred)).cpu().sum().item()

                    total_correct += correct
                    total_samples += len(data)

        # Compute the LIPC metric, accuracy.
        if total_samples > 0:
            lipc = 100.0 * (float(total_correct) / float(total_samples))  # Percentage form.
            # Or use the value between 0 and 1:
            # lipc = float(total_correct) / float(total_samples)
        else:
            lipc = 0.0

        representative_lipc[i] = lipc

        # Print statistics for each validation model, optionally.
        print(f"Representative model {i} tested on {len(validation_models)} validation sets:")
        print(f"  Total samples: {total_samples}, correct predictions: {total_correct}")

    print("\n" + "=" * 50)
    print("Representative model LIPC metric calculation complete")
    print("=" * 50)

    for i, lipc in enumerate(representative_lipc):
        print(f"Model {i} LIPC: {lipc:.2f}%")

    return representative_lipc


# Support multiple normalized LIPC variants.
def validate_models_lipc_advanced(represent_models, validation_models):
    """
    Advanced composite LIPC metric calculation.

    Args:
        represent_models: representative model list.
        validation_models: validation model list.
        accuracy_weight: accuracy weight.
        loss_weight: loss weight.
        loss_norm_method: loss normalization method.
            - 'range': linear range normalization.
            - 'exp': exponential decay exp(-beta * loss).
            - 'sigmoid': sigmoid normalization.
        loss_range: loss range used by the range method.
        beta: decay coefficient used by the exp method.

    Returns:
        Composite LIPC scores and detailed results.
    """
    accuracy_weight = 0.5
    loss_weight = 0.5
    loss_norm_method = 'range'  # 'range', 'exp', 'sigmoid'
    loss_range = (0.1, 5.0)
    beta = 2.0
    # Validate weights.
    if abs(accuracy_weight + loss_weight - 1.0) > 1e-8:
        total_weight = accuracy_weight + loss_weight
        accuracy_weight /= total_weight
        loss_weight /= total_weight

    num_models = len(represent_models)
    lipc_scores = [0.0] * num_models
    accuracies = [0.0] * num_models
    losses = [0.0] * num_models

    print(f"\nLIPC calculation (method: {loss_norm_method})")

    for i, model in enumerate(represent_models):
        model.eval()

        total_correct = 0
        total_loss = 0.0
        total_samples = 0

        with torch.no_grad():
            for val_model in validation_models:
                for data, target in val_model.train_loader:
                    if torch.cuda.is_available():
                        data = data.cuda()
                        target = target.cuda()

                    output = model(data)
                    batch_size = len(data)

                    # Accuracy.
                    pred = output.argmax(dim=1)
                    correct = (pred == target).sum().item()
                    total_correct += correct

                    # Loss.
                    batch_loss = F.cross_entropy(output, target, reduction='sum').item()
                    total_loss += batch_loss

                    total_samples += batch_size

        if total_samples > 0:
            # Compute accuracy.
            accuracy = total_correct / total_samples
            accuracies[i] = accuracy

            # Compute average loss.
            avg_loss = total_loss / total_samples
            losses[i] = avg_loss

            # Compute the loss score according to the selected method.
            if loss_norm_method == 'range':
                # Linear range normalization.
                min_loss, max_loss = loss_range
                normalized_loss = (avg_loss - min_loss) / (max_loss - min_loss) if max_loss > min_loss else 0
                normalized_loss = max(0.0, min(1.0, normalized_loss))
                loss_score = 1.0 - normalized_loss

            elif loss_norm_method == 'exp':
                # Exponential decay.
                loss_score = np.exp(-beta * avg_loss)

            elif loss_norm_method == 'sigmoid':
                # Sigmoid normalization.
                # Assume loss around 2 is the boundary point.
                loss_score = 1.0 / (1.0 + np.exp(avg_loss - 2.0))

            else:
                raise ValueError(f"Unknown normalization method: {loss_norm_method}")

            # Compute composite LIPC.
            lipc_scores[i] = (accuracy_weight * accuracy) + (loss_weight * loss_score)

    # Return results.
    # return lipc_scores, accuracies, losses

    # Return results.
    return lipc_scores

# Support multiple normalized LIPC variants; this extends the previous same-name function.
def validate_models_lipc_advanced(represent_models, validation_models):
    """
    Advanced composite LIPC metric calculation.

    Args:
        represent_models: representative model list.
        validation_models: validation model list.
        accuracy_weight: accuracy weight.
        loss_weight: loss weight.
        loss_norm_method: loss normalization method.
            - 'range': linear range normalization.
            - 'exp': exponential decay exp(-beta * loss).
            - 'sigmoid': sigmoid normalization.
        loss_range: loss range used by the range method.
        beta: decay coefficient used by the exp method.

    Returns:
        Composite LIPC scores and detailed results.
    """
    accuracy_weight = 0.5
    loss_weight = 0.5
    loss_norm_method = 'range'  # 'range', 'exp', 'sigmoid'
    loss_range = (0.1, 5.0)
    beta = 2.0

    # Validate weights.
    if abs(accuracy_weight + loss_weight - 1.0) > 1e-8:
        total_weight = accuracy_weight + loss_weight
        accuracy_weight /= total_weight
        loss_weight /= total_weight

    num_models = len(represent_models)
    lipc_scores = [0.0] * num_models
    accuracies = [0.0] * num_models
    losses = [0.0] * num_models
    loss_scores = [0.0] * num_models
    total_samples_list = [0] * num_models
    total_correct_list = [0] * num_models

    print(f"\n{'=' * 80}")
    print("Representative model LIPC metric calculation start")
    print(f"Accuracy weight: {accuracy_weight:.2f}, loss weight: {loss_weight:.2f}")
    print(f"Loss normalization method: {loss_norm_method}")
    print(f"{'=' * 80}")

    for i, model in enumerate(represent_models):
        model.eval()

        total_correct = 0
        total_loss = 0.0
        total_samples = 0

        with torch.no_grad():
            for val_model in validation_models:
                for data, target in val_model.train_loader:
                    if torch.cuda.is_available():
                        data = data.cuda()
                        target = target.cuda()

                    output = model(data)
                    batch_size = len(data)

                    # Accuracy.
                    pred = output.argmax(dim=1)
                    correct = (pred == target).sum().item()
                    total_correct += correct

                    # Loss.
                    batch_loss = F.cross_entropy(output, target, reduction='sum').item()
                    total_loss += batch_loss

                    total_samples += batch_size

        if total_samples > 0:
            # Compute accuracy.
            accuracy = total_correct / total_samples
            accuracies[i] = accuracy

            # Compute average loss.
            avg_loss = total_loss / total_samples
            losses[i] = avg_loss

            # Store sample information.
            total_samples_list[i] = total_samples
            total_correct_list[i] = total_correct

            # Compute the loss score according to the selected method.
            if loss_norm_method == 'range':
                # Linear range normalization.
                min_loss, max_loss = loss_range
                if max_loss <= min_loss:
                    print(f"Warning: invalid loss_range: {loss_range}; max_loss should be greater than min_loss")
                    loss_score = 1.0 if avg_loss <= min_loss else 0.0
                else:
                    normalized_loss = (avg_loss - min_loss) / (max_loss - min_loss)
                    normalized_loss = max(0.0, min(1.0, normalized_loss))
                    loss_score = 1.0 - normalized_loss

            elif loss_norm_method == 'exp':
                # Exponential decay.
                loss_score = np.exp(-beta * avg_loss)

            elif loss_norm_method == 'sigmoid':
                # Sigmoid normalization.
                # Assume loss around 2 is the boundary point.
                loss_score = 1.0 / (1.0 + np.exp(avg_loss - 2.0))

            else:
                raise ValueError(f"Unknown normalization method: {loss_norm_method}")

            loss_scores[i] = loss_score

            # Compute composite LIPC.
            lipc_scores[i] = (accuracy_weight * accuracy) + (loss_weight * loss_score)

            # Compute contribution.
            accuracy_contribution = accuracy_weight * accuracy
            loss_contribution = loss_weight * loss_score

            # Print detailed results for each model.
            print(f"\n{'─' * 60}")
            print(f"Model {i} detailed results:")
            print(f"{'─' * 60}")
            print(f"  Samples: {total_samples}, correct: {total_correct}")
            print(f"  Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
            print(f"  Average loss: {avg_loss:.4f}")
            print(f"  Normalized loss score: {loss_score:.4f}")
            print(f"  -> Composite LIPC: {lipc_scores[i]:.4f}")
            print(f"    (accuracy contribution: {accuracy_contribution:.4f}, loss contribution: {loss_contribution:.4f})")

        else:
            print(f"\nModel {i}: no valid sample data")
            lipc_scores[i] = 0.0
            accuracies[i] = 0.0
            losses[i] = 0.0
            loss_scores[i] = 0.0

    # Print summary table.
    print(f"\n{'=' * 80}")
    print("Representative model LIPC metric summary")
    print(f"{'=' * 80}")
    print(f"{'Model':<8} {'LIPC Score':<10} {'Accuracy':<10} {'Avg Loss':<10} {'Loss Score':<10} {'Samples':<10} {'Correct':<10}")
    print(f"{'─' * 80}")

    for i in range(num_models):
        print(f"Model{i:<4}  {lipc_scores[i]:<10.4f}  {accuracies[i]:<10.4f}  "
              f"{losses[i]:<10.4f}  {loss_scores[i]:<10.4f}  "
              f"{total_samples_list[i]:<10}  {total_correct_list[i]:<10}")

    # Model ranking.
    if num_models > 0 and any(lipc_scores):
        sorted_indices = sorted(range(num_models), key=lambda i: lipc_scores[i], reverse=True)

        print(f"\n{'=' * 80}")
        print("Model ranking (descending by LIPC score)")
        print(f"{'=' * 80}")

        for rank, idx in enumerate(sorted_indices, 1):
            star = "★" if rank == 1 else ""
            print(f"{rank:2d}. Model{idx} {star:2s} LIPC: {lipc_scores[idx]:.4f}, "
                  f"accuracy: {accuracies[idx]:.4f} ({accuracies[idx] * 100:.2f}%), "
                  f"loss: {losses[idx]:.4f}")

        # Best and worst models.
        best_idx = sorted_indices[0]
        worst_idx = sorted_indices[-1]

        print(f"\n{'=' * 80}")
        print("Best model analysis")
        print(f"{'=' * 80}")
        print(f"Model: Model{best_idx}")
        print(f"  LIPC score: {lipc_scores[best_idx]:.4f}")
        print(f"  Accuracy: {accuracies[best_idx]:.4f} ({accuracies[best_idx] * 100:.2f}%)")
        print(f"  Average loss: {losses[best_idx]:.4f}")
        print(f"  Normalized loss score: {loss_scores[best_idx]:.4f}")
        print(f"  Samples: {total_samples_list[best_idx]}")
        print(f"  Correct: {total_correct_list[best_idx]}")


    print(f"\n{'=' * 80}")
    print("Representative model LIPC metric calculation complete")
    print(f"{'=' * 80}")

    # Return detailed result dictionary.
    detailed_results = {
        'lipc_scores': lipc_scores,
        'accuracies': accuracies,
        'losses': losses,
        'loss_scores': loss_scores,
        'total_samples': total_samples_list,
        'total_correct': total_correct_list,
        'accuracy_weight': accuracy_weight,
        'loss_weight': loss_weight,
        'loss_norm_method': loss_norm_method
    }

    
    # return lipc_scores, detailed_results
    return lipc_scores

# D-S evidence-theory fusion, using accuracy + loss/gradient evidence.
def validate_models_lipc_ds(
    represent_models,
    validation_models,
):
    """
    Use Dempster-Shafer evidence theory to fuse two evidence sources, accuracy and normalized loss score, into a credibility score.

    Args:
        represent_models: representative model list.
        validation_models: validation model list, each containing train_loader.
        loss_norm_method: loss normalization method ('range','exp','sigmoid').
        loss_range: (min, max) for the range method.
        beta: decay coefficient for the exp method.
        tau: smoothing coefficient for mapping sample count to evidence reliability.

    Returns:
        scores: D-S fusion score for each representative model, where larger is more credible.
        If return_detailed=True, also return a detailed dictionary with mG, mB, K, BetP, and other analysis fields.
    """
    loss_norm_method = 'range'
    loss_range = (0.1, 5.0)
    beta = 2.0
    tau = 100.0
    verbose = True
    return_detailed = False
    num_models = len(represent_models)
    scores = [0.0] * num_models
    # Store data for debugging.
    detailed = {
        'accuracies': [0.0] * num_models,
        'losses': [0.0] * num_models,
        'loss_scores': [0.0] * num_models,
        'total_samples': [0] * num_models,
        'total_correct': [0] * num_models,
        'alpha_a': [0.0] * num_models,
        'alpha_g': [0.0] * num_models,
        'mG': [0.0] * num_models,
        'mB': [0.0] * num_models,
        'mTheta': [0.0] * num_models,
        'K': [0.0] * num_models,
        'BetP': [0.0] * num_models,
        'score': [0.0] * num_models,
        'loss_norm_method': loss_norm_method,
        'loss_range': loss_range,
        'beta': beta,
        'tau': tau,
    }

    eps = 1e-12

    if verbose:
        print(f"\n{'=' * 80}")
        print("Representative model D-S evidence-fusion scoring start")
        print("Evidence sources: (1) accuracy (2) normalized loss score")
        print(f"loss_norm_method: {loss_norm_method}, loss_range: {loss_range}, beta: {beta}")
        print(f"Reliability mapping: alpha = 1 - exp(-n/tau), tau: {tau}")
        print("Output score: score = BetP(G) * (1 - K)")
        print(f"{'=' * 80}")

    for i, model in enumerate(represent_models):
        model.eval()

        total_correct = 0
        total_loss = 0.0
        total_samples = 0

        with torch.no_grad():
            for val_model in validation_models:
                for data, target in val_model.train_loader:
                    if torch.cuda.is_available():
                        data = data.cuda()
                        target = target.cuda()

                    output = model(data)
                    batch_size = len(data)

                    pred = output.argmax(dim=1)
                    correct = (pred == target).sum().item()
                    total_correct += correct

                    batch_loss = F.cross_entropy(output, target, reduction='sum').item()
                    total_loss += batch_loss

                    total_samples += batch_size

        detailed['total_samples'][i] = int(total_samples)
        detailed['total_correct'][i] = int(total_correct)

        if total_samples <= 0:
            # No samples; mark as uncertain.
            detailed['accuracies'][i] = 0.0
            detailed['losses'][i] = 0.0
            detailed['loss_scores'][i] = 0.0
            scores[i] = 0.0
            detailed['mTheta'][i] = 1.0
            detailed['score'][i] = 0.0
            if verbose:
                print(f"\nModel {i}: no valid sample data")
            continue

        # Basic values.
        accuracy = total_correct / total_samples
        avg_loss = total_loss / total_samples
        detailed['accuracies'][i] = accuracy
        detailed['losses'][i] = avg_loss

        # Loss normalization, using options consistent with validate_models_lipc_advanced.
        if loss_norm_method == 'range':
            min_loss, max_loss = loss_range
            if max_loss <= min_loss:
                loss_score = 1.0 if avg_loss <= min_loss else 0.0
            else:
                normalized_loss = (avg_loss - min_loss) / (max_loss - min_loss)
                normalized_loss = max(0.0, min(1.0, normalized_loss))
                loss_score = 1.0 - normalized_loss
        elif loss_norm_method == 'exp':
            loss_score = float(np.exp(-beta * avg_loss))
        elif loss_norm_method == 'sigmoid':
            loss_score = 1.0 / (1.0 + np.exp(avg_loss - 2.0))
        else:
            raise ValueError(f"Unknown normalization method: {loss_norm_method}")

        detailed['loss_scores'][i] = loss_score

        # Map values to evidence strength s in [0,1].
        s_a = float(np.clip(accuracy, 0.0, 1.0))
        s_g = float(np.clip(loss_score, 0.0, 1.0))

        # Reliability alpha: adaptively set based on sample count.
        alpha_a = 1.0 - np.exp(- total_samples / (tau + eps))
        alpha_g = alpha_a
        detailed['alpha_a'][i] = float(alpha_a)
        detailed['alpha_g'][i] = float(alpha_g)

        # Build the BBA for each evidence source.
        m_aG = alpha_a * s_a
        m_aB = alpha_a * (1.0 - s_a)
        m_aTheta = 1.0 - alpha_a

        m_gG = alpha_g * s_g
        m_gB = alpha_g * (1.0 - s_g)
        m_gTheta = 1.0 - alpha_g

        # Conflict coefficient K.
        K = m_aG * m_gB + m_aB * m_gG

        # Combine evidence using the closed-form expansion for two evidence sources.
        if 1.0 - K <= eps:
            # Extreme conflict: degrade to complete uncertainty.
            mG = 0.0
            mB = 0.0
            mTheta = 1.0
        else:
            norm = 1.0 / (1.0 - K)
            mG = norm * (m_aG * m_gG + m_aG * m_gTheta + m_aTheta * m_gG)
            mB = norm * (m_aB * m_gB + m_aB * m_gTheta + m_aTheta * m_gB)
            mTheta = norm * (m_aTheta * m_gTheta)

        # Pignistic probability: split uncertainty mass equally.
        BetP_G = mG + 0.5 * mTheta

        # Final score: account for both BetP and conflict degree, penalizing high conflict.
        score = float(BetP_G * (1.0 - K))

        # Save results.
        detailed['mG'][i] = mG
        detailed['mB'][i] = mB
        detailed['mTheta'][i] = mTheta
        detailed['K'][i] = K
        detailed['BetP'][i] = BetP_G
        detailed['score'][i] = score
        scores[i] = score

        if verbose:
            print(f"\n{'─' * 60}")
            print(f"Model {i} detailed results (D-S fusion)")
            print(f"{'─' * 60}")
            print(f"  Samples: {total_samples}, correct: {total_correct}")
            print(f"  Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
            print(f"  Average loss: {avg_loss:.4f}")
            print(f"  Normalized loss score: {loss_score:.4f}")
            print(f"  Reliability: alpha_a={alpha_a:.4f}, alpha_g={alpha_g:.4f}")
            print(f"  BBA fusion: m(G)={mG:.4f}, m(B)={mB:.4f}, m(Theta)={mTheta:.4f}")
            print(f"  Conflict degree: K={K:.4f}")
            print(f"  BetP(G)={BetP_G:.4f}")
            print(f"  -> D-S fusion score: {score:.4f}")

    if verbose:
        print(f"\n{'=' * 80}")
        print("Representative model D-S fusion score summary")
        print(f"{'=' * 80}")
        print(f"{'Model':<8} {'DS Score':<10} {'BetP(G)':<10} {'K':<10} {'Accuracy':<10} {'Avg Loss':<10} {'Loss Score':<10} {'Samples':<10}")
        print(f"{'─' * 80}")

        for i in range(num_models):
            print(
                f"Model{i:<4}  {scores[i]:<10.4f}  {detailed['BetP'][i]:<10.4f}  {detailed['K'][i]:<10.4f}  "
                f"{detailed['accuracies'][i]:<10.4f}  {detailed['losses'][i]:<10.4f}  {detailed['loss_scores'][i]:<10.4f}  "
                f"{detailed['total_samples'][i]:<10}"
            )

        # Model ranking.
        if num_models > 0 and any(scores):
            sorted_indices = sorted(range(num_models), key=lambda j: scores[j], reverse=True)

            print(f"\n{'=' * 80}")
            print("Model ranking (descending by D-S fusion score)")
            print(f"{'=' * 80}")

            for rank, idx in enumerate(sorted_indices, 1):
                star = "★" if rank == 1 else ""
                print(
                    f"{rank:2d}. Model{idx} {star:2s} DS: {scores[idx]:.4f}, "
                    f"BetP(G): {detailed['BetP'][idx]:.4f}, K: {detailed['K'][idx]:.4f}, "
                    f"accuracy: {detailed['accuracies'][idx]:.4f} ({detailed['accuracies'][idx] * 100:.2f}%), "
                    f"loss: {detailed['losses'][idx]:.4f}"
                )

            best_idx = sorted_indices[0]
            print(f"\n{'=' * 80}")
            print("Best model analysis")
            print(f"{'=' * 80}")
            print(f"Model: Model{best_idx}")
            print(f"  D-S fusion score: {scores[best_idx]:.4f}")
            print(f"  BetP(G): {detailed['BetP'][best_idx]:.4f}")
            print(f"  Conflict degree K: {detailed['K'][best_idx]:.4f}")
            print(f"  Accuracy: {detailed['accuracies'][best_idx]:.4f} ({detailed['accuracies'][best_idx] * 100:.2f}%)")
            print(f"  Average loss: {detailed['losses'][best_idx]:.4f}")
            print(f"  Normalized loss score: {detailed['loss_scores'][best_idx]:.4f}")
            print(f"  Samples: {detailed['total_samples'][best_idx]}")
            print(f"  Correct: {detailed['total_correct'][best_idx]}")

        print(f"\n{'=' * 80}")
        print("Representative model D-S evidence-fusion scoring complete")
        print(f"{'=' * 80}")

    if return_detailed:
        return scores, detailed

    return scores


# Select the top 50% models.
def find_top_50_percent_models(lipc_scores):
    """
    Find the model indices corresponding to the top 50% highest LIPC scores.

    Args:
        lipc_scores: LIPC score list.

    Returns:
        List of indices for the top 50% models.
    """
    if not lipc_scores:
        print("Warning: LIPC score array is empty")
        return []

    # Compute how many models to select, rounding up and ensuring at least one.
    n = len(lipc_scores)
    k = max(1, (n + 1) // 2)  # Top 50%, rounded up.

    # Create a list of (score, index) tuples.
    scored_models = [(score, idx) for idx, score in enumerate(lipc_scores)]

    # Sort by score in descending order.
    scored_models.sort(key=lambda x: x[0], reverse=True)

    # Get the indices of the top k models.
    top_indices = [idx for score, idx in scored_models[:k]]

    # Sort indices so they appear in their original order.
    top_indices.sort()

    # Format the output string.
    if top_indices:
        indices_str = ','.join(str(idx) for idx in top_indices)
        print(f"Top 50% good model indices in order: {indices_str}")

        # Print detailed information.
        print("\nDetails:")
        print(f"- Total models: {n}")
        print(f"- Number of top 50% models selected: {k}")
        print(f"- Top {k} highest LIPC scores: {[scored_models[i][0] for i in range(k)]}")
        print(f"- Corresponding indices: {top_indices}")
    else:
        print("No qualifying models found")

    return top_indices


def malicious_model_create(original_model, strength=0.8):
    """
    Simplest malicious model creation method.
    Operate directly on the state_dict to avoid model reconstruction issues.
    """
    print(f"Creating a simple malicious model, strength: {strength}")

    # Get the original model's state_dict.
    original_state = original_model.state_dict()
    malicious_state = {}

    for name, param in original_state.items():
        param_data = param.clone()

        # Only process floating-point parameters.
        if param_data.is_floating_point() and param_data.numel() > 0:
            # Add noise.
            noise_scale = param_data.std() * strength * 5.0
            noise = torch.randn_like(param_data) * noise_scale
            param_data = param_data + noise

            # Random scaling with 30% probability.
            if random.random() < 0.3:
                scale = random.uniform(0.1, 10.0)
                param_data = param_data * scale

            # Partial sign inversion with 20% probability.
            if random.random() < 0.2 and param_data.numel() > 10:
                num_invert = max(1, int(param_data.numel() * 0.1))
                indices = random.sample(range(param_data.numel()), num_invert)
                param_flat = param_data.view(-1)
                param_flat[indices] = -param_flat[indices] * 3.0

        malicious_state[name] = param_data

    # Key step: do not create a new model directly; copy the original model and then load parameters.
    # Use deepcopy to copy the model.
    malicious_model = copy.deepcopy(original_model)
    malicious_model.load_state_dict(malicious_state)
    malicious_model.eval()

    print("Simple malicious model creation complete")
    return malicious_model
# Malicious-model generator that calls the generation methods below.
def create_malicious_model(original_model):
    """
    Create a malicious model from the original model to significantly reduce model accuracy.

    Args:
        original_model: original normal model.
        destruction_level: destruction level from 0 to 1, where 1 means complete destruction.
        method: destruction method.
            - 'smart_destroy': smart targeted destruction, recommended.
            - 'weight_corruption': weight corruption.
            - 'structure_attack': structure attack.
            - 'gradient_reverse': gradient reversal.
            - 'random_chaos': complete randomness.

    Returns:
        malicious_model: malicious model.
    """
    destruction_level = 0.8
    # Select the malicious attack method.
    method = 'smart_destroy'
    # Ensure the destruction level is in a reasonable range.
    destruction_level = max(0.0, min(1.0, destruction_level))

    print("\nCreating malicious model...")
    print(f"Destruction level: {destruction_level:.2f}")
    print(f"Destruction method: {method}")

    # Method 1: use deepcopy and then modify parameters; this is the safest method.
    try:
        # Try to deepcopy the full model.
        import copy
        malicious_model = copy.deepcopy(original_model)
        print("Copied model structure with deepcopy")
    except Exception as e:
        # If deepcopy fails, use method 2.
        print(f"Deepcopy failed: {e}")
        print("Trying to copy the model through state_dict...")

        # Method 2: recreate through the model class and parameters.
        model_class = type(original_model)

        # Try to get the model initialization parameters.
        try:
            # Check whether the model has specific initialization parameters.
            if hasattr(original_model, '__init__'):
                # Get the model initialization signature.
                import inspect
                init_signature = inspect.signature(model_class.__init__)
                params = init_signature.parameters

                # Try to extract initialization parameters.
                init_args = {}
                for param_name in params:
                    if param_name != 'self' and hasattr(original_model, param_name):
                        init_args[param_name] = getattr(original_model, param_name)

                if init_args:
                    malicious_model = model_class(**init_args)
                    print(f"Recreated model with initialization arguments: {init_args}")
                else:
                    # If no parameters are found, try default construction.
                    malicious_model = model_class()
                    print("Created model with the default constructor")
            else:
                malicious_model = model_class()
                print("Created model with the default constructor")
        except Exception as e2:
            print(f"Unable to recreate model: {e2}")
            print("Trying to directly modify the original model state_dict...")

            # Method 3: directly modify the original model state_dict; highest risk.
            malicious_model = original_model
            print("Warning: modifying the original model directly; make sure a backup exists")

    # Get original model parameters.
    original_state = original_model.state_dict()

    # Create malicious parameters according to the selected destruction method.
    if method == 'smart_destroy':
        malicious_state = smart_destroy_method(original_state, destruction_level)
    elif method == 'weight_corruption':
        malicious_state = weight_corruption_method(original_state, destruction_level)
    elif method == 'structure_attack':
        malicious_state = structure_attack_method(original_state, destruction_level)
    elif method == 'gradient_reverse':
        malicious_state = gradient_reverse_method(original_state, destruction_level)
    elif method == 'random_chaos':
        malicious_state = random_chaos_method(original_state, destruction_level)
    else:
        raise ValueError(f"Unknown destruction method: {method}")

    # Load malicious parameters into the new model.
    malicious_model.load_state_dict(malicious_state)

    # Switch to evaluation mode.
    malicious_model.eval()

    return malicious_model

# One malicious model generation method.
def smart_destroy_method(original_state, destruction_level):
    """
    Smart destruction method: selectively damage key parameters.
    """
    malicious_state = {}

    for param_name, param in original_state.items():
        param_data = param.clone()

        # Skip non-floating-point parameters, such as integer parameters.
        if not param_data.is_floating_point():
            # If the parameter is integer typed, first convert it to floating point for processing.
            if param_data.dtype in [torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8]:
                param_data_float = param_data.float()
                is_integer = True
            else:
                # Skip other non-floating-point types directly.
                malicious_state[param_name] = param_data
                continue
        else:
            param_data_float = param_data
            is_integer = False

        # Apply different destruction levels based on parameter type and importance.
        if 'weight' in param_name.lower():
            # Weight matrices are core neural-network parameters, so damage them most heavily.
            destroy_factor = destruction_level * 1.2
        elif 'bias' in param_name.lower():
            # Bias terms are also important, but damage them slightly less.
            destroy_factor = destruction_level * 0.8
        else:
            # Other parameters.
            destroy_factor = destruction_level * 0.5

        # Ensure the factor does not exceed 1.
        destroy_factor = min(destroy_factor, 1.0)

        if destroy_factor > 0 and param_data_float.numel() > 0:
            # Compute statistics while avoiding empty tensors.
            try:
                param_std = param_data_float.std().item()
                param_mean = param_data_float.mean().item()
            except:
                # If std computation fails, use default values.
                param_std = 1.0
                param_mean = 0.0

            # Method 1: add destructive noise.
            noise_scale = param_std * destroy_factor * 3.0
            if noise_scale > 0:
                noise = torch.randn_like(param_data_float) * noise_scale
                param_data_float = param_data_float + noise

            # Method 2: randomize some parameters.
            if destroy_factor > 0.3 and param_data_float.numel() > 1:
                num_elements = param_data_float.numel()
                random_indices = random.sample(range(num_elements),
                                               min(int(num_elements * destroy_factor * 0.3), num_elements))
                param_flat = param_data_float.view(-1)
                for idx in random_indices:
                    param_flat[idx] = torch.randn(1).item() * param_std * 2.0

            # Method 3: apply special corruption to weight matrices.
            if len(param_data_float.shape) >= 2 and destroy_factor > 0.5 and param_data_float.numel() > 1:
                # Try to damage the matrix rank.
                try:
                    param_np = param_data_float.cpu().numpy()
                    if param_np.size > 1 and param_np.shape[0] > 0 and param_np.shape[1] > 0:
                        # Add low-rank noise to damage structure.
                        U = np.random.randn(param_np.shape[0], 1)
                        V = np.random.randn(1, param_np.shape[1])
                        low_rank_noise = U @ V * np.std(param_np) * destroy_factor * 2.0
                        param_np = param_np + low_rank_noise
                        param_data_float = torch.from_numpy(param_np).to(param.device)
                except Exception as e:
                    # If matrix operations fail, skip this step.
                    pass

        # If the original parameter was integer typed, convert it back.
        if is_integer:
            # Round to the nearest integer.
            param_data = param_data_float.round().to(param.dtype)
        else:
            param_data = param_data_float

        malicious_state[param_name] = param_data

    return malicious_state
# One malicious model generation method.
def weight_corruption_method(original_state, destruction_level):
    """
    Weight corruption method: directly corrupt weight parameters.
    """
    malicious_state = {}

    for param_name, param in original_state.items():
        param_data = param.clone()

        if param_data.numel() > 0:
            # Compute the corruption ratio from the destruction level.
            corruption_ratio = destruction_level

            if corruption_ratio > 0:
                # Flatten parameters.
                param_flat = param_data.view(-1)
                num_elements = param_flat.numel()

                # Compute the number of elements to corrupt.
                num_corrupt = int(num_elements * corruption_ratio)
                num_corrupt = max(1, min(num_elements, num_corrupt))

                # Randomly select positions to corrupt.
                corrupt_indices = random.sample(range(num_elements), num_corrupt)

                # Severely corrupt the selected positions.
                for idx in corrupt_indices:
                    # Combine multiple corruption modes.
                    rand_val = random.random()

                    if rand_val < 0.3:
                        # Set to an extreme value.
                        param_flat[idx] = param_data.mean().item() + param_data.std().item() * 10 * random.choice(
                            [-1, 1])
                    elif rand_val < 0.6:
                        # Set to 0.
                        param_flat[idx] = 0.0
                    else:
                        # Invert and amplify.
                        param_flat[idx] = -param_flat[idx] * (1.0 + destruction_level * 5.0)

                # Reshape back.
                param_data = param_flat.view(param.shape)

        malicious_state[param_name] = param_data

    return malicious_state

# One malicious model generation method.
def structure_attack_method(original_state, destruction_level):
    """
    Structure attack method: damage structural information in the network.
    """
    malicious_state = {}

    for param_name, param in original_state.items():
        param_data = param.clone()

        if len(param_data.shape) >= 2:  # Matrix parameter.
            # Damage matrix structure.
            if destruction_level > 0.5:
                # Add correlation corruption.
                try:
                    param_np = param_data.cpu().numpy()

                    # Damage correlations between rows.
                    for i in range(param_np.shape[0]):
                        for j in range(i + 1, min(i + 3, param_np.shape[0])):
                            if random.random() < destruction_level * 0.5:
                                # Make some rows similar or opposite.
                                if random.random() < 0.5:
                                    param_np[j] = param_np[i] * (1.0 + random.uniform(-0.5, 0.5))
                                else:
                                    param_np[j] = -param_np[i] * (1.0 + random.uniform(-0.5, 0.5))

                    param_data = torch.from_numpy(param_np).to(param.device)
                except:
                    pass

            # Add structural noise.
            structure_noise = torch.randn_like(param_data) * param_data.std().item() * destruction_level * 2.0
            param_data = param_data + structure_noise

        elif len(param_data.shape) == 1:  # Vector parameter.
            # Damage vector structure.
            if destruction_level > 0.3:
                # Create a destructive pattern.
                pattern = torch.sin(torch.arange(param_data.numel()).float() * 0.5) * 2.0
                pattern = pattern * param_data.std().item() * destruction_level
                param_data = param_data + pattern.to(param.device)

        malicious_state[param_name] = param_data

    return malicious_state

# One malicious model generation method.
def gradient_reverse_method(original_state, destruction_level):
    """
    Gradient reversal method: construct an adversarial-direction update.
    """
    malicious_state = {}

    for param_name, param in original_state.items():
        param_data = param.clone()

        if param_data.numel() > 0:
            # Approximate the gradient direction for the parameter.
            mean_val = param_data.mean().item()

            # Update parameters in the "wrong" direction.
            reverse_direction = -1.0  # Opposite to the normal gradient.

            # Determine the update magnitude from the destruction level.
            update_magnitude = param_data.std().item() * destruction_level * 3.0

            # Apply destructive updates.
            if 'weight' in param_name.lower():
                # Weights: stronger corruption.
                param_data = param_data + torch.randn_like(param_data) * update_magnitude * 1.5
                param_data = param_data * (1.0 - destruction_level * 0.5)  # Shrink weights.
            elif 'bias' in param_name.lower():
                # Bias: more complex corruption.
                param_data = param_data * (1.0 + random.uniform(-1, 1) * destruction_level)
                param_data = param_data + torch.randn_like(param_data) * update_magnitude
            else:
                # Other parameters.
                param_data = param_data * (1.0 + random.choice([-1, 1]) * destruction_level * 0.3)

        malicious_state[param_name] = param_data

    return malicious_state

# One malicious model generation method.
def random_chaos_method(original_state, destruction_level):
    """
    Fully random method: maximize random corruption.
    """
    malicious_state = {}

    for param_name, param in original_state.items():
        param_data = param.clone()

        if param_data.numel() > 0:
            # Completely shuffle parameters.
            if random.random() < destruction_level:
                if len(param_data.shape) == 1:
                    # One-dimensional parameter: fully random permutation.
                    indices = torch.randperm(param_data.numel())
                    param_data = param_data.view(-1)[indices].view(param.shape)
                elif len(param_data.shape) == 2:
                    # Two-dimensional parameter: shuffle both rows and columns.
                    row_indices = torch.randperm(param_data.shape[0])
                    col_indices = torch.randperm(param_data.shape[1])
                    param_data = param_data[row_indices, :][:, col_indices]

            # Add extreme noise.
            extreme_noise = torch.randn_like(param_data) * param_data.std().item() * 10.0 * destruction_level
            param_data = param_data + extreme_noise

            # Random scaling.
            random_scale = random.uniform(0.1, 10.0) if random.random() < 0.5 else 1.0
            param_data = param_data * random_scale

            # Random shift.
            random_shift = random.uniform(-5.0, 5.0) * param_data.std().item() if random.random() < 0.3 else 0.0
            param_data = param_data + random_shift

        malicious_state[param_name] = param_data

    return malicious_state


