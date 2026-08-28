import torch
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def get_flatten_params(model):
    """Collect all trainable model parameters and flatten them into a one-dimensional vector."""
    return torch.cat([p.data.view(-1) for p in model.parameters() if p.requires_grad])


def pca_reduce_model_dicts(model_dicts, n_components=2):
    """
    Apply PCA dimensionality reduction to a list of model dictionaries with CUDA tensor support.
    Input: [{"layer1": params1, ...}, {"layer2": params2, ...}, ...] (5 models)
    Output: a list of dimension-reduced model dictionaries.
    """
    # 1. Convert model dictionaries into parameter vectors with CUDA tensor handling.
    param_vectors = []
    for model_dict in model_dicts:
        sorted_layers = sorted(model_dict.items(), key=lambda x: x[0])
        # Key change: add .cpu() to move data into host memory.
        vec = torch.cat([p.cpu().flatten() for _, p in sorted_layers]).numpy()
        param_vectors.append(vec)

    # 2. Build the parameter matrix (n_models x n_params).
    params_matrix = np.vstack(param_vectors)
    print(f"Input matrix shape: {params_matrix.shape}")

    # 3. Standardization and PCA.
    scaler = StandardScaler()
    params_scaled = scaler.fit_transform(params_matrix)
    pca = PCA(n_components=min(n_components, params_scaled.shape[1]))
    reduced = pca.fit_transform(params_scaled)

    # 4. Reconstruct parameters.
    reconstructed = scaler.inverse_transform(pca.inverse_transform(reduced))

    # 5. Rebuild model dictionaries while preserving the original device type.
    results = []
    for i, model_dict in enumerate(model_dicts):
        new_dict = {}
        start = 0
        for layer_name, original_params in sorted(model_dict.items(), key=lambda x: x[0]):
            param_size = original_params.numel()
            # Get the tensor's current device (GPU/CPU).
            device = original_params.device
            # Reconstruct parameters and move them back to the original device.
            new_params = torch.from_numpy(reconstructed[i, start:start + param_size])
            new_dict[layer_name] = new_params.reshape(original_params.shape).to(device)
            start += param_size
        results.append(new_dict)

    print(f"Dimensionality reduction complete, explained variance: {sum(pca.explained_variance_ratio_):.2%}")
    return results
