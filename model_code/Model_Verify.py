from random import random

import numpy as np
import torch
from numpy.ma import copy
import torch.nn.functional as F
import copy
import random

# 生成代表模型
def generate_representative_models(model_storage):
    num_models = len(model_storage)
    represent_model_storage = []

    for i in range(num_models):
        # 1. 创建新模型（复制当前模型的结构和初始参数）
        represent_model = type(model_storage[i])()  # 创建同类型空模型
        represent_model.load_state_dict(model_storage[i].state_dict())  # 初始化参数

        # 2. 设置权重：当前模型50%，其他模型平均分配剩余的50%
        weights = [0.0] * num_models
        weights[i] = 0.5  # 自身权重50%

        remaining_weight = 0.5  # 剩余50%的权重
        for j in range(num_models):
            if j != i:  # 其他模型的权重
                weights[j] = remaining_weight / (num_models - 1)  # 剩余权重平均分配给其他模型

        # 3. 获取新模型的 state_dict（用于存储融合后的参数）
        blended_state_dict = represent_model.state_dict()

        # 4. 逐层融合参数
        for name, param in blended_state_dict.items():
            if not param.is_floating_point():
                blended_state_dict[name] = model_storage[i].state_dict()[name].clone()
                continue
            blended_param = torch.zeros_like(param)
            for j in range(num_models):
                # 加权累加所有模型的参数
                blended_param += weights[j] * model_storage[j].state_dict()[name]
            blended_state_dict[name] = blended_param

        # 5. 更新代表模型参数
        represent_model.load_state_dict(blended_state_dict)
        represent_model_storage.append(represent_model)

        print(f"生成代表模型 {i+1}/{num_models}（基础模型={i}，自身权重=50%）")

    return


# 生成代表模型
def create_representative_models(local_models):
    """
    高效版本的代表模型生成

    优化点：
    1. 预先缓存所有模型的状态
    2. 批量计算融合
    """
    num_models = len(local_models)
    self_weight = 0.5

    if num_models == 0:
        return []

    if not (0 <= self_weight <= 1):
        raise ValueError(f"自身权重必须在0-1之间，当前为: {self_weight}")

    # 计算权重
    other_weight = (1.0 - self_weight) / (num_models - 1) if num_models > 1 else 0.0

    # 1. 预先获取所有模型的状态
    print("正在加载所有模型状态...")
    model_states = [model.state_dict() for model in local_models]

    # 2. 获取参数名称列表
    param_names = list(model_states[0].keys())

    # 3. 创建代表模型
    representative_models = []

    for i in range(num_models):
        print(f"\n融合生成代表模型 {i}...")

        # 深拷贝一个模型作为模板
        rep_model = copy.deepcopy(local_models[i])
        fused_state = {}

        # 对每个参数进行融合
        for name in param_names:
            if not model_states[i][name].is_floating_point():
                fused_state[name] = model_states[i][name].clone()
                continue
            # 当前模型的贡献
            fused_param = self_weight * model_states[i][name]

            # 其他模型的贡献
            for j in range(num_models):
                if j != i:
                    fused_param += other_weight * model_states[j][name]

            fused_state[name] = fused_param

        # 加载融合后的参数
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
    """只融合后层(ResNet-18推荐: layer4 + fc)的代表模型生成。

    设计目的：
    - 保留每个客户端的前层特征提取能力（不做参数平均，避免全层融合不稳定）；
    - 仅在判别更敏感的后层形成“共识”，更适合恶意更新检测/验证。

    参数:
        local_models: 本轮参与的本地模型列表（已训练后的模型）
        self_weight: 自身权重（0~1），其余权重均分到其他客户端
        late_layer_prefixes: 需要融合的参数名前缀集合。
            - 对 torchvision ResNet-18，默认 ("layer4", "fc")。
            - 若你想把 layer3 也纳入后层，可传 ("layer3", "layer4", "fc")。

    返回:
        representative_models: 代表模型列表，与 local_models 等长。
    """

    num_models = len(local_models)
    if num_models == 0:
        return []
    if not (0.0 <= float(self_weight) <= 1.0):
        raise ValueError(f"自身权重必须在0-1之间，当前为: {self_weight}")
    if num_models == 1:
        # 只有一个客户端时，无需融合
        rep_model = copy.deepcopy(local_models[0])
        rep_model.eval()
        return [rep_model]

    other_weight = (1.0 - float(self_weight)) / (num_models - 1)
    prefixes = tuple(str(p) for p in late_layer_prefixes)

    print("正在加载所有模型状态(后层融合版)...")
    model_states = [model.state_dict() for model in local_models]
    param_names = list(model_states[0].keys())

    def _is_late_param(name: str) -> bool:
        return any(name.startswith(prefix) for prefix in prefixes)

    representative_models = []
    for i in range(num_models):
        print(f"\n融合生成代表模型 {i} (仅融合后层: {prefixes}) ...")

        rep_model = copy.deepcopy(local_models[i])
        fused_state = {}

        # 仅对后层参数做融合；其余参数保持客户端自身
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
    """从 state_dict 中提取指定后层前缀的浮点张量(用于跨轮差异分析)。

    返回：dict[name] = cpu_float_tensor_clone
    - 只保留浮点类型，避免 running stats 的整型项无法做差。
    - 一律拷贝到 CPU，减小显存占用并便于跨轮保存。
    """

    prefixes = tuple(str(p) for p in late_layer_prefixes)
    # late_state 用来保存“后层参数快照”（不是差分），用于下一轮计算 Δw
    late_state = {}
    for name, tensor in state_dict.items():
        if not any(name.startswith(p) for p in prefixes):
            continue
        if not torch.is_tensor(tensor):
            continue
        if not tensor.is_floating_point():
            continue
        # 统一保存到 CPU：
        # 1) 避免 GPU 显存被跨轮快照吃满
        # 2) 下一轮只需要做数值差分/相似度，CPU 足够
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
    """把“本轮参与客户端”的后层参数快照写入 prev_late_state_storage。

    prev_late_state_storage 设计为数组(list)：
        - 下标 = client_id
        - 内容 = dict(参数名 -> CPU浮点张量)

    这能确保即使每轮 candidates 随机变化，我们也能按 client_id 做跨轮对齐。
    """

    if prev_late_state_storage is None:
        raise ValueError("prev_late_state_storage 不能为空；请在 main.py 初始化为长度=conf['no_models'] 的数组")
    if len(current_local_models) != len(client_ids):
        raise ValueError("current_local_models 与 client_ids 长度不一致")

    # updated 仅用于打印：本轮我们更新了多少个 client_id 的快照
    updated = 0
    for model, cid in zip(current_local_models, client_ids):
        if cid is None:
            continue
        snap = _extract_late_layer_float_tensors_from_state(
            model.state_dict(),
            late_layer_prefixes=late_layer_prefixes,
        )
        if 0 <= int(cid) < len(prev_late_state_storage):
            # 核心：prev_late_state_storage 是“跨轮记忆”，按 client_id 写入
            prev_late_state_storage[int(cid)] = snap
            updated += 1

    if verbose:
        prefixes = tuple(str(p) for p in late_layer_prefixes)
        print(f"\n[结构异常检测] 已更新上一轮快照: {updated} 个客户端")
        print(f"[结构异常检测] prev_late_state_storage 存储内容: prev_late_state_storage[client_id] = {{后层参数名 -> CPU浮点张量}}")
        print(f"[结构异常检测] 后层前缀: {prefixes}")

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
    """基于“同一客户端跨轮后层参数变化(Δw)”计算结构异常分数。

    核心思想（简洁但够论文写）：
    - 对每个客户端 i，取后层参数快照 w_i^{t-1} 与本轮 w_i^t，计算 Δw_i = w_i^t - w_i^{t-1}
    - 以本轮参与客户端的 Δw 集合构造群体中心 Δw̄（这里用均值中心）
    - 两个证据：
        1) 方向一致性：cos(Δw_i, Δw̄)
        2) 幅值异常性：||Δw_i|| 相对群体中位数/ MAD 的偏离
    - 输出 struct_score_i ∈ [0,1]，越大越“结构正常”。

    约束：
    - 第 0 轮(第一轮)默认结构差异正常，struct_score 全为 1.0（并写入快照供下一轮分析）。
    """

    num_models = len(current_local_models)
    if num_models == 0:
        return [], prev_late_state_storage, {}
    if len(client_ids) != num_models:
        raise ValueError("current_local_models 与 client_ids 长度不一致")
    if prev_late_state_storage is None:
        raise ValueError("prev_late_state_storage 不能为空")

    prefixes = tuple(str(p) for p in late_layer_prefixes)

    # round_index<=0（第一轮）：
    # - 还没有上一轮可对比，因此默认所有客户端结构差异“正常”(1.0)
    # - 但要把本轮后层参数快照写入 prev_late_state_storage，为下一轮准备
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
            "note": "round_index<=0: 默认结构差异正常(1.0)，仅存储快照用于下一轮",
        }
        return struct_scores, prev_late_state_storage, detailed

    # 1) 取出本轮后层 state（快照 w_i^t）
    current_late_states = []
    for model in current_local_models:
        current_late_states.append(
            _extract_late_layer_float_tensors_from_state(
                model.state_dict(),
                late_layer_prefixes=late_layer_prefixes,
            )
        )

    # 2) 构造 Δw_i = w_i^t - w_i^{t-1}
    #    注意：只对“上一轮存在快照”的客户端计算（valid_mask=True）
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
        # shared_names：本轮与上一轮都存在的后层参数名（可安全做差）
        shared_names = [n for n in curr_state.keys() if n in prev_state]
        if not shared_names:
            continue

        d = {}
        for n in shared_names:
            d[n] = curr_state[n] - prev_state[n]
        delta_states[i] = d
        valid_mask[i] = True

    # 如果本轮没人能算 Δw（比如：本轮参与的客户端上轮没参与）
    # 就默认结构分数=1.0，但仍更新快照，保证后续轮次可用
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
            "note": "无可用历史快照(可能是参与客户端变化/首次参与)：本轮结构分数默认 1.0，并更新快照",
        }
        return struct_scores, prev_late_state_storage, detailed

    # 3) 计算群体中心 Δw̄
    # 这里按“参数逐元素均值”做中心，而不是把所有参数拼接成一个超长向量，
    # 这样更省内存，也更直观。
    center_delta = {}
    # 取任意一个 valid 的参数集合做基准
    base_i = next(i for i, ok in enumerate(valid_mask) if ok)
    base_names = list(delta_states[base_i].keys())

    for name in base_names:
        # 仅对 valid 客户端里“确实包含该参数”的项参与均值（更稳健）
        tensors = []
        for i in range(num_models):
            if not valid_mask[i]:
                continue
            if name not in delta_states[i]:
                continue
            tensors.append(delta_states[i][name])
        if tensors:
            center_delta[name] = torch.stack(tensors, dim=0).mean(dim=0)

    # 4) 计算每个客户端结构指标
    # - cos(Δw_i, Δw̄)：方向是否与群体一致（越一致越正常）
    # - ||Δw_i||：幅值是否异常（太大或太小都可疑）
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

    # 幅值异常：用 median + MAD（中位数绝对偏差）估计群体尺度
    # MAD 比均值/方差对极端值更鲁棒，适合“可能存在恶意客户端”的场景
    valid_norms = [norms[i] for i in range(num_models) if valid_mask[i]]
    med = float(np.median(valid_norms))
    mad = float(np.median([abs(x - med) for x in valid_norms]))
    mad = max(mad, 1e-6)

    struct_scores = [1.0] * num_models
    for i in range(num_models):
        if not valid_mask[i]:
            struct_scores[i] = 1.0
            continue

        # cos 从 [-1,1] 映射到 [0,1]
        cos01 = max(0.0, min(1.0, (cosines[i] + 1.0) * 0.5))
        # z 越大，表示与群体中位数偏离越大；用 exp(-z) 映射到 (0,1]
        z = abs(norms[i] - med) / mad
        norm_score = float(np.exp(-z))
        norm_score = max(0.0, min(1.0, norm_score))

        # 结构分数：方向(50%) + 幅值(50%)
        struct_scores[i] = 0.5 * cos01 + 0.5 * norm_score

    if verbose:
        print(f"\n[结构异常检测] round={int(round_index)} 后层Δw结构分数(越大越正常)")
        print(f"[结构异常检测] 使用后层前缀: {prefixes}")
        print(f"[结构异常检测] 幅值统计: median(||Δw||)={med:.6f}, MAD={mad:.6f}")
        for i in range(num_models):
            if valid_mask[i]:
                print(
                    f"  idx={i} client_id={client_ids[i]} struct={struct_scores[i]:.4f} "
                    f"cos={cosines[i]:.4f} norm={norms[i]:.6f}"
                )
            else:
                print(f"  idx={i} client_id={client_ids[i]} struct=1.0000 (无历史快照)")

    # 5) 无论本轮结构分数如何，都要把“本轮后层快照”存起来，供下一轮对比
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
    """最终评分 = 0.5 * D-S分数 + 0.5 * 结构异常分数。

    - D-S分数：复用 validate_models_lipc_ds 的计算方式（准确率+损失证据）
    - 结构异常分数：跨轮后层差异 Δw 的方向一致性 + 幅值异常
    """

    if abs(float(weight_ds) + float(weight_struct) - 1.0) > 1e-8:
        s = float(weight_ds) + float(weight_struct)
        weight_ds = float(weight_ds) / s
        weight_struct = float(weight_struct) / s

    # 先算结构异常分数（需要跨轮存储，因此会返回更新后的 prev_late_state_storage）
    struct_scores, prev_late_state_storage, struct_detail = compute_structural_anomaly_scores_cross_round(
        current_local_models,
        client_ids,
        prev_late_state_storage,
        round_index=int(round_index),
        late_layer_prefixes=late_layer_prefixes,
        verbose=verbose,
    )

    # 再算 D-S 证据融合得分（沿用你原来的 624-864 那套逻辑）
    ds_scores = validate_models_lipc_ds(represent_models, validation_models)

    if len(ds_scores) != len(struct_scores):
        raise ValueError("ds_scores 与 struct_scores 长度不一致")

    # 最终评分：两者线性融合（默认 0.5 / 0.5）
    final_scores = [
        float(weight_ds) * float(ds_scores[i]) + float(weight_struct) * float(struct_scores[i])
        for i in range(len(ds_scores))
    ]

    if verbose:
        print(f"\n[最终评分] final = {weight_ds:.2f}*DS + {weight_struct:.2f}*Struct")
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


# # 取一半数量的本地客户端作为验证数据集
# def split_validation_models(clients):
#     num_models = len(clients)
#     num_validation = num_models // 2  # 严格取前一半
#
#     validation_models = []
#     for i in range(num_validation):
#         original_model = clients[i]
#
#         # 核心方法：通过state_dict完全复制模型
#         new_state_dict = {
#             name: param.clone()  # 显式克隆张量，确保内存独立
#             for name, param in original_model.state_dict().items()
#         }
#
#         # 创建新模型方案（无需调用__init__）
#         new_model = original_model.__class__.__new__(original_model.__class__)
#         if hasattr(new_model, '_apply'):
#             new_model._apply(lambda t: t)  # 触发基础初始化
#
#         # 加载复制的参数
#         new_model.load_state_dict(new_state_dict, strict=True)
#         validation_models.append(new_model)
#
#     print(f"已生成验证模型（共 {num_validation} 个）")
#     return validation_models

#构建验证数据集
def split_validation_models(model_storage):
    num_models = len(model_storage)
    num_validation = num_models // 2  # 严格取前一半

    validation_models = []
    for i in range(num_validation):
        validation_models.append(model_storage[i])

    print(f"已随机抽取验证数据集（共 {num_validation} 个）")
    return validation_models

#模型验证LIPC损失偏差
def validate_models_loss(represent_models, validation_models):
    num_models = len(represent_models)
    representative_loss = [None] * num_models

    for i in range(num_models):
        model = represent_models[i]  # 代表模型
        model.eval()  # 切换到评估模式
        total_loss = 0.0
        total_samples = 0  # 用于计算平均 loss

        with torch.no_grad():  # 不计算梯度，节省内存
            for batch_id, batch in enumerate(validation_models[0].train_loader):
                data, target = batch

                if torch.cuda.is_available():
                    data = data.cuda()
                    target = target.cuda()

                output = model(data)  # 用代表模型计算输出
                batch_loss = torch.nn.functional.cross_entropy(output, target, reduction='sum').item()
                total_loss += batch_loss
                total_samples += len(data)  # 统计总样本数

        # 计算平均 loss（可选）
        representative_loss[i] = total_loss / total_samples if total_samples > 0 else 0.0

    print("---------代表模型LIPC指标计算完毕-------------")

    for i, loss in enumerate(representative_loss):
        print(f"模型 {i} 的 LIPC: {loss}")

    return representative_loss

#模型验证LIPC准确率
def validate_models_acc(represent_models, validation_models):
    """
    计算代表模型在验证集上的LIPC指标（准确率）

    参数:
        represent_models: 代表模型列表
        validation_models: 验证模型列表，每个包含train_loader

    返回:
        每个代表模型的LIPC（准确率）列表
    """
    num_models = len(represent_models)
    representative_lipc = [0.0] * num_models  # 存储LIPC指标（准确率）

    for i in range(num_models):
        model = represent_models[i]  # 代表模型
        model.eval()  # 切换到评估模式

        total_correct = 0
        total_samples = 0

        with torch.no_grad():  # 不计算梯度，节省内存
            # 遍历所有验证模型的train_loader
            for val_model in validation_models:
                for batch_id, batch in enumerate(val_model.train_loader):
                    data, target = batch


                    if torch.cuda.is_available():
                        data = data.cuda()
                        target = target.cuda()

                    output = model(data)  # 用代表模型计算输出

                    # 获取预测结果
                    pred = output.data.max(1)[1]  # 获取最大概率的索引（预测类别）

                    # 统计正确的预测数量
                    correct = pred.eq(target.data.view_as(pred)).cpu().sum().item()

                    total_correct += correct
                    total_samples += len(data)

        # 计算LIPC指标（准确率）
        if total_samples > 0:
            lipc = 100.0 * (float(total_correct) / float(total_samples))  # 百分比形式
            # 或者如果你想要0-1之间的值：
            # lipc = float(total_correct) / float(total_samples)
        else:
            lipc = 0.0

        representative_lipc[i] = lipc

        # 打印每个验证模型的统计信息（可选）
        print(f"代表模型 {i} 在 {len(validation_models)} 个验证集上测试:")
        print(f"  总样本数: {total_samples}, 正确预测: {total_correct}")

    print("\n" + "=" * 50)
    print("代表模型LIPC指标计算完毕")
    print("=" * 50)

    for i, lipc in enumerate(representative_lipc):
        print(f"模型 {i} 的 LIPC: {lipc:.2f}%")

    return representative_lipc


# 支持多种归一化LIPC
def validate_models_lipc_advanced(represent_models, validation_models):
    """
    高级版本的综合LIPC指标计算

    参数:
        represent_models: 代表模型列表
        validation_models: 验证模型列表
        accuracy_weight: 准确率权重
        loss_weight: 损失权重
        loss_norm_method: 损失归一化方法
            - 'range': 线性范围归一化
            - 'exp': 指数衰减 exp(-beta * loss)
            - 'sigmoid': sigmoid函数归一化
        loss_range: 损失范围，用于range方法
        beta: 衰减系数，用于exp方法

    返回:
        综合LIPC得分和详细结果
    """
    accuracy_weight = 0.5
    loss_weight = 0.5
    loss_norm_method = 'range'  # 'range', 'exp', 'sigmoid'
    loss_range = (0.1, 5.0)
    beta = 2.0
    # 验证权重
    if abs(accuracy_weight + loss_weight - 1.0) > 1e-8:
        total_weight = accuracy_weight + loss_weight
        accuracy_weight /= total_weight
        loss_weight /= total_weight

    num_models = len(represent_models)
    lipc_scores = [0.0] * num_models
    accuracies = [0.0] * num_models
    losses = [0.0] * num_models

    print(f"\nLIPC计算 (方法: {loss_norm_method})")

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

                    # 准确率
                    pred = output.argmax(dim=1)
                    correct = (pred == target).sum().item()
                    total_correct += correct

                    # 损失
                    batch_loss = F.cross_entropy(output, target, reduction='sum').item()
                    total_loss += batch_loss

                    total_samples += batch_size

        if total_samples > 0:
            # 计算准确率
            accuracy = total_correct / total_samples
            accuracies[i] = accuracy

            # 计算平均损失
            avg_loss = total_loss / total_samples
            losses[i] = avg_loss

            # 根据选择的方法计算损失得分
            if loss_norm_method == 'range':
                # 线性范围归一化
                min_loss, max_loss = loss_range
                normalized_loss = (avg_loss - min_loss) / (max_loss - min_loss) if max_loss > min_loss else 0
                normalized_loss = max(0.0, min(1.0, normalized_loss))
                loss_score = 1.0 - normalized_loss

            elif loss_norm_method == 'exp':
                # 指数衰减
                loss_score = np.exp(-beta * avg_loss)

            elif loss_norm_method == 'sigmoid':
                # Sigmoid函数归一化
                # 假设损失在2附近是分界点
                loss_score = 1.0 / (1.0 + np.exp(avg_loss - 2.0))

            else:
                raise ValueError(f"未知的归一化方法: {loss_norm_method}")

            # 计算综合LIPC
            lipc_scores[i] = (accuracy_weight * accuracy) + (loss_weight * loss_score)

    # 返回结果
    # return lipc_scores, accuracies, losses

    # 返回结果
    return lipc_scores

# 支持多种归一化LIPC(是上一个同名函数的延申版 不用管上一个同名函数)
def validate_models_lipc_advanced(represent_models, validation_models):
    """
    高级版本的综合LIPC指标计算

    参数:
        represent_models: 代表模型列表
        validation_models: 验证模型列表
        accuracy_weight: 准确率权重
        loss_weight: 损失权重
        loss_norm_method: 损失归一化方法
            - 'range': 线性范围归一化
            - 'exp': 指数衰减 exp(-beta * loss)
            - 'sigmoid': sigmoid函数归一化
        loss_range: 损失范围，用于range方法
        beta: 衰减系数，用于exp方法

    返回:
        综合LIPC得分和详细结果
    """
    accuracy_weight = 0.5
    loss_weight = 0.5
    loss_norm_method = 'range'  # 'range', 'exp', 'sigmoid'
    loss_range = (0.1, 5.0)
    beta = 2.0

    # 验证权重
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
    print(f"代表模型LIPC指标计算开始")
    print(f"准确率权重: {accuracy_weight:.2f}, 损失权重: {loss_weight:.2f}")
    print(f"损失归一化方法: {loss_norm_method}")
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

                    # 准确率
                    pred = output.argmax(dim=1)
                    correct = (pred == target).sum().item()
                    total_correct += correct

                    # 损失
                    batch_loss = F.cross_entropy(output, target, reduction='sum').item()
                    total_loss += batch_loss

                    total_samples += batch_size

        if total_samples > 0:
            # 计算准确率
            accuracy = total_correct / total_samples
            accuracies[i] = accuracy

            # 计算平均损失
            avg_loss = total_loss / total_samples
            losses[i] = avg_loss

            # 保存样本信息
            total_samples_list[i] = total_samples
            total_correct_list[i] = total_correct

            # 根据选择的方法计算损失得分
            if loss_norm_method == 'range':
                # 线性范围归一化
                min_loss, max_loss = loss_range
                if max_loss <= min_loss:
                    print(f"警告: loss_range无效: {loss_range}, max_loss应大于min_loss")
                    loss_score = 1.0 if avg_loss <= min_loss else 0.0
                else:
                    normalized_loss = (avg_loss - min_loss) / (max_loss - min_loss)
                    normalized_loss = max(0.0, min(1.0, normalized_loss))
                    loss_score = 1.0 - normalized_loss

            elif loss_norm_method == 'exp':
                # 指数衰减
                loss_score = np.exp(-beta * avg_loss)

            elif loss_norm_method == 'sigmoid':
                # Sigmoid函数归一化
                # 假设损失在2附近是分界点
                loss_score = 1.0 / (1.0 + np.exp(avg_loss - 2.0))

            else:
                raise ValueError(f"未知的归一化方法: {loss_norm_method}")

            loss_scores[i] = loss_score

            # 计算综合LIPC
            lipc_scores[i] = (accuracy_weight * accuracy) + (loss_weight * loss_score)

            # 计算贡献度
            accuracy_contribution = accuracy_weight * accuracy
            loss_contribution = loss_weight * loss_score

            # 打印每个模型的详细结果
            print(f"\n{'─' * 60}")
            print(f"模型 {i} 详细结果:")
            print(f"{'─' * 60}")
            print(f"  样本数: {total_samples}, 正确数: {total_correct}")
            print(f"  准确率: {accuracy:.4f} ({accuracy * 100:.2f}%)")
            print(f"  平均损失: {avg_loss:.4f}")
            print(f"  损失归一化得分: {loss_score:.4f}")
            print(f"  → 综合LIPC: {lipc_scores[i]:.4f}")
            print(f"    (准确率贡献: {accuracy_contribution:.4f}, 损失贡献: {loss_contribution:.4f})")

        else:
            print(f"\n模型 {i}: 无有效样本数据")
            lipc_scores[i] = 0.0
            accuracies[i] = 0.0
            losses[i] = 0.0
            loss_scores[i] = 0.0

    # 打印汇总表格
    print(f"\n{'=' * 80}")
    print(f"代表模型LIPC指标汇总")
    print(f"{'=' * 80}")
    print(f"{'模型':<8} {'LIPC得分':<10} {'准确率':<10} {'平均损失':<10} {'损失得分':<10} {'样本数':<10} {'正确数':<10}")
    print(f"{'─' * 80}")

    for i in range(num_models):
        print(f"模型{i:<4}  {lipc_scores[i]:<10.4f}  {accuracies[i]:<10.4f}  "
              f"{losses[i]:<10.4f}  {loss_scores[i]:<10.4f}  "
              f"{total_samples_list[i]:<10}  {total_correct_list[i]:<10}")

    # 模型排名
    if num_models > 0 and any(lipc_scores):
        sorted_indices = sorted(range(num_models), key=lambda i: lipc_scores[i], reverse=True)

        print(f"\n{'=' * 80}")
        print(f"模型排名 (按LIPC得分降序)")
        print(f"{'=' * 80}")

        for rank, idx in enumerate(sorted_indices, 1):
            star = "★" if rank == 1 else ""
            print(f"{rank:2d}. 模型{idx} {star:2s} LIPC: {lipc_scores[idx]:.4f}, "
                  f"准确率: {accuracies[idx]:.4f} ({accuracies[idx] * 100:.2f}%), "
                  f"损失: {losses[idx]:.4f}")

        # 最佳和最差模型
        best_idx = sorted_indices[0]
        worst_idx = sorted_indices[-1]

        print(f"\n{'=' * 80}")
        print(f"最佳模型分析")
        print(f"{'=' * 80}")
        print(f"模型: 模型{best_idx}")
        print(f"  LIPC得分: {lipc_scores[best_idx]:.4f}")
        print(f"  准确率: {accuracies[best_idx]:.4f} ({accuracies[best_idx] * 100:.2f}%)")
        print(f"  平均损失: {losses[best_idx]:.4f}")
        print(f"  损失归一化得分: {loss_scores[best_idx]:.4f}")
        print(f"  样本数: {total_samples_list[best_idx]}")
        print(f"  正确数: {total_correct_list[best_idx]}")


    print(f"\n{'=' * 80}")
    print(f"代表模型LIPC指标计算完成")
    print(f"{'=' * 80}")

    # 返回详细结果字典
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

# D-S证据理论融合（准确率 + 损失/梯度证据）
def validate_models_lipc_ds(
    represent_models,
    validation_models,
):
    """
    使用 Dempster–Shafer 证据理论把两个证据源（准确率, 损失归一化得分）融合为可信度分数。

    参数:
        represent_models: 代表模型列表
        validation_models: 验证模型列表，每个包含 train_loader
        loss_norm_method: 损失归一化方法 ('range','exp','sigmoid')
        loss_range: range 方法的 (min, max)
        beta: exp 方法的衰减系数
        tau: 控制样本数到证据可靠性映射的平滑系数

    返回:
        scores: 每个代表模型的 D-S 融合分数（越大越可信）
        若 return_detailed=True，则额外返回 detailed 字典（包含 mG, mB, K, BetP 等，便于分析）
    """
    loss_norm_method = 'range'
    loss_range = (0.1, 5.0)
    beta = 2.0
    tau = 100.0
    verbose = True
    return_detailed = False
    num_models = len(represent_models)
    scores = [0.0] * num_models
    # 存储用于调试的数据
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
        print("代表模型 D-S 证据融合评分计算开始")
        print("证据源: (1) 准确率 (2) 损失归一化得分")
        print(f"loss_norm_method: {loss_norm_method}, loss_range: {loss_range}, beta: {beta}")
        print(f"可靠性映射: alpha = 1 - exp(-n/tau), tau: {tau}")
        print(f"输出分数: score = BetP(G) * (1 - K)")
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
            # 无样本，标为不确定
            detailed['accuracies'][i] = 0.0
            detailed['losses'][i] = 0.0
            detailed['loss_scores'][i] = 0.0
            scores[i] = 0.0
            detailed['mTheta'][i] = 1.0
            detailed['score'][i] = 0.0
            if verbose:
                print(f"\n模型 {i}: 无有效样本数据")
            continue

        # 基本数值
        accuracy = total_correct / total_samples
        avg_loss = total_loss / total_samples
        detailed['accuracies'][i] = accuracy
        detailed['losses'][i] = avg_loss

        # 损失归一化（与 validate_models_lipc_advanced 保持一致的选项）
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
            raise ValueError(f"未知的归一化方法: {loss_norm_method}")

        detailed['loss_scores'][i] = loss_score

        # 把数值映射为证据强度 s in [0,1]
        s_a = float(np.clip(accuracy, 0.0, 1.0))
        s_g = float(np.clip(loss_score, 0.0, 1.0))

        # 可靠性 alpha：以样本量为依据，自适应设定
        alpha_a = 1.0 - np.exp(- total_samples / (tau + eps))
        alpha_g = alpha_a
        detailed['alpha_a'][i] = float(alpha_a)
        detailed['alpha_g'][i] = float(alpha_g)

        # 构造每个证据源的 BBA
        m_aG = alpha_a * s_a
        m_aB = alpha_a * (1.0 - s_a)
        m_aTheta = 1.0 - alpha_a

        m_gG = alpha_g * s_g
        m_gB = alpha_g * (1.0 - s_g)
        m_gTheta = 1.0 - alpha_g

        # 冲突系数 K
        K = m_aG * m_gB + m_aB * m_gG

        # 合并（两条证据的情形下的封闭形式展开）
        if 1.0 - K <= eps:
            # 极端冲突: 退化到完全不确定
            mG = 0.0
            mB = 0.0
            mTheta = 1.0
        else:
            norm = 1.0 / (1.0 - K)
            mG = norm * (m_aG * m_gG + m_aG * m_gTheta + m_aTheta * m_gG)
            mB = norm * (m_aB * m_gB + m_aB * m_gTheta + m_aTheta * m_gB)
            mTheta = norm * (m_aTheta * m_gTheta)

        # Pignistic probability（赌徒概率），将不确定质量平均分配
        BetP_G = mG + 0.5 * mTheta

        # 最终分数：同时考虑 BetP 与冲突度（惩罚高冲突）
        score = float(BetP_G * (1.0 - K))

        # 保存结果
        detailed['mG'][i] = mG
        detailed['mB'][i] = mB
        detailed['mTheta'][i] = mTheta
        detailed['K'][i] = K
        detailed['BetP'][i] = BetP_G
        detailed['score'][i] = score
        scores[i] = score

        if verbose:
            print(f"\n{'─' * 60}")
            print(f"模型 {i} 详细结果 (D-S 融合)")
            print(f"{'─' * 60}")
            print(f"  样本数: {total_samples}, 正确数: {total_correct}")
            print(f"  准确率: {accuracy:.4f} ({accuracy * 100:.2f}%)")
            print(f"  平均损失: {avg_loss:.4f}")
            print(f"  损失归一化得分: {loss_score:.4f}")
            print(f"  可靠性: alpha_a={alpha_a:.4f}, alpha_g={alpha_g:.4f}")
            print(f"  BBA合成: m(G)={mG:.4f}, m(B)={mB:.4f}, m(Θ)={mTheta:.4f}")
            print(f"  冲突度: K={K:.4f}")
            print(f"  BetP(G)={BetP_G:.4f}")
            print(f"  → D-S融合分数: {score:.4f}")

    if verbose:
        print(f"\n{'=' * 80}")
        print("代表模型 D-S 融合评分汇总")
        print(f"{'=' * 80}")
        print(f"{'模型':<8} {'DS分数':<10} {'BetP(G)':<10} {'K':<10} {'准确率':<10} {'平均损失':<10} {'损失得分':<10} {'样本数':<10}")
        print(f"{'─' * 80}")

        for i in range(num_models):
            print(
                f"模型{i:<4}  {scores[i]:<10.4f}  {detailed['BetP'][i]:<10.4f}  {detailed['K'][i]:<10.4f}  "
                f"{detailed['accuracies'][i]:<10.4f}  {detailed['losses'][i]:<10.4f}  {detailed['loss_scores'][i]:<10.4f}  "
                f"{detailed['total_samples'][i]:<10}"
            )

        # 模型排名
        if num_models > 0 and any(scores):
            sorted_indices = sorted(range(num_models), key=lambda j: scores[j], reverse=True)

            print(f"\n{'=' * 80}")
            print("模型排名 (按D-S融合分数降序)")
            print(f"{'=' * 80}")

            for rank, idx in enumerate(sorted_indices, 1):
                star = "★" if rank == 1 else ""
                print(
                    f"{rank:2d}. 模型{idx} {star:2s} DS: {scores[idx]:.4f}, "
                    f"BetP(G): {detailed['BetP'][idx]:.4f}, K: {detailed['K'][idx]:.4f}, "
                    f"准确率: {detailed['accuracies'][idx]:.4f} ({detailed['accuracies'][idx] * 100:.2f}%), "
                    f"损失: {detailed['losses'][idx]:.4f}"
                )

            best_idx = sorted_indices[0]
            print(f"\n{'=' * 80}")
            print("最佳模型分析")
            print(f"{'=' * 80}")
            print(f"模型: 模型{best_idx}")
            print(f"  D-S融合分数: {scores[best_idx]:.4f}")
            print(f"  BetP(G): {detailed['BetP'][best_idx]:.4f}")
            print(f"  冲突度K: {detailed['K'][best_idx]:.4f}")
            print(f"  准确率: {detailed['accuracies'][best_idx]:.4f} ({detailed['accuracies'][best_idx] * 100:.2f}%)")
            print(f"  平均损失: {detailed['losses'][best_idx]:.4f}")
            print(f"  损失归一化得分: {detailed['loss_scores'][best_idx]:.4f}")
            print(f"  样本数: {detailed['total_samples'][best_idx]}")
            print(f"  正确数: {detailed['total_correct'][best_idx]}")

        print(f"\n{'=' * 80}")
        print("代表模型 D-S 证据融合评分计算完成")
        print(f"{'=' * 80}")

    if return_detailed:
        return scores, detailed

    return scores


#前50%模型筛选
def find_top_50_percent_models(lipc_scores):
    """
    找出LIPC评分数组中前50%最高分对应的模型下标

    参数:
        lipc_scores: LIPC评分列表

    返回:
        前50%模型的下标列表
    """
    if not lipc_scores:
        print("警告: LIPC评分数组为空")
        return []

    # 计算需要选择的模型数量（向上取整，确保至少选择一个）
    n = len(lipc_scores)
    k = max(1, (n + 1) // 2)  # 前50%，向上取整

    # 创建一个(评分, 下标)的元组列表
    scored_models = [(score, idx) for idx, score in enumerate(lipc_scores)]

    # 按评分降序排序
    scored_models.sort(key=lambda x: x[0], reverse=True)

    # 获取前k个模型的下标
    top_indices = [idx for score, idx in scored_models[:k]]

    # 对下标进行排序，使其按原始顺序排列
    top_indices.sort()

    # 格式化输出字符串
    if top_indices:
        indices_str = ','.join(str(idx) for idx in top_indices)
        print(f"前50%的良好模型下标依次为: {indices_str}")

        # 打印详细信息
        print(f"\n详细信息:")
        print(f"- 总模型数: {n}")
        print(f"- 选择前50%的模型数: {k}")
        print(f"- 前{k}个最高LIPC评分: {[scored_models[i][0] for i in range(k)]}")
        print(f"- 对应下标: {top_indices}")
    else:
        print("未找到符合条件的模型")

    return top_indices


def malicious_model_create(original_model, strength=0.8):
    """
    最简单的恶意模型创建方法
    通过直接操作状态字典，避免模型重建问题
    """
    print(f"创建简单恶意模型，强度: {strength}")

    # 获取原始模型的状态字典
    original_state = original_model.state_dict()
    malicious_state = {}

    for name, param in original_state.items():
        param_data = param.clone()

        # 只处理浮点数参数
        if param_data.is_floating_point() and param_data.numel() > 0:
            # 添加噪声
            noise_scale = param_data.std() * strength * 5.0
            noise = torch.randn_like(param_data) * noise_scale
            param_data = param_data + noise

            # 随机缩放（30%概率）
            if random.random() < 0.3:
                scale = random.uniform(0.1, 10.0)
                param_data = param_data * scale

            # 部分取反（20%概率）
            if random.random() < 0.2 and param_data.numel() > 10:
                num_invert = max(1, int(param_data.numel() * 0.1))
                indices = random.sample(range(param_data.numel()), num_invert)
                param_flat = param_data.view(-1)
                param_flat[indices] = -param_flat[indices] * 3.0

        malicious_state[name] = param_data

    # 现在关键步骤：我们不能直接创建新模型，而是复制原始模型然后加载参数
    # 使用深拷贝来复制模型
    malicious_model = copy.deepcopy(original_model)
    malicious_model.load_state_dict(malicious_state)
    malicious_model.eval()

    print("✅ 简单恶意模型创建完成")
    return malicious_model
#调用下面所有生成方法的恶意模型生成方法
def create_malicious_model(original_model):
    """
    在原始模型基础上创建恶意模型，显著降低模型准确率

    参数:
        original_model: 原始正常模型
        destruction_level: 破坏程度 (0-1, 1为完全破坏)
        method: 破坏方法
            - 'smart_destroy': 智能破坏（推荐）
            - 'weight_corruption': 权重破坏
            - 'structure_attack': 结构攻击
            - 'gradient_reverse': 梯度反转
            - 'random_chaos': 完全随机

    返回:
        malicious_model: 恶意模型
    """
    destruction_level = 0.8
    #选择恶意攻击方式
    method = 'smart_destroy'
    # 确保破坏程度在合理范围内
    destruction_level = max(0.0, min(1.0, destruction_level))

    print(f"\n正在创建恶意模型...")
    print(f"破坏程度: {destruction_level:.2f}")
    print(f"破坏方法: {method}")

    # 方法1: 使用深拷贝然后修改参数（最安全的方法）
    try:
        # 尝试深拷贝整个模型
        import copy
        malicious_model = copy.deepcopy(original_model)
        print("✅ 使用深拷贝复制模型结构")
    except Exception as e:
        # 如果深拷贝失败，使用方法2
        print(f"⚠️ 深拷贝失败: {e}")
        print("尝试通过state_dict复制模型...")

        # 方法2: 通过模型类名和参数重新创建
        model_class = type(original_model)

        # 尝试获取模型的初始化参数
        try:
            # 查看模型是否有特定的初始化参数
            if hasattr(original_model, '__init__'):
                # 获取模型的初始化签名
                import inspect
                init_signature = inspect.signature(model_class.__init__)
                params = init_signature.parameters

                # 尝试提取初始化参数
                init_args = {}
                for param_name in params:
                    if param_name != 'self' and hasattr(original_model, param_name):
                        init_args[param_name] = getattr(original_model, param_name)

                if init_args:
                    malicious_model = model_class(**init_args)
                    print(f"✅ 使用初始化参数重新创建模型: {init_args}")
                else:
                    # 如果没有找到参数，尝试默认创建
                    malicious_model = model_class()
                    print("✅ 使用默认构造函数创建模型")
            else:
                malicious_model = model_class()
                print("✅ 使用默认构造函数创建模型")
        except Exception as e2:
            print(f"❌ 无法重新创建模型: {e2}")
            print("尝试直接修改原始模型的状态字典...")

            # 方法3: 直接修改原始模型的状态字典（风险最高）
            malicious_model = original_model
            print("⚠️ 警告: 直接在原始模型上修改，请确保有备份")

    # 获取原始模型参数
    original_state = original_model.state_dict()

    # 根据不同的破坏方法创建恶意参数
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
        raise ValueError(f"未知的破坏方法: {method}")

    # 加载恶意参数到新模型
    malicious_model.load_state_dict(malicious_state)

    # 设置为评估模式
    malicious_model.eval()

    return malicious_model

#恶意模型生成方法之一
def smart_destroy_method(original_state, destruction_level):
    """
    智能破坏方法：有针对性地破坏关键参数
    """
    malicious_state = {}

    for param_name, param in original_state.items():
        param_data = param.clone()

        # 跳过非浮点数类型的参数（如整数类型的参数）
        if not param_data.is_floating_point():
            # 如果是整数类型，先转换为浮点数进行处理
            if param_data.dtype in [torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8]:
                param_data_float = param_data.float()
                is_integer = True
            else:
                # 其他非浮点类型，直接跳过
                malicious_state[param_name] = param_data
                continue
        else:
            param_data_float = param_data
            is_integer = False

        # 根据参数类型和重要性进行不同程度的破坏
        if 'weight' in param_name.lower():
            # 权重矩阵是神经网络的核心，重点破坏
            destroy_factor = destruction_level * 1.2
        elif 'bias' in param_name.lower():
            # 偏置项也重要，但程度稍轻
            destroy_factor = destruction_level * 0.8
        else:
            # 其他参数
            destroy_factor = destruction_level * 0.5

        # 确保不超过1
        destroy_factor = min(destroy_factor, 1.0)

        if destroy_factor > 0 and param_data_float.numel() > 0:
            # 计算统计信息，避免空张量
            try:
                param_std = param_data_float.std().item()
                param_mean = param_data_float.mean().item()
            except:
                # 如果计算std失败，使用默认值
                param_std = 1.0
                param_mean = 0.0

            # 方法1: 添加破坏性噪声
            noise_scale = param_std * destroy_factor * 3.0
            if noise_scale > 0:
                noise = torch.randn_like(param_data_float) * noise_scale
                param_data_float = param_data_float + noise

            # 方法2: 部分参数随机化
            if destroy_factor > 0.3 and param_data_float.numel() > 1:
                num_elements = param_data_float.numel()
                random_indices = random.sample(range(num_elements),
                                               min(int(num_elements * destroy_factor * 0.3), num_elements))
                param_flat = param_data_float.view(-1)
                for idx in random_indices:
                    param_flat[idx] = torch.randn(1).item() * param_std * 2.0

            # 方法3: 对权重矩阵进行特殊破坏
            if len(param_data_float.shape) >= 2 and destroy_factor > 0.5 and param_data_float.numel() > 1:
                # 尝试破坏矩阵的秩
                try:
                    param_np = param_data_float.cpu().numpy()
                    if param_np.size > 1 and param_np.shape[0] > 0 and param_np.shape[1] > 0:
                        # 添加低秩噪声破坏结构
                        U = np.random.randn(param_np.shape[0], 1)
                        V = np.random.randn(1, param_np.shape[1])
                        low_rank_noise = U @ V * np.std(param_np) * destroy_factor * 2.0
                        param_np = param_np + low_rank_noise
                        param_data_float = torch.from_numpy(param_np).to(param.device)
                except Exception as e:
                    # 如果矩阵操作失败，跳过
                    pass

        # 如果是整数类型，转换回整数
        if is_integer:
            # 四舍五入到最接近的整数
            param_data = param_data_float.round().to(param.dtype)
        else:
            param_data = param_data_float

        malicious_state[param_name] = param_data

    return malicious_state
#恶意模型生成方法之一
def weight_corruption_method(original_state, destruction_level):
    """
    权重破坏方法：直接破坏权重参数
    """
    malicious_state = {}

    for param_name, param in original_state.items():
        param_data = param.clone()

        if param_data.numel() > 0:
            # 根据破坏程度计算要破坏的比例
            corruption_ratio = destruction_level

            if corruption_ratio > 0:
                # 展平参数
                param_flat = param_data.view(-1)
                num_elements = param_flat.numel()

                # 计算要破坏的元素数量
                num_corrupt = int(num_elements * corruption_ratio)
                num_corrupt = max(1, min(num_elements, num_corrupt))

                # 随机选择要破坏的位置
                corrupt_indices = random.sample(range(num_elements), num_corrupt)

                # 对选中的位置进行严重破坏
                for idx in corrupt_indices:
                    # 多种破坏方式组合
                    rand_val = random.random()

                    if rand_val < 0.3:
                        # 设置为极端值
                        param_flat[idx] = param_data.mean().item() + param_data.std().item() * 10 * random.choice(
                            [-1, 1])
                    elif rand_val < 0.6:
                        # 设置为0
                        param_flat[idx] = 0.0
                    else:
                        # 取反并放大
                        param_flat[idx] = -param_flat[idx] * (1.0 + destruction_level * 5.0)

                # 重新整形
                param_data = param_flat.view(param.shape)

        malicious_state[param_name] = param_data

    return malicious_state

#恶意模型生成方法之一
def structure_attack_method(original_state, destruction_level):
    """
    结构攻击方法：破坏网络的结构性信息
    """
    malicious_state = {}

    for param_name, param in original_state.items():
        param_data = param.clone()

        if len(param_data.shape) >= 2:  # 矩阵参数
            # 破坏矩阵结构
            if destruction_level > 0.5:
                # 添加相关性破坏
                try:
                    param_np = param_data.cpu().numpy()

                    # 破坏行之间的相关性
                    for i in range(param_np.shape[0]):
                        for j in range(i + 1, min(i + 3, param_np.shape[0])):
                            if random.random() < destruction_level * 0.5:
                                # 使某些行相似或相反
                                if random.random() < 0.5:
                                    param_np[j] = param_np[i] * (1.0 + random.uniform(-0.5, 0.5))
                                else:
                                    param_np[j] = -param_np[i] * (1.0 + random.uniform(-0.5, 0.5))

                    param_data = torch.from_numpy(param_np).to(param.device)
                except:
                    pass

            # 添加结构噪声
            structure_noise = torch.randn_like(param_data) * param_data.std().item() * destruction_level * 2.0
            param_data = param_data + structure_noise

        elif len(param_data.shape) == 1:  # 向量参数
            # 破坏向量结构
            if destruction_level > 0.3:
                # 创建破坏性模式
                pattern = torch.sin(torch.arange(param_data.numel()).float() * 0.5) * 2.0
                pattern = pattern * param_data.std().item() * destruction_level
                param_data = param_data + pattern.to(param.device)

        malicious_state[param_name] = param_data

    return malicious_state

#恶意模型生成方法之一
def gradient_reverse_method(original_state, destruction_level):
    """
    梯度反转方法：构造对抗方向更新
    """
    malicious_state = {}

    for param_name, param in original_state.items():
        param_data = param.clone()

        if param_data.numel() > 0:
            # 计算参数的梯度方向（近似）
            mean_val = param_data.mean().item()

            # 沿着"错误"的方向更新参数
            reverse_direction = -1.0  # 与正常梯度相反

            # 根据破坏程度决定更新幅度
            update_magnitude = param_data.std().item() * destruction_level * 3.0

            # 应用破坏性更新
            if 'weight' in param_name.lower():
                # 权重：更大程度的破坏
                param_data = param_data + torch.randn_like(param_data) * update_magnitude * 1.5
                param_data = param_data * (1.0 - destruction_level * 0.5)  # 缩小权重
            elif 'bias' in param_name.lower():
                # 偏置：更复杂的破坏
                param_data = param_data * (1.0 + random.uniform(-1, 1) * destruction_level)
                param_data = param_data + torch.randn_like(param_data) * update_magnitude
            else:
                # 其他参数
                param_data = param_data * (1.0 + random.choice([-1, 1]) * destruction_level * 0.3)

        malicious_state[param_name] = param_data

    return malicious_state

#恶意模型生成方法之一
def random_chaos_method(original_state, destruction_level):
    """
    完全随机方法：最大程度的随机破坏
    """
    malicious_state = {}

    for param_name, param in original_state.items():
        param_data = param.clone()

        if param_data.numel() > 0:
            # 完全打乱参数
            if random.random() < destruction_level:
                if len(param_data.shape) == 1:
                    # 一维参数：完全随机排列
                    indices = torch.randperm(param_data.numel())
                    param_data = param_data.view(-1)[indices].view(param.shape)
                elif len(param_data.shape) == 2:
                    # 二维参数：行和列都打乱
                    row_indices = torch.randperm(param_data.shape[0])
                    col_indices = torch.randperm(param_data.shape[1])
                    param_data = param_data[row_indices, :][:, col_indices]

            # 添加极端噪声
            extreme_noise = torch.randn_like(param_data) * param_data.std().item() * 10.0 * destruction_level
            param_data = param_data + extreme_noise

            # 随机缩放
            random_scale = random.uniform(0.1, 10.0) if random.random() < 0.5 else 1.0
            param_data = param_data * random_scale

            # 随机偏移
            random_shift = random.uniform(-5.0, 5.0) * param_data.std().item() if random.random() < 0.3 else 0.0
            param_data = param_data + random_shift

        malicious_state[param_name] = param_data

    return malicious_state


