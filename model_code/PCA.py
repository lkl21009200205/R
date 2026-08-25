import torch
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def get_flatten_params(model):
    """获取模型所有可训练参数并展平为一维向量"""
    return torch.cat([p.data.view(-1) for p in model.parameters() if p.requires_grad])


def pca_reduce_model_dicts(model_dicts, n_components=2):
    """
    处理模型字典列表的PCA降维（兼容CUDA张量）
    输入: [{"layer1": params1, ...}, {"layer2": params2, ...}, ...] (5个模型)
    输出: 降维后的模型字典列表
    """
    # 1. 将模型字典转换为参数向量（自动处理CUDA张量）
    param_vectors = []
    for model_dict in model_dicts:
        sorted_layers = sorted(model_dict.items(), key=lambda x: x[0])
        # 关键修改：添加.cpu()转移数据到内存
        vec = torch.cat([p.cpu().flatten() for _, p in sorted_layers]).numpy()
        param_vectors.append(vec)

    # 2. 构建参数矩阵 (n_models × n_params)
    params_matrix = np.vstack(param_vectors)
    print(f"输入矩阵形状: {params_matrix.shape}")

    # 3. 标准化和PCA
    scaler = StandardScaler()
    params_scaled = scaler.fit_transform(params_matrix)
    pca = PCA(n_components=min(n_components, params_scaled.shape[1]))
    reduced = pca.fit_transform(params_scaled)

    # 4. 重构参数
    reconstructed = scaler.inverse_transform(pca.inverse_transform(reduced))

    # 5. 重建模型字典（保持原始设备类型）
    results = []
    for i, model_dict in enumerate(model_dicts):
        new_dict = {}
        start = 0
        for layer_name, original_params in sorted(model_dict.items(), key=lambda x: x[0]):
            param_size = original_params.numel()
            # 获取张量所在的设备（GPU/CPU）
            device = original_params.device
            # 重构参数并放回原始设备
            new_params = torch.from_numpy(reconstructed[i, start:start + param_size])
            new_dict[layer_name] = new_params.reshape(original_params.shape).to(device)
            start += param_size
        results.append(new_dict)

    print(f"降维完成，解释方差: {sum(pca.explained_variance_ratio_):.2%}")
    return results