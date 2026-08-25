import argparse
import json
import random

import torch

from client import Client
from server import Server
from Model_Encourage import compute_incentive_Si, IncentiveConfig
from Stackelberg import stackelberg_settle_from_model_encourage, print_stackelberg_round
import datasets
import Model_Verify


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Federated Learning')
    parser.add_argument('-c', '--conf', dest='conf')
    args = parser.parse_args()

    with open(args.conf, 'r') as f:
        conf = json.load(f)

    train_datasets, eval_datasets = datasets.get_dataset("./data/", conf["type"])

    server = Server(conf, eval_datasets)
    clients = []
    for c in range(conf["no_models"]):
        clients.append(Client(conf, server.global_model, train_datasets, c))

    print("\n\n")

    # Stackelberg 跨轮状态：上一轮结算输出的 effort_next 控制下一轮本地训练强度。
    stackelberg_game = None
    initial_effort = float(conf.get("initial_effort", 1.0))
    client_efforts = {str(c.client_id): initial_effort for c in clients}

    # 结构异常检测：跨轮保存每个客户端的后层参数快照。
    prev_late_state_storage = [None] * conf["no_models"]

    for e in range(conf["global_epochs"]):
        candidates = random.sample(clients, conf["k"])

        gradient_storage = [None] * conf["k"]
        model_storage = [None] * conf["k"]
        model_storage_represent = [None] * conf["k"]
        avg_loss_dict = [None] * conf["k"]
        diff_storage = [None] * conf["k"]
        round_efforts = {}

        weight_accumulator = {}
        for name, params in server.global_model.state_dict().items():
            weight_accumulator[name] = torch.zeros_like(params)

        # 本轮真实训练强度来自上一轮 Stackelberg 结算结果。
        for idx, c in enumerate(candidates):
            cid = str(c.client_id)
            effort = float(client_efforts.get(cid, initial_effort))
            round_efforts[cid] = effort

            diff, total_gradient_norm, local_model = c.local_train(server.global_model, effort=effort)
            model_storage[idx] = local_model
            diff_storage[idx] = diff
            gradient_storage[idx] = total_gradient_norm

        # 代表模型生成：仅融合 ResNet-18 后层（默认 layer4 + fc）。
        model_storage_represent = Model_Verify.create_representative_models_fuse_late_layers(
            model_storage,
            self_weight=0.5,
            late_layer_prefixes=("layer4", "fc"),
        )

        # 本轮设置一个恶意代表模型，用于验证评分和聚合降权效果。
        model_storage_represent[0] = Model_Verify.malicious_model_create(model_storage_represent[0])

        validation_models = Model_Verify.split_validation_models(clients)

        client_ids = [c.client_id for c in candidates]
        avg_loss_dict, prev_late_state_storage, _detail = Model_Verify.validate_models_lipc_ds_with_structure_anomaly(
            model_storage_represent,
            validation_models,
            current_local_models=model_storage,
            client_ids=client_ids,
            prev_late_state_storage=prev_late_state_storage,
            round_index=e,
            late_layer_prefixes=("layer4", "fc"),
            weight_ds=0.5,
            weight_struct=0.5,
            verbose=True,
        )

        # 多指标评分：除 ds_scores 外，其余指标可按配置生成样本值或由调用方传入真实值。
        result = compute_incentive_Si(
            client_ids,
            ds_scores=avg_loss_dict,
            config=IncentiveConfig(
                verbose=True,
                novelty_input="similarity",
            ),
            seed=None,
        )
        print("\n[Incentive Result] Si:", result["Si"])

        # 降权聚合：Si<=0 的客户端不参与本轮聚合，其余客户端按 Si 归一化加权。
        si_map = {str(k): float(v) for k, v in result["Si"].items()}
        positive_score_sum = sum(max(0.0, si_map.get(str(c.client_id), 0.0)) for c in candidates)
        if positive_score_sum > 0:
            for idx, c in enumerate(candidates):
                cid = str(c.client_id)
                agg_weight = max(0.0, si_map.get(cid, 0.0)) / positive_score_sum
                diff = diff_storage[idx]
                if diff is None:
                    continue
                for name, params in server.global_model.state_dict().items():
                    # state_dict 中包含 BatchNorm 的整型计数 buffer，只对浮点权重/浮点 buffer 做加权更新。
                    if not params.is_floating_point():
                        continue
                    weight_accumulator[name].add_(diff[name].to(params.dtype) * agg_weight)
        else:
            print("[Aggregation] 本轮没有正向评分客户端，保持全局模型不变。")

        # Stackelberg 结算：更新声誉、奖励，并输出下一轮训练强度。
        stack_out, stackelberg_game = stackelberg_settle_from_model_encourage(
            result,
            game=stackelberg_game,
            realized_efforts=round_efforts,
        )
        for cid, effort in (stack_out.get("effort_next") or {}).items():
            client_efforts[str(cid)] = float(effort)
        print_stackelberg_round(stack_out, round_title=f"[Stackelberg] Epoch={e + 1}", per_client=True, show_maps=False)

        server.model_aggregate(weight_accumulator)

        acc, loss = server.model_eval()
        print("AllEpoch %d, acc: %f, loss: %f\n" % (e, acc, loss))
