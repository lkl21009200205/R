import math

import torch

import models


class Client(object):
    def __init__(self, conf, model, train_dataset, id=-1):
        self.conf = conf
        self.local_model = models.get_model(self.conf["model_name"])
        self.client_id = id
        self.train_dataset = train_dataset

        all_range = list(range(len(self.train_dataset)))
        data_len = int(len(self.train_dataset) / self.conf['no_models'])
        train_indices = all_range[id * data_len: (id + 1) * data_len]

        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=conf["batch_size"],
            sampler=torch.utils.data.sampler.SubsetRandomSampler(train_indices),
        )

    def local_train(self, model, effort=1.0):
        # effort is provided by the previous Stackelberg contract and reputation state to control this round's actual training intensity.
        try:
            effort = float(effort)
        except (TypeError, ValueError):
            effort = 1.0

        effort_floor = float(self.conf.get("effort_floor", 0.0))
        effort_cap = float(self.conf.get("effort_cap", 1.0))
        if effort_cap <= 0:
            effort_cap = 1.0
        effort_ratio = max(effort_floor, min(effort_cap, effort)) / effort_cap
        effort_ratio = max(0.0, min(1.0, effort_ratio))

        total_batches = len(self.train_loader)
        max_batches = 0
        if effort_ratio > 0 and total_batches > 0:
            max_batches = max(1, int(math.ceil(total_batches * effort_ratio)))

        for name, param in model.state_dict().items():
            self.local_model.state_dict()[name].copy_(param.clone())

        optimizer = torch.optim.SGD(
            self.local_model.parameters(),
            lr=self.conf['lr'],
            momentum=self.conf['momentum'],
        )

        self.local_model.train()
        gradients = {name: torch.zeros_like(param) for name, param in self.local_model.named_parameters()}

        for _ in range(self.conf["local_epochs"]):
            if not hasattr(self, 'gradients_history'):
                self.gradients_history = []

            for batch_id, batch in enumerate(self.train_loader):
                if batch_id >= max_batches:
                    break
                data, target = batch

                if torch.cuda.is_available():
                    data = data.cuda()
                    target = target.cuda()

                optimizer.zero_grad()
                output = self.local_model(data)
                loss = torch.nn.functional.cross_entropy(output, target)
                loss.backward()

                gradients = {
                    name: param.grad.clone()
                    for name, param in self.local_model.named_parameters()
                    if param.grad is not None
                }
                optimizer.step()

        diff = {}
        for name, data in self.local_model.state_dict().items():
            diff[name] = data - model.state_dict()[name]

        return diff, gradients, self.local_model

    def cosine_similarity(self, gradients_1, gradients_2):
        similarities = []

        for layer_name in gradients_1.keys():
            grad_1 = gradients_1[layer_name]
            grad_2 = gradients_2[layer_name]

            if grad_1.is_sparse:
                grad_1 = grad_1.cpu()
                grad_1_values = grad_1.values()
            else:
                grad_1_values = grad_1.view(-1)

            if grad_2.is_sparse:
                grad_2 = grad_2.cpu()
                grad_2_values = grad_2.values()
            else:
                grad_2_values = grad_2.view(-1)

            grad_1_flat = torch.cat([grad_1_values.view(-1)])
            grad_2_flat = torch.cat([grad_2_values.view(-1)])

            similarity = torch.nn.functional.cosine_similarity(grad_1_flat, grad_2_flat, dim=0)
            similarities.append(similarity)
            print(f"Layer: {layer_name}, Cosine Similarity: {similarity.item()}")

        average_similarity = torch.mean(torch.tensor(similarities))
        print(f"Average Cosine Similarity: {average_similarity.item()}")

        return similarities, average_similarity
