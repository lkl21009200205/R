# Federated Learning Incentive Simulation

This repository contains a local Python simulation of horizontal federated learning for image classification. It is designed as a research prototype for testing client verification, lazy-client incentive scoring, malicious-update simulation, score-weighted aggregation, and multi-round Stackelberg incentive settlement in one workflow.

The project is intended for research prototyping. It runs all clients and the server in one local process and does not implement network communication, distributed deployment, secure aggregation, or blockchain execution.

## Features

- Local federated learning simulation with configurable numbers of clients and participating clients per round.
- CIFAR-10, CIFAR-100, and MNIST dataset loading through `torchvision`.
- Torchvision model support, including ResNet, DenseNet, AlexNet, VGG, Inception, and GoogLeNet.
- Lightweight FEMNIST CNN definition and auxiliary FEMNIST/LEAF loading utilities.
- Client-side effort control driven by previous-round Stackelberg settlement results.
- Representative-model generation by fusing late layers, currently designed around ResNet-style `layer4` and `fc` layers.
- Validation scoring that combines Dempster-Shafer evidence scoring with cross-round structural-anomaly detection.
- Multi-metric incentive score `Si` based on min-max normalization, entropy weighting, TOPSIS, and optional novelty gating.
- Score-weighted aggregation that ignores clients with non-positive incentive scores.
- Stackelberg contract settlement with reputation updates and next-round effort recommendations.
- Built-in malicious representative-model injection for testing verification and down-weighting behavior.

## Repository Layout

```text
.
+-- README.md
`-- model_code/
    |-- main.py                 # End-to-end federated-learning simulation entry point
    |-- client.py               # Local client training and effort-controlled batch usage
    |-- server.py               # Global model ownership, aggregation, and evaluation
    |-- datasets.py             # MNIST/CIFAR-10/CIFAR-100 dataset loaders
    |-- models.py               # Model factory and FEMNIST CNN definition
    |-- Model_Verify.py         # Representative models, validation, anomaly scoring, attacks
    |-- Model_Encourage.py      # Multi-metric incentive scoring
    |-- Stackelberg.py          # Contract, reputation, and effort dynamics
    |-- PCA.py                  # PCA helper utilities for model parameters
    |-- trydata.py              # Auxiliary FEMNIST/LEAF data-loading utilities
    |-- utils/
    |   `-- conf.json           # Default experiment configuration
    `-- data/                   # Local dataset archives/extracted datasets
```

## Environment Setup

The code is tested as a standard Python project. Use Python 3.8 or later. Runtime dependencies are PyTorch, torchvision, NumPy, and scikit-learn.

Linux/macOS setup:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install torch torchvision numpy scikit-learn
```

Windows PowerShell setup:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install torch torchvision numpy scikit-learn
```

Conda setup:

```bash
conda create -n fl-incentive python=3.10 -y
conda activate fl-incentive
python -m pip install --upgrade pip
pip install torch torchvision numpy scikit-learn
```

Verify the environment:

```bash
python - <<'PY'
import torch
import torchvision
import numpy
import sklearn

print("torch:", torch.__version__)
print("torchvision:", torchvision.__version__)
print("cuda available:", torch.cuda.is_available())
print("numpy:", numpy.__version__)
print("sklearn:", sklearn.__version__)
PY
```

If you need a specific CUDA build, install the matching PyTorch and torchvision packages for your machine before running the experiment. For reproducible experiments, pin exact dependency versions in your environment manager before collecting results.

## Dataset Preparation

The default configuration uses CIFAR-100:

```json
"type": "cifar100"
```

Datasets are loaded under `model_code/data/`. The `torchvision` loaders use `download=True` for MNIST, CIFAR-10, and CIFAR-100, so the data can be downloaded automatically when network access is available.

If you already have dataset archives, place them under:

```text
model_code/data/
```

Supported dataset names in `datasets.py` are:

- `mnist`
- `cifar`
- `cifar100`

## Configuration

The default configuration is stored in `model_code/utils/conf.json`:

```json
{
  "model_name": "resnet18",
  "no_models": 4,
  "type": "cifar100",
  "global_epochs": 20,
  "local_epochs": 1,
  "k": 4,
  "batch_size": 32,
  "lr": 0.001,
  "momentum": 0.0001,
  "lambda": 0.1
}
```

Important fields:

- `model_name`: model created by `models.get_model`, such as `resnet18`, `resnet50`, `densenet121`, `alexnet`, `vgg16`, `vgg19`, `inception_v3`, `googlenet`, or `FEMNIST_CNN`.
- `no_models`: total number of simulated clients.
- `type`: dataset name loaded by `datasets.get_dataset`.
- `global_epochs`: number of server-client communication rounds.
- `local_epochs`: local training epochs per selected client.
- `k`: number of clients sampled in each global round.
- `batch_size`: local and evaluation batch size.
- `lr`, `momentum`: SGD optimizer hyperparameters.
- `lambda`: global aggregation step size applied by the server.

Optional effort-control fields can also be added to the configuration:

- `initial_effort`: initial training effort used before Stackelberg feedback is available. Defaults to `1.0`.
- `effort_floor`: lower bound for effort-controlled local batch usage. Defaults to `0.0`.
- `effort_cap`: upper bound for effort normalization. Defaults to `1.0`.

Validate the JSON configuration before running:

```bash
python -m json.tool model_code/utils/conf.json > /tmp/conf.validated.json
```

## Quick Start

Run the experiment from `model_code/`; the current scripts use local imports and load datasets relative to the active working directory. Start a full experiment with the default configuration:

```bash
cd model_code
python main.py -c ./utils/conf.json
```

Equivalent one-line command from the repository root:

```bash
(cd model_code && python main.py -c ./utils/conf.json)
```

The script prints per-round information including validation scores, incentive scores, Stackelberg settlement details, global accuracy, and global loss.

For a short smoke run, create a temporary one-round configuration first:

```bash
cd model_code
python - <<'PY'
import json

source = "utils/conf.json"
target = "utils/conf.quick.json"

with open(source, "r", encoding="utf-8") as f:
    conf = json.load(f)

conf.update({
    "global_epochs": 1,
    "local_epochs": 1,
    "no_models": 2,
    "k": 2,
    "batch_size": 32
})

with open(target, "w", encoding="utf-8") as f:
    json.dump(conf, f, indent=2)
    f.write("\n")

print(target)
PY
python main.py -c ./utils/conf.quick.json
```

Return to the repository root after running from `model_code`:

```bash
cd ..
```

Useful standalone checks:

```bash
python -m py_compile model_code/main.py model_code/client.py model_code/server.py
python -m py_compile model_code/models.py model_code/datasets.py
python -m py_compile model_code/Model_Verify.py model_code/Model_Encourage.py model_code/Stackelberg.py
```

## Training Workflow

At a high level, `main.py` executes the following loop:

1. Load the experiment configuration.
2. Load the selected dataset and initialize the global server model.
3. Create local clients by partitioning the training dataset by client ID.
4. Randomly sample `k` clients per global round.
5. Train selected clients locally using effort levels from the previous Stackelberg settlement.
6. Build representative models by fusing selected late-layer parameters.
7. Inject one malicious representative model for robustness testing.
8. Score representative models using validation evidence and structural-anomaly detection.
9. Compute each selected client's incentive score `Si`.
10. Aggregate only positive-score client updates using normalized `Si` weights.
11. Run Stackelberg settlement to update payments, reputation, and next-round effort.
12. Evaluate the updated global model.

## Incentive Scoring

`Model_Encourage.py` computes the final incentive score `Si` from multiple indicators:

- D-S validation score, treated as a benefit metric.
- Data size, treated as a benefit metric.
- Model novelty, treated as a benefit metric.
- Local training delay, treated as a cost metric.
- End-to-end delay, treated as a cost metric.

The scoring pipeline is:

```text
raw metrics -> min-max normalization -> entropy weights -> TOPSIS score -> optional novelty gating -> Si
```

If real metric arrays are not provided, the module generates sample values for demonstration. In the current main workflow, D-S/structural validation scores are passed in, while other metrics use the scoring module's defaults unless extended by the caller.

## Verification and Attack Simulation

`Model_Verify.py` contains utilities for:

- Representative-model creation.
- Late-layer fusion for ResNet-style models.
- D-S evidence-based validation scoring.
- Cross-round structural-anomaly scoring over late-layer parameter changes.
- Malicious model generation and attack-style parameter corruption.

By default, `main.py` intentionally modifies the first representative model in each selected-client round:

```python
model_storage_represent[0] = Model_Verify.malicious_model_create(model_storage_represent[0])
```

This is useful for testing the validation, incentive, and aggregation logic. Disable or remove that line for clean federated-training experiments.

## Stackelberg Settlement

`Stackelberg.py` uses the score map from `Model_Encourage.py` to run a contract-based leader-follower settlement. It maintains cross-round state for effort and reputation. The settlement output includes contract payments, reputation updates, and `effort_next`, which is fed into the next global training round.

The core payment form is:

```text
R_i(S_i) = a_i + k_i * S_i - (eta / 2) * S_i^2
```

The current integration stores the game object across rounds, so clients' training effort evolves over time rather than resetting every epoch.

## Notes and Known Limitations

- The code is a local simulator, not a production federated-learning framework.
- No networking, client failure handling, secure aggregation, authentication, or on-chain execution is implemented.
- Torchvision model classifier heads are not automatically adapted to the selected dataset class count. Adjust `models.py` if an experiment requires an exact output dimension.
- The default ResNet late-layer fusion assumes parameter names such as `layer4` and `fc`. Update `late_layer_prefixes` when using architectures with different naming conventions.
- Some helper modules are experimental or auxiliary and are not invoked by the default `main.py` workflow.

## Citation / Research Use

If this code is used in a paper or experiment, document the exact configuration file, dataset, random seeds, hardware environment, and whether the malicious-model injection line was enabled. These choices materially affect reported accuracy, loss, incentive scores, and settlement behavior.
