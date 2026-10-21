# FedACS: Adaptive Client Selection Framework for Federated Learning

[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2%2B-EE4C2C.svg)](https://pytorch.org/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-green.svg)](LICENSE)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/nianhuawu9-ops/fedacs/pulls)

**FedACS** is a modular, high-performance Federated Learning (FL) optimization framework designed for non-IID data distributions, client heterogeneity, and communication-constrained environments. 

Under severe statistical skew, conventional federated algorithms (such as FedAvg) suffer from client drift, slow convergence, and biased participation. FedACS tackles these challenges through an end-to-end coordinated pipeline: **multi-factor client utility scoring**, **budget-aware tiered selection**, and **contribution-aware adaptive weighted aggregation**.

---

## 🌟 Key Highlights

- **Multi-Factor Client Scoring**: Dynamically integrates loss reduction, participation fairness compensation, and communication/computation cost with EMA smoothing to mitigate noisy updates.
- **Budget-Aware Tiered Selection**: A cascaded selection policy prioritizing high-utility clients, starvation mitigation (fairness tier), and cost-effective exploration under rigid communication constraints.
- **Adaptive Aggregation & Drift Control**: Dynamic re-weighting of client gradients based on empirical update quality, combined with server-side Adam optimization to stabilize global convergence.
- **Extreme Non-IID Stability**: Features customized backbones (e.g., `ResNet-8` with GroupNorm) to eliminate the BatchNorm running statistics collapse typical in Dirichlet ($\alpha=0.1$) partitions.
- **Extensive Baseline Suite**: Plug-and-play implementations for **FedAvg**, **FedProx**, **FedNova**, **SCAFFOLD**, and **FedACS**.
- **Comprehensive Evaluation**: Automated tracking of Top-1/Top-5 accuracy, rounds/time/communication-to-target, Jain's fairness index, selection entropy, and client drift.

---

## 📁 Repository Structure

```text
fedacs/
├── fl_framework/
│   ├── data/                   # Dataset cache and partition metadata
│   ├── figures/                # Evaluation plots, heatmaps, and summary tables
│   ├── fl_framework/           # Core library
│   │   ├── servers/            # Server implementations (FedACS, FedAvg, FedProx, FedNova, SCAFFOLD)
│   │   ├── client.py           # Local client trainer and drift estimation
│   │   ├── config.py           # Hyperparameter dataclasses (FLConfig)
│   │   ├── dataset.py          # Dirichlet and shard Non-IID partitioning
│   │   ├── models.py           # Model definitions (ResNet-8 with GroupNorm, SimpleCNN)
│   │   ├── metrics.py          # Multi-dimensional evaluation suite
│   │   ├── main.py             # Single-run experiment entry point
│   │   └── run_paper_experiments.py # Paper benchmark & ablation runner
│   ├── make_figures.py         # Visualizations and paper asset generator
│   ├── requirements.txt        # Python dependency manifest
│   └── run_all_experiments.ps1 # One-click PowerShell execution pipeline
├── LICENSE                     # Open source license
└── README.md                   # Project documentation
```

---

## 🚀 Quick Start

### 1. Environment Setup

Clone the repository and install dependencies in an isolated virtual environment:

```bash
git clone https://github.com/nianhuawu9-ops/fedacs.git
cd fedacs/fl_framework

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate

# Install required dependencies
pip install -r requirements.txt
```

### 2. Run a Single FL Experiment

Run FedACS on MNIST with Dirichlet Non-IID partitioning ($\alpha=0.1$):

```bash
python -m fl_framework.main \
  --method FedACS \
  --dataset mnist \
  --backbone cnn \
  --partition dirichlet \
  --alpha 0.1 \
  --num_clients 100 \
  --fraction 0.1 \
  --rounds 50 \
  --local_epochs 5 \
  --local_lr 0.01
```

You can switch `--method` to any supported baseline: `FedAvg`, `FedProx`, `FedNova`, `SCAFFOLD`, or `FedACS`.

### 3. Run Full Benchmark & Ablation Suite

To execute the automated evaluation comparing all methods across multiple Non-IID settings:

```bash
# Windows PowerShell
.\run_all_experiments.ps1

# Or run directly via Python
python -m fl_framework.run_paper_experiments --data-dir fl_framework/data
```

### 4. Generate Figures and Tables

Once experiment logs are generated, plot the comparative convergence heatmaps and fairness curves:

```bash
python make_figures.py
```
Output figures and CSV summaries will be saved in `fl_framework/figures/`.

---

## 📊 Supported Baselines & Methods

| Algorithm | Client Selection | Local Objective Adjustment | Aggregation Strategy |
| :--- | :--- | :--- | :--- |
| **FedAvg** | Uniform Random | Standard Local SGD | Normalized Sample Weighting |
| **FedProx** | Uniform Random | Proximal Regularizer ($\mu$) | Normalized Sample Weighting |
| **SCAFFOLD** | Uniform Random | Control Variates ($c_i, c$) | Parameter Drift Aggregation |
| **FedNova** | Uniform Random | Normalized Local Steps | Normalized Momentum Aggregation |
| **FedACS (Ours)** | **Budget-Aware Tiered Selection** | **Drift & Gain Regularized** | **Adaptive Quality Weighting + Server Adam** |

---

## 📈 Evaluation Metrics

The framework evaluates federated performance along multiple dimensions:
- **Optimization Efficiency**: Final Top-1 Accuracy, Rounds-to-Target (e.g. 80%/90%), Time-to-Target, and Total Communication (MB).
- **System Fairness**: Jain's fairness index on client test accuracies, client participation frequency, and selection entropy.
- **Training Stability**: Client drift norms, tail standard deviation of loss/accuracy, and peak performance drops.

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!  
Feel free to open an issue or submit a pull request if you want to integrate new client selection algorithms or datasets.

---

## 📄 License

This project is licensed under the [Apache License 2.0](LICENSE).
