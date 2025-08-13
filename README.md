# FinMamba: Financial Bankruptcy Prediction via Selective State Space with Expert Networks

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1.0-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Paper](https://img.shields.io/badge/Paper-arXiv-green.svg)](https://arxiv.org/abs/xxxx.xxxxx)

## 📋 Overview

FinMamba is a novel deep learning architecture that integrates **selective state space modeling** with **domain-specific expert networks** for cross-regional bankruptcy prediction. The framework addresses computational bottlenecks in traditional methods while achieving superior predictive performance through linear computational complexity and enhanced cross-regional adaptability.

### 🎯 Key Features

- **Linear Complexity**: Selective state space modeling enables O(L) computational complexity vs O(L²) for attention-based models
- **Domain Expertise Integration**: Mixture-of-experts network explicitly models four financial analysis dimensions
- **Cross-Regional Robustness**: Validated across Taiwan and US markets with consistent performance
- **Early Warning Capability**: Provides bankruptcy alerts up to 22 months in advance
- **Regulatory Compliance**: SHAP-based interpretability for transparent decision-making

## 🏗️ Architecture

FinMamba consists of three core components:

1. **Financial Selective State Space Model (FinSSS)**: Adaptively focuses on temporally important information through content-aware parameter selection
2. **Mixture-of-Experts (MoE) Networks**: Decomposes financial analysis into specialized dimensions (profitability, liquidity, leverage, operational efficiency)
3. **Cross-Modal Fusion Mechanism**: Combines temporal and expert-driven features through bidirectional attention

![FinMamba Architecture](assets/architecture.png)

## 📊 Performance Results

| Dataset | Metric | FinMamba | QTIAH-GNN | Text-ML Fusion | Improvement |
|---------|--------|----------|-----------|----------------|-------------|
| Taiwan  | Accuracy | **93.7%** | 91.9% | 92.4% | +1.8% |
| Taiwan  | AUC | **89.2%** | 87.8% | 87.1% | +1.4% |
| Taiwan  | Recall | **83.4%** | 80.2% | 78.9% | +3.2% |
| US      | Accuracy | **92.5%** | 90.4% | 91.1% | +2.1% |
| US      | AUC | **87.6%** | 85.4% | 86.2% | +2.2% |

*Statistical significance: p < 0.01, Cohen's d > 1.4*

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/litlelight/FinMamba.git
cd FinMamba

# Create virtual environment
conda create -n finmamba python=3.8
conda activate finmamba

# Install dependencies
pip install -r requirements.txt
```

### Requirements

```
torch>=2.1.0
numpy>=1.24.0
pandas>=1.5.0
scikit-learn>=1.3.0
transformers>=4.35.0
shap>=0.42.0
matplotlib>=3.7.0
seaborn>=0.12.0
```

### Basic Usage

```python
from finmamba import FinMambaModel, DataProcessor

# Load and preprocess data
processor = DataProcessor()
X_train, y_train = processor.load_data('data/taiwan_bankruptcy.csv')
X_train, y_train = processor.preprocess(X_train, y_train)

# Initialize and train model
model = FinMambaModel(
    input_dim=95,
    hidden_dim=256,
    num_experts=4,
    sequence_length=12
)

model.fit(X_train, y_train, epochs=50, batch_size=32)

# Make predictions
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)
```

## 📁 Project Structure

```
FinMamba/
├── finmamba/
│   ├── models/
│   │   ├── finmamba.py          # Main FinMamba architecture
│   │   ├── finsss.py            # Financial Selective State Space Model
│   │   ├── experts.py           # Mixture-of-Experts networks
│   │   └── fusion.py            # Cross-modal fusion mechanism
│   ├── data/
│   │   ├── processor.py         # Data preprocessing utilities
│   │   └── loader.py            # Dataset loading functions
│   └── utils/
│       ├── metrics.py           # Evaluation metrics
│       ├── visualization.py     # Plotting utilities
│       └── interpretability.py  # SHAP analysis tools
├── experiments/
│   ├── train.py                 # Training script
│   ├── evaluate.py              # Evaluation script
│   ├── ablation_study.py        # Ablation experiments
│   └── hyperparameter_tuning.py # Hyperparameter optimization
├── configs/
│   ├── taiwan_config.yaml       # Taiwan dataset configuration
│   └── us_config.yaml           # US dataset configuration
├── notebooks/
│   ├── data_exploration.ipynb   # Data analysis notebooks
│   ├── model_analysis.ipynb     # Model interpretation
│   └── case_studies.ipynb       # Bankruptcy case studies
├── data/
│   ├── processed/               # Processed datasets
│   └── synthetic/               # Synthetic replication data
├── results/
│   ├── figures/                 # Generated plots and figures
│   ├── logs/                    # Training logs
│   └── models/                  # Saved model checkpoints
└── README.md
```

## 📈 Reproducing Results

### 1. Data Preparation

```bash
# Download and prepare Taiwan dataset
python data/prepare_taiwan_data.py

# Download and prepare US dataset  
python data/prepare_us_data.py
```

### 2. Train FinMamba Model

```bash
# Train on Taiwan dataset
python experiments/train.py --config configs/taiwan_config.yaml

# Train on US dataset
python experiments/train.py --config configs/us_config.yaml
```

### 3. Run Experiments

```bash
# Comparative analysis
python experiments/comparative_analysis.py

# Ablation study
python experiments/ablation_study.py

# Hyperparameter tuning
python experiments/hyperparameter_tuning.py

# Case study analysis
python experiments/case_studies.py
```

## 📊 Data Availability

### Taiwan Bankruptcy Dataset
- **Source**: Taiwan Economic Journal (TEJ) - *Requires institutional subscription*
- **Period**: 1999-2009
- **Companies**: 6,819 (220 bankrupt, 6,599 non-bankrupt)
- **Features**: 95 financial indicators

### US Bankruptcy Dataset
- **Source**: Public SEC filings (NYSE/NASDAQ)
- **Period**: 1999-2018  
- **Companies**: 8,262 (~2.5% bankruptcy rate)
- **Features**: Corresponding financial indicators

### Synthetic Replication Data
For reproducibility, we provide synthetic datasets that preserve statistical properties:
```bash
# Generate synthetic Taiwan data
python data/generate_synthetic_taiwan.py

# Generate synthetic US data
python data/generate_synthetic_us.py
```

## 🔧 Hyperparameter Configuration

Key hyperparameters and their optimal values:

| Parameter | Taiwan | US | Description |
|-----------|--------|----|----|
| Learning Rate | 0.001 | 0.001 | Adam optimizer learning rate |
| Batch Size | 32 | 64 | Training batch size |
| Hidden Dimension | 256 | 256 | Model hidden state dimension |
| Sequence Length | 12 | 16 | Input temporal sequence length |
| Dropout Rate | 0.3 | 0.4 | Regularization dropout rate |
| Num Experts | 4 | 4 | Number of expert networks |

## 📖 Citation

If you use this code in your research, please cite our paper:

```bibtex
@article{finmamba2024,
  title={FinMamba: Financial Bankruptcy Prediction via Selective State Space with Expert Networks},
  author={[Authors]},
  journal={[Journal]},
  year={2024},
  url={https://github.com/litlelight/FinMamba}
}
```

## 🤝 Contributing

We welcome contributions! Please see our [contributing guidelines](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Create a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Taiwan Economic Journal (TEJ) for providing bankruptcy data
- SEC for public financial filings access
- The open-source community for foundational tools and libraries

## 📞 Contact

For questions or collaboration inquiries:

- **Issues**: Please use [GitHub Issues](https://github.com/litlelight/FinMamba/issues)
- **Discussions**: Join our [GitHub Discussions](https://github.com/litlelight/FinMamba/discussions)
- **Email**: [your-email@institution.edu]

---

⭐ **Star this repository if you find it helpful!** ⭐
