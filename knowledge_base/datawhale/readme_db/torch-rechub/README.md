# [Torch-RecHub] - Lightweight Recommender System Framework based on PyTorch

[![License](https://img.shields.io/badge/license-MIT-blue?style=for-the-badge)](LICENSE) 
![GitHub Repo stars](https://img.shields.io/github/stars/datawhalechina/torch-rechub?style=for-the-badge)
![GitHub forks](https://img.shields.io/github/forks/datawhalechina/torch-rechub?style=for-the-badge)
![GitHub issues](https://img.shields.io/github/issues/datawhalechina/torch-rechub?style=for-the-badge)
[![Python Version](https://img.shields.io/badge/python-3.8%2B-orange?style=for-the-badge)](https://www.python.org/) 
[![PyTorch Version](https://img.shields.io/badge/pytorch-1.7%2B-orange?style=for-the-badge)](https://pytorch.org/) 
[![annoy Version](https://img.shields.io/badge/annoy-1.17%2B-orange?style=for-the-badge)](https://pytorch.org/) 
[![pandas Version](https://img.shields.io/badge/pandas-1.2%2B-orange?style=for-the-badge)](https://pandas.pydata.org/) 
[![numpy Version](https://img.shields.io/badge/numpy-1.19%2B-orange?style=for-the-badge)](https://numpy.org/) 
[![scikit-learn Version](https://img.shields.io/badge/scikit_learn-0.23%2B-orange?style=for-the-badge)](https://scikit-learn.org/)
[![torch-rechub Version](https://img.shields.io/badge/torch_rechub-0.0.3%2B-orange?style=for-the-badge)](https://pypi.org/project/torch-rechub/)

English | [简体中文](README_zh.md)

**Torch-RecHub** is a flexible and extensible recommender system framework built with PyTorch. It aims to simplify research and application of recommendation algorithms by providing common model implementations, data processing tools, and evaluation metrics.

## ✨ Features

* **Modular Design:** Easy to add new models, datasets, and evaluation metrics.
* **PyTorch-based:** Leverages PyTorch's dynamic graph and GPU acceleration capabilities.
* **Rich Model Library:** Contains various classic and cutting-edge recommendation algorithms.
* **Standardized Pipeline:** Provides unified data loading, training, and evaluation workflows.
* **Easy Configuration:** Adjust experiment settings via config files or command-line arguments.
* **Reproducibility:** Designed to ensure reproducible experimental results.
* **Additional Features:** Negative sampling, multi-task learning, etc.

## 📖 Table of Contents

- [\[Torch-RecHub\] - Lightweight Recommender System Framework based on PyTorch](#torch-rechub---lightweight-recommender-system-framework-based-on-pytorch)
  - [✨ Features](#-features)
  - [📖 Table of Contents](#-table-of-contents)
  - [🔧 Installation](#-installation)
    - [Requirements](#requirements)
    - [Installation Steps](#installation-steps)
  - [🚀 Quick Start](#-quick-start)
  - [📂 Project Structure](#-project-structure)
  - [💡 Supported Models](#-supported-models)
  - [📊 Supported Datasets](#-supported-datasets)
  - [🧪 Examples](#-examples)
    - [Ranking (CTR Prediction)](#ranking-ctr-prediction)
    - [Multi-Task Ranking](#multi-task-ranking)
    - [Matching Model](#matching-model)
  - [🤝 Contributing](#-contributing)
  - [📜 License](#-license)
  - [📚 Citation](#-citation)
  - [📫 Contact](#-contact)

## 🔧 Installation

### Requirements

* Python 3.8+
* PyTorch 1.7+ (CUDA-enabled version recommended for GPU acceleration)
* NumPy
* Pandas
* SciPy
* Scikit-learn

### Installation Steps
- **Stable Version**
```bash
pip install torch-rechub
```

- **Latest Version (Recommended)**
```bash
git clone https://github.com/datawhalechina/torch-rechub.git
cd torch-rechub
python setup.py install
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## 🚀 Quick Start

Here's a simple example of training a model (e.g., MF - Matrix Factorization) on the MovieLens-100k dataset:

```bash
# 1. Prepare data (if preprocessing needed)
# python examples/matching/data/ml-1m/preprocess_ml.py

# 2. Train model
python run_ml_dssm.py
# Or override config with command-line arguments:
# python run_ml_dssm.py --model_name dssm --device 'cuda:0' --learning_rate 0.001 --epoch 50 --batch_size 4096 --weight_decay 0.0001 --save_dir 'saved/dssm_ml-100k'
```

After training, model files will be saved in the `saved/dssm_ml-100k` directory (or your configured directory).

## 📂 Project Structure

```
torch-rechub/             # Root directory
├── README.md             # Project documentation
├── torch_rechub/         # Core library
│   ├── basic/            # Basic components
│   ├── models/           # Recommendation model implementations
│   │   ├── matching/     # Matching models (DSSM/MIND/GRU4Rec etc.)
│   │   └── ranking/      # Ranking models (WideDeep/DeepFM/DIN etc.)
│   │   └── multi_task/   # Multi-task models (MMoE/ESMM etc.)
│   ├── trainers/         # Trainers
│   ├── utils/            # Utility functions
├── examples/             # Example scripts
│   ├── matching/         # Matching task examples
│   └── ranking/          # Ranking task examples
├── docs/                 # Documentation
├── tutorials/            # Jupyter tutorials
├── setup.py              # Package installation script
├── mkdocs.yml            # MkDocs config file
└── requirements.txt      # Project dependencies
```

## 💡 Supported Models

The framework currently supports the following recommendation models:

**General Recommendation:**

* **[DSSM](https://posenhuang.github.io/papers/cikm2013_DSSM_fullversion.pdf):** Deep Structured Semantic Model
* **[Wide&Deep](https://arxiv.org/abs/1606.07792):** Wide & Deep Learning for Recommender Systems
* **[FM](https://www.csie.ntu.edu.tw/~b97053/paper/Rendle2010FM.pdf):** Factorization Machines
* **[DeepFM](https://arxiv.org/abs/1703.04247):** Deep Factorization Machine
* ... 

**Sequential Recommendation:**

* **[DIN](https://arxiv.org/pdf/1706.06978.pdf):** Deep Interest Network
* **[DIEN](https://arxiv.org/pdf/1809.03672.pdf):** Deep Interest Evolution Network
* **[BST](https://arxiv.org/pdf/1905.06874.pdf):** Behavior Sequence Transformer
* **[GRU4Rec](https://arxiv.org/pdf/1511.06939.pdf):** Gated Recurrent Unit for Recommendation
* **[SASRec](https://arxiv.org/pdf/1808.09781.pdf):** Self-Attentive Sequential Recommendation
* ... 

**Multi-Interest Recommendation:**

* **[MIND](https://arxiv.org/pdf/1904.08030.pdf):** Multi-Interest Network with Dynamic Routing
* **[SINE](https://arxiv.org/pdf/2103.06920.pdf):** Self-Interested Network for Recommendation
* ... 

**Multi-Task Recommendation:**

* **[ESMM](https://arxiv.org/pdf/1804.07931.pdf):** Entire Space Multi-Task Model
* **[MMoE](https://dl.acm.org/doi/pdf/10.1145/3219819.3220007):** Multi-Task Multi-Interest Network for Recommendation
* **[PLE](https://dl.acm.org/doi/pdf/10.1145/3394486.3403394):** Personalized Learning to Rank
* **[AITM](https://arxiv.org/pdf/2005.02553.pdf):** Adaptive Interest-Task Matching
* ... 

## 📊 Supported Datasets

The framework provides built-in support or preprocessing scripts for the following common datasets:

* **MovieLens**
* **Amazon**
* **Criteo**
* **Avazu** 
* **Census-Income**
* **BookCrossing**
* **Ali-ccp**
* **Yidian**
* ...

The expected data format is typically an interaction file containing:
- User ID
- Item ID 
- Rating (optional)
- Timestamp (optional)

For specific format requirements, please refer to the example code in the `tutorials` directory.

You can easily integrate your own datasets by ensuring they conform to the framework's data format requirements or by writing custom data loaders.


## 🧪 Examples

All model usage examples can be found in `/examples`

### Ranking (CTR Prediction)

```python
from torch_rechub.models.ranking import DeepFM
from torch_rechub.trainers import CTRTrainer
from torch_rechub.utils.data import DataGenerator

dg = DataGenerator(x, y)
train_dataloader, val_dataloader, test_dataloader = dg.generate_dataloader(split_ratio=[0.7, 0.1], batch_size=256)

model = DeepFM(deep_features=deep_features, fm_features=fm_features, mlp_params={"dims": [256, 128], "dropout": 0.2, "activation": "relu"})

ctr_trainer = CTRTrainer(model)
ctr_trainer.fit(train_dataloader, val_dataloader)
auc = ctr_trainer.evaluate(ctr_trainer.model, test_dataloader)
```

### Multi-Task Ranking

```python
from torch_rechub.models.multi_task import SharedBottom, ESMM, MMOE, PLE, AITM
from torch_rechub.trainers import MTLTrainer

task_types = ["classification", "classification"] 
model = MMOE(features, task_types, 8, expert_params={"dims": [32,16]}, tower_params_list=[{"dims": [32, 16]}, {"dims": [32, 16]}])

mtl_trainer = MTLTrainer(model)
mtl_trainer.fit(train_dataloader, val_dataloader)
auc = ctr_trainer.evaluate(ctr_trainer.model, test_dataloader)
```

### Matching Model

```python
from torch_rechub.models.matching import DSSM
from torch_rechub.trainers import MatchTrainer
from torch_rechub.utils.data import MatchDataGenerator

dg = MatchDataGenerator(x y)
train_dl, test_dl, item_dl = dg.generate_dataloader(test_user, all_item, batch_size=256)

model = DSSM(user_features, item_features, temperature=0.02,
             user_params={
                 "dims": [256, 128, 64],
                 "activation": 'prelu',  
             },
             item_params={
                 "dims": [256, 128, 64],
                 "activation": 'prelu', 
             })

match_trainer = MatchTrainer(model)
match_trainer.fit(train_dl)
```

## 🤝 Contributing

We welcome all types of contributions! If you'd like to contribute to this project, please follow these steps:

1. **Fork the repository:** Click the "Fork" button in the upper right corner.
2. **Make your changes:** Implement new features or fix bugs.
3. **Commit changes:** `git commit -m "feat: add new feature"` or `fix: fix some issue"` (Following [Conventional Commits](https://www.conventionalcommits.org/) is preferred).
4. **Push to branch:** `git push origin`
5. **Create Pull Request:** Go back to the original repository page, click "New pull request", compare your branch with the `main` branch of the main repository, and submit PR.

Please ensure your PR description clearly explains the changes and their purpose.

We also welcome bug reports and feature suggestions through [Issues](https://github.com/datawhalechina/torch-rechub/issues).

## 📜 License

This project is licensed under the [MIT License](LICENSE).

## 📚 Citation

If you use this framework in your research or work, please consider citing:

```bibtex
@misc{torch_rechub,
    title = {Torch-RecHub},
    author = {Datawhale},
    year = {2024},
    publisher = {GitHub},
    journal = {GitHub repository},
    howpublished = {\url{https://github.com/datawhalechina/torch-rechub}},
    note = {A PyTorch-based recommender system framework providing easy-to-use and extensible solutions}
}
```

## 📫 Contact

* **Project Lead:** [morningsky](https://github.com/morningsky) 
* [**GitHub Issues**](https://github.com/datawhalechina/torch-rechub/issues)


---

*Last updated: [2025-03-31]*