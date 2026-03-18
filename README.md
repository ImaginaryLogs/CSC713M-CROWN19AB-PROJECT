# CoV-AbDab Machine Learning Model

![Static Badge](https://img.shields.io/badge/AY2527_T2-CSC713M-blue?style=plastic) 	[![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=fff)](#)

A machine learning model project designed to identifying neutralizing and binding status of antibody proteins.

| <div><img src="./docs/assets/waltuh.png" style="width:100px"/></div> | By the **Alphafold Cooking Lab Group** (G01)
|---|----|

## Proposal

### Justification

Traditional laboratory experiments for protein analysis require immense time and cost, so accelerating the scientific process enables faster breakthroughs in life-saving areas like drug-discovery. Producing Machine Learning models like these help democratize bioinformatics resources, so that those countries who have weak spending on to biomedical research can also utilize and participate in analysis proteins through just computers.

### Dataset

Here, we listed the following website on which we based our data from:

Antibody Dataset: Coronavirus-Binding Antibody Sequences & Structures
[CoV-AbDab Website](https://opig.stats.ox.ac.uk/webapps/covabdab/)

Antibody Article: AbSet: A Standardized Data Set of Antibody Structures for Machine Learning Applications
[PMC Article](https://pmc.ncbi.nlm.nih.gov/articles/PMC3766990/)

Viruses Properties Data Entries:

1. Sequence Reference - [Uniprot Link](https://www.uniprot.org/uniprotkb/P0DTC2/entry)
2. Isoelectric Reference - [PMC Link](https://pmc.ncbi.nlm.nih.gov/articles/PMC8401476/)
3. Physicochemical - [PMC Link](https://pmc.ncbi.nlm.nih.gov/articles/PMC7283733/#elps7215-sec-0040)

## Installation & Usage

Please install `uv` pack manager [UV DOCS](https://docs.astral.sh/uv/).

`uv sync`

## File Structure

```txt
root/
├── data/                    # For raw/processed protein data
│   ├── raw_cdr.csv
│   └── binding_labels.csv
├── src/                     # Source code directory
│   ├── data_module/          # ProteinDataModule implementations
│   │   ├── \_\_init\_\_.py
│   │   └── data_module.py    # Main DataModule definition
│   ├── features/             # Feature extractors
│   │   ├── \_\_init\_\_.py
│   │   ├── physchem.py        # Physicochemical feature extraction
│   │   ├── sequence_embeddings.py  # Sequence embedding models
│   │   └── structural_recursive.py # Graph/recursive features
│   ├── models/               # Model implementations
│   │   ├── \_\_init\_\_.py
│   │   ├── classical_ml.py   # Standard ML models (MLP)
│   │   ├── deep_learning.py   # Deep learning models (CNN, Transformer)
│   │   ├── recursive_dl.py   # Recursive deep learning models (Tree-LSTM)
│   │   └── quantum_dl.py     # PennyLane hybrid quantum-classical models
│   ├── lightning_modules/     # Lightning wrappers
│   │   ├── \_\_init\_\_.py
│   │   └── classifier.py     # ProteinClassifier LightningModule
│   └── utils/                # Utility scripts
│       ├── \_\_init\_\_.py
│       └── metrics.py        # Metric definitions for W&B logging
├── config/                  # Configuration files
│   ├── train_config.yaml    # General training parameters
│   └── model_configs/       # Hyperparameters for different models
│       ├── mlp.yaml
│       ├── deep_cnn.yaml
│       └── quantum_hybrid.yaml
├── notebooks/                # Jupyter Notebooks for exploration
│   └── eda.ipynb
├── tests/                   # Unit tests
│   └── test_data_module.py
├── main.py                   # Main script to run training
├── requirements.txt
├── .gitignore
├── .wandb/                   # Wandb configurations (internal)
└── README.md
```
