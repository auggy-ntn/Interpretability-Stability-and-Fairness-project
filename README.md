# Credit scoring - Interpretability, Stability and Fairness course final project

This GitHub repository contains the code for the final project of the Interpretability, Stability and Fairness course from the Data Science and AI MScT at X-HEC.

The objective of this project is to apply to methods seen in class to study the interpretabililty, stability and fairness of an ML model trained on a credit scoring dataset.

## Repository structure
```
project/
├── data/                 # Raw and processed datasets
├── notebooks/            # Jupyter notebooks for steps
├── src/                  # Source code for model training and useful functions
├── models/               # Saved trained models
├── results/              # Results, figures, and outputs
├── requirements.txt      # Python dependencies
└── pyproject.toml
```



## Installation & Usage

### Development
To reproduce the development environment, follow these steps:

0. **(Prerequisite)** Have ```uv``` installed. See [the project's website](https://docs.astral.sh/uv/) for more information. In your terminal (MacOS and Linux users), run 
```zsh
curl -LsSf https://astral.sh/uv/install.sh | sh
```

1. Clone the project:
```zsh
git clone https://github.com/auggy-ntn/Interpretability-Stability-and-Fairness-project.git
```

2. In the project's workspace run the following command to synchronize your environment with the project's development requirements:
```zsh
uv sync --dev
```
You are all set!

Alternatively, if you don't want to use ```uv```, you can run the following command:
```zsh
pip install -r requirements.txt
```
