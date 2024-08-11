# Wine Quality Classification

This project is part of a lab assignment that focuses on classifying wine quality using physicochemical properties from the Wine Quality dataset.<br> The goal is to build a model that can predict the quality of wine based on various attributes such as acidity, pH, and alcohol concentration.

## Table of Contents

- [Project Overview](#project-overview)
- [Directory Structure](#directory-structure)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Usage](#usage)
- [Data](#data)
- [Modeling and Evaluation](#modeling-and-evaluation)
- [Optimization](#optimization)
- [Contributing](#contributing)
- [License](#license)
- [Contact Information](#contact-information)
- [References](#references)

## Project Overview

This project leverages the Wine Quality dataset to model and predict the quality of wines based on their physicochemical properties.<br> The project includes steps for data preprocessing, feature engineering (including PCA), custom logistic regression implementations, model training, and hyperparameter optimization. <br> This work is part of a lab assignment aimed at extending the concepts of logistic regression.

## Directory Structure

```plaintext
Wine-Quality-Classification/
│
├── notebooks/
│   ├── EDA.ipynb                      # Exploratory Data Analysis notebook
│
├── src/
│   ├── __init__.py                    # Package initialization
│   ├── data_preparation.py            # Data loading and preprocessing
│   ├── pca_analysis.py                # PCA and dimensionality reduction
│   ├── custom_logistic_regression.py  # Custom logistic regression implementations
│   ├── model_training.py              # Model training and evaluation functions
│   ├── main.py                        # Main script to run the entire pipeline
│
├── README.md                          # Project overview and instructions
├── requirements.txt                   # Python dependencies
├── .gitignore                         # Ignored files and directories
└── LICENSE                            # License for the project
```

## Getting Started

### Prerequisites

- Python 3.8 or later
- pip package manager

### Installation

1. **Clone the repository:**

```bash
git clone https://github.com/minhosong88/wine-quality-classification.git
cd wine-quality-classification
```

2.**Install dependencies:**

```bash
pip install -r requirements.txt
```

### Usage

To run the full analysis pipeline, use the following command:

```bash
python src/main.py
```

Alternatively, you can explore the data and models interactively using the provided Jupyter notebooks.

## Data

The dataset used is the [Wine Quality dataset](https://archive.ics.uci.edu/dataset/186/wine+quality) from UCI Machine Learning Repository, containing physicochemical properties of red and white wine samples.

## Modeling and Evaluation

`model_training.py` script cover the training of logistic regression models and their evaluation using custom implementations and standard libraries.

## Optimization

Hyperparameter optimization is performed in the `model_training.py` script to find the best parameters for the logistic regression model.

## Contributing

Contributions are welcome! Please fork this repository and submit a pull request with your improvements.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

## Contact Information

For any questions or inquiries, please contact:

- **GitHub**: [minhosong88](https://github.com/minhosong88)
- **Email**: [hominsong@naver.com](mailto:hominsong@naver.com)

## References

- [Wine Quality Dataset](https://archive.ics.uci.edu/dataset/186/wine+quality)
- [Cortez et al., 2009](https://www.semanticscholar.org/paper/Modeling-wine-preferences-by-data-mining-from-Cortez-Cerdeira/bf15a0ccc14ac1deb5cea570c870389c16be019c)
