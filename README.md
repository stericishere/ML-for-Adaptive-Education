# Machine Learning Models for Adaptive Education

[](https://github.com/stericishere/Machine-Learning-Model-for-Adaptive-Education)

\<p align="center"\>
\<img src="[https://img.shields.io/badge/Python-3776AB?logo=python\&logoColor=white](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white)" alt="Python"/\>
\<img src="[https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch\&logoColor=white](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)" alt="PyTorch"/\>
\<img src="[https://img.shields.io/badge/Numpy-013243?logo=numpy\&logoColor=white](https://www.google.com/search?q=https://img.shields.io/badge/Numpy-013243%3Flogo%3Dnumpy%26logoColor%3Dwhite)" alt="Numpy"/\>
\<img src="[https://img.shields.io/badge/SciPy-8DBC30?logo=scipy\&logoColor=white](https://www.google.com/search?q=https://img.shields.io/badge/SciPy-8DBC30%3Flogo%3Dscipy%26logoColor%3Dwhite)" alt="SciPy"/\>
\<img src="https://[https://img.shields.io/badge/scikit--learn-F7931E?logo=scikit-learn\&logoColor=white](https://www.google.com/search?q=https://img.shields.io/badge/scikit--learn-F7931E%3Flogo%3Dscikit-learn%26logoColor%3Dwhite)" alt="scikit-learn"/\>
\<img src="[https://img.shields.io/badge/Matplotlib-5A95CD?logo=matplotlib\&logoColor=white](https://www.google.com/search?q=https://img.shields.io/badge/Matplotlib-5A95CD%3Flogo%3Dmatplotlib%26logoColor%3Dwhite)" alt="Matplotlib"/\>
\</p\>

A collection of machine learning models for adaptive education, designed to predict student performance on a set of questions. The project implements and evaluates various models, including KNN, Item Response Theory, Neural Networks, and a final ensemble model.

## Overview

This project focuses on applying machine learning techniques to student data to predict the correctness of their answers to questions. The implemented models are designed to learn from user-question interaction data, estimating student abilities and question difficulties to improve predictions. The project includes individual models as well as an ensemble approach for enhanced accuracy.

### Key Features

  * **Multiple Model Implementations:** Includes implementations of KNN, Item Response Theory, and a Neural Network (AutoEncoder).
  * **Ensemble Modeling:** A majority vote ensemble model combines predictions from the core models to improve overall performance.
  * **Model Evaluation:** Scripts are provided to tune and evaluate each model using validation data and report final test accuracy.
  * **Data Handling:** The project includes utility functions for loading and processing the student and question data.

## Technology Stack

  * **Languages:** Python
  * **Libraries:** NumPy, PyTorch, scikit-learn, Matplotlib
  * **Data Format:** CSV, NPZ

## Architecture

The project is structured to allow for the independent training and evaluation of each model, with a main ensemble script bringing the predictions together.

1.  **Individual Models:** KNN, IRT, and Neural Network models are trained and optimized separately.
2.  **Ensemble Script:** The `ensemble.py` script orchestrates the process by loading data, training each base model, gathering their predictions, and applying a majority vote to produce the final result.

-----

## Quick Start

### Prerequisites

  * Python 3.8+
  * The required libraries listed in the Technology Stack. You can install them with pip:
    `pip install numpy scikit-learn matplotlib torch`

### Development Setup

1.  **Clone the repository** (if applicable)

2.  **Download the data files** and place them in a `data/` directory at the root of the project. The scripts assume the data is structured as follows:

    ```
    .
    ├── data/
    │   ├── train_data.csv
    │   ├── valid_data.csv
    │   ├── test_data.csv
    │   ├── train_sparse.npz
    │   ├── question_meta.csv
    │   └── subject_meta.csv
    └── [project files]
    ```

## Usage

  * **To run the ensemble model:**
    `python ensemble.py`

  * **To evaluate the KNN models:**
    `python knn.py`

  * **To evaluate the Item Response Theory model:**
    `python item_response.py`

  * **To evaluate the Neural Network model:**
    `python neural_network.py`

  * **To evaluate the Dual IRT model:**
    `python part b/demo_dual_irt.py`

Each script will output its results and display relevant plots, such as accuracy curves and negative log-likelihood over epochs.

## Project Structure

```
.
├── data/                             # All data files for the project
├── ensemble.py                       # Main ensemble script combining all models
├── item_response.py                  # Item Response Theory model implementation
├── knn.py                            # K-Nearest Neighbors model implementation
├── neural_network.py                 # AutoEncoder Neural Network model implementation
├── matrix_factorization.py           # Matrix Factorization implementation
├── part b/                           # Additional models and demo scripts
│   ├── DualIRT.py                    # Dual IRT model combining Question and Subject IRT
│   ├── demo_dual_irt.py              # Demo script for Dual IRT
│   ├── question_irt.py               # Question-based IRT model
│   └── subject_irt.py                # Subject-based IRT model
└── utils.py                          # Utility functions for data loading and evaluation
```
