# CMPE-255 Data Mining
# Credit Card Fraud Detection

This project aims to detect fraudulent credit card transactions using supervised machine learning algorithms. It uses a real-world dataset from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and includes end-to-end pipeline workflows for data preprocessing, feature engineering, model training, threshold tuning, and evaluation.

## Project Structure


## Dataset

- **Source**: [Kaggle Credit Card Fraud Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Description**: Contains 284,807 transactions, of which 492 are fraudulent (0.172%)
- **Features**:
  - V1 to V28: PCA-transformed features
  - `Time`, `Amount`: raw features
  - `Class`: target label (1 = fraud, 0 = non-fraud)

## Methodology

- **Preprocessing**
  - Remove exact duplicates
  - Log-transform and scale `Amount`
  - Handle class imbalance using sampling

- **Feature Engineering**
  - Extract transaction hour from `Time`
  - Compute time since last transaction
  - Encode time cyclically to capture daily patterns

- **Models Used**
  - Logistic Regression
  - Decision Tree
  - Random Forest (with manual tuning)

- **Evaluation**
  - Classification reports
  - AUPRC (Area Under Precision-Recall Curve)
  - Threshold tuning for best F1-score
  - Confusion matrices and precision-recall curve analysis

## How to Run

1. **Clone the repository**:

2. **Install dependencies**:

3. **Place the dataset**:
- Download `creditcard.csv` from Kaggle and place it in the `data/` directory.

4. **Run the notebook**:
- Open `group_project_9_version7.ipynb` in Jupyter or VS Code and execute all cells.




