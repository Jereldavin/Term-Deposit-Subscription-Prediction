# Term Deposit Subscription Prediction

A machine learning project that predicts whether a bank customer will subscribe to a term deposit based on demographic, contact, campaign, and economic information.

## Project Overview

This project uses the **Bank Marketing** dataset to analyze customer characteristics and build classification models for predicting the target variable `y`:

- `yes` — the customer subscribed to a term deposit
- `no` — the customer did not subscribe to a term deposit

The dataset contains information such as age, job, education, contact method, campaign details, previous campaign outcomes, and macroeconomic indicators.

## Dataset

The notebook expects the dataset file below in the working directory:

```text
bank-additional-full.csv
```

The CSV file is loaded using a semicolon separator:

```python
df = pd.read_csv("bank-additional-full.csv", sep=";")
```

## Project Workflow

The notebook follows these main steps:

1. **Data preparation**
   - Import Python and machine learning libraries.
   - Load the bank marketing dataset.

2. **Data understanding**
   - Inspect the first rows and dataset structure.
   - Review summary statistics and categorical value counts.
   - Check missing values and duplicated records.

3. **Data cleaning and feature engineering**
   - Remove duplicate rows.
   - Create `was_contacted` from `pdays` to indicate whether the customer was previously contacted.
   - Group customers into `Student`, `Adult`, and `Senior` age groups.
   - Consolidate basic education categories into `basic.school`.
   - Remove selected columns after feature creation.

4. **Exploratory data analysis**
   - Analyze categorical and numerical feature distributions.
   - Examine relationships between customer attributes and subscription outcomes.
   - Create visualizations using Matplotlib and Seaborn.

5. **Model preparation and evaluation**
   - Encode categorical variables and scale numerical variables where appropriate.
   - Address class imbalance using sampling techniques such as under-sampling, SMOTE, and SMOTETomek.
   - Train and compare classification models.
   - Evaluate performance with accuracy, precision, recall, F1-score, ROC-AUC, confusion matrices, and classification reports.

## Models and Techniques

The notebook imports and evaluates several machine learning approaches, including:

- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier
- XGBoost Classifier
- LightGBM Classifier

It also uses hyperparameter-search tools such as `GridSearchCV` and `RandomizedSearchCV`.

## Features

The dataset includes customer and campaign attributes such as:

- Age, job, marital status, and education
- Housing loan and personal loan status
- Contact method, month, and day of the week
- Call duration and campaign contact count
- Previous campaign history and outcome
- Employment variation rate and consumer confidence indicators
- Euribor 3-month rate and number of employees

Additional engineered features include:

- `was_contacted`
- `age_group`
- Consolidated education categories

## Requirements

Install the required dependencies with:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy imbalanced-learn xgboost lightgbm jupyter
```

## How to Run

1. Clone the repository:

   ```bash
   git clone https://github.com/Jereldavin/Term-Deposit-Subscription-Prediction.git
   cd Term-Deposit-Subscription-Prediction
   ```

2. Place `bank-additional-full.csv` in the project directory.

3. Launch Jupyter Notebook or open the notebook in Google Colab:

   ```bash
   jupyter notebook
   ```

4. Open `code notebook.ipynb` and run the cells from top to bottom.

## Important Notes

- The target variable is highly imbalanced, with substantially more customers not subscribing than subscribing.
- Sampling methods should be applied only to the training data to avoid data leakage.
- `duration` can be highly predictive, but it is only known after a contact has occurred. Its use should be considered carefully when designing a real-world campaign prediction system.
- The notebook currently contains an exploratory cell that attempts to drop `pdays` after it has already been removed or is unavailable in the current dataframe state. Use `errors="ignore"` or ensure the column is dropped only once if rerunning the notebook.

## Project Goal

The goal is to identify customers who are more likely to subscribe to a term deposit, helping financial institutions improve campaign targeting and reduce unnecessary customer contacts.

## License

No license has been specified for this repository.
