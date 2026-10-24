# Bank Term Deposit Subscription Prediction (Imbalanced ML)

---
Direct telemarketing for bank term deposits is expensive and inefficient because only **11.3%** of contacted customers actually subscribe (severe class imbalance: 88.7% `no` vs. 11.3% `yes`). Standard classification models trained on this data suffer from the **Accuracy Paradox**—achieving ~89% accuracy while missing over 70% of potential subscribers.
This project builds an operationally deployable machine learning pipeline that solves class imbalance without data leakage, identifying significantly more prospective depositors while minimizing total operational costs.
---
## 🔄 End-to-End Workflow
```
[Raw Data: 41,188 rows]
        │
        ▼
[1. Data Cleaning & Feature Engineering]
   • Dropped 12 duplicate records (41,176 remaining).
   • Converted 'pdays' to binary 'was_contacted' (0 if 999, else 1) and dropped raw 'pdays'.
   • Binned 'age' into generational cohorts: Student (0–25), Adult (25–60), Senior (60+).
   • Grouped basic education levels into 'basic.school'.
   • Pruned 'default' (near-zero variance) and strictly dropped 'duration' (anti-leakage).
        │
        ▼
[2. Stratified Train-Test Split]
   • 80% Training (32,940) / 20% Held-Out Testing (8,236).
        │
        ▼
[3. Leakage-Free Preprocessing & Resampling Pipeline (imblearn)]
   • Numerical: Median Imputer + StandardScaler.
   • Categorical: Constant Imputer ('unknown') + OneHotEncoder(drop='first').
   • SMOTE Resampling: Executed STRICTLY inside training folds during Cross-Validation.
        │
        ▼
[4. Hyperparameter Tuning & Selection]
   • 3-Fold Stratified GridSearchCV optimizing for F1-Score.
   • Models: Logistic Regression, Random Forest, XGBoost.
        │
        ▼
[5. Evaluation on Unseen Test Set & Cost Analysis]
```
---
## 🏆 Benchmark Results
Evaluated on the completely unseen held-out test set ($N = 8,236$):
| Strategy | Model | Recall (Sensitivity) | Precision | F1-Score | AUC | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **SMOTE** | **Random Forest** | **59.38%** | **46.26%** | **0.5201** | **0.7531** | 🥇 **Champion** |
| **SMOTE** | **XGBoost** | 55.60% | 48.68% | 0.5191 | 0.7410 | 🥈 Runner-Up |
| **SMOTE** | **Logistic Regression** | 65.95% | 35.31% | 0.4600 | 0.7535 | High False Positives |
| Baseline | Random Forest | 28.88% | 62.76% | 0.3956 | 0.6305 | Conservative |
| Baseline | XGBoost | 25.00% | 62.20% | 0.3566 | 0.6128 | Conservative |
| Baseline | Logistic Regression | 21.23% | 64.80% | 0.3198 | 0.5960 | Baseline |
### Champion Model Configuration
- **Model:** `RandomForestClassifier` with SMOTE
- **Best Hyperparameters:** `{'max_depth': 10, 'n_estimators': 50}`
- **Subscribers Detected:** **551 out of 928** (vs. only 197 by baseline Logistic Regression) — **2.8x increase in lead capture**.
---
