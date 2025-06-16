# 1. Import libraries
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from tpot import TPOTClassifier
from operator import itemgetter

# Set a different port for Dask to avoid conflicts
os.environ['DASK_SCHEDULER_PORT'] = '8788'  # Change to an unused port

# 2. Load dataset
transfusion = pd.read_csv(r"C:\Users\GK\OneDrive\Desktop\GK\imp docs\Give Life_ Predict Blood Donations\datasets\transfusion.data")

# 3. Rename columns for simplicity
transfusion.columns = ['Recency', 'Frequency', 'Monetary', 'Time', 'target']

# 4. Data check
print(transfusion.info())
print(transfusion.describe())
print(transfusion['target'].value_counts(normalize=True).round(3))

# 5. Split data into training and testing sets
X = transfusion.drop('target', axis=1)
y = transfusion['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42
)

# 6. TPOT AutoML - Raw data
tpot = TPOTClassifier(
    max_time_mins=30,  # Set a time limit instead of generations
    population_size=20,
    verbose=2,
    random_state=42,
    n_jobs=1  # Use 1 core to avoid Dask port conflict in Spyder
)
tpot.fit(X_train, y_train)

# 7. Evaluate TPOT
y_pred_tpot = tpot.predict_proba(X_test)[:, 1]
tpot_auc_score = roc_auc_score(y_test, y_pred_tpot)
print(f"\nTPOT AUC Score (raw): {tpot_auc_score:.4f}")

# 8. Feature variance and normalization
col_to_normalize = X_train.var().idxmax()
X_train_normed = X_train.copy()
X_test_normed = X_test.copy()

# Check if the column to normalize is not empty
if col_to_normalize:
    X_train_normed[col_to_normalize + '_log'] = np.log1p(X_train_normed[col_to_normalize])
    X_test_normed[col_to_normalize + '_log'] = np.log1p(X_test_normed[col_to_normalize])
    X_train_normed.drop(columns=col_to_normalize, inplace=True)
    X_test_normed.drop(columns=col_to_normalize, inplace=True)

# 9. Logistic Regression
logreg = LogisticRegression(solver='liblinear', random_state=42)
logreg.fit(X_train_normed, y_train)
y_pred_logreg = logreg.predict_proba(X_test_normed)[:, 1]
logreg_auc_score = roc_auc_score(y_test, y_pred_logreg)
print(f"\nLogistic Regression AUC Score: {logreg_auc_score:.4f}")

# 10. Retrain TPOT on normalized data
tpot.fit(X_train_normed, y_train)
y_pred_tpot_normed = tpot.predict_proba(X_test_normed)[:, 1]
tpot_auc_normed = roc_auc_score(y_test, y_pred_tpot_normed)
print(f"\nTPOT AUC Score (normalized): {tpot_auc_normed:.4f}")

# 11. Compare models
results = [
    ('TPOT (normalized)', tpot_auc_normed),
    ('Logistic Regression', logreg_auc_score)
]

print("\nModel Performance (sorted by AUC):")
for name, score in sorted(results, key=itemgetter(1), reverse=True):
    print(f"{name}: {score:.4f}")

# 12. Optional: Save pipeline manually (no .export() due to TPOT version)
# Write pipeline code to a file if needed
with open("best_pipeline_code.txt", "w") as f:
    f.write(str(tpot.fitted_pipeline_))
