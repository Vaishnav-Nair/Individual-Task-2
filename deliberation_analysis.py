import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, learning_curve
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
import matplotlib.pyplot as plt

# ---------- Load & clean ----------
df = pd.read_csv('data/telco_churn.csv')
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())

# Keep gender as a raw column for fairness analysis before encoding
gender_raw = df['gender'].copy()

df_model = df.drop('customerID', axis=1).copy()
df_model['Churn'] = df_model['Churn'].map({'Yes': 1, 'No': 0})

cat_cols = df_model.select_dtypes(include='object').columns.tolist()
le_dict = {}
for col in cat_cols:
    le = LabelEncoder()
    df_model[col] = le.fit_transform(df_model[col])
    le_dict[col] = le

X = df_model.drop('Churn', axis=1)
y = df_model['Churn']

X_train, X_test, y_train, y_test, gender_train, gender_test = train_test_split(
    X, y, gender_raw, test_size=0.25, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)
X_all_s = scaler.fit_transform(X)  # for cross-val on full data

# =========================================================
# PART 2.1: Cross-validation for unbiased evaluation
# =========================================================
print("="*60)
print("2.1 STRATIFIED K-FOLD CROSS-VALIDATION")
print("="*60)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

knn = KNeighborsClassifier(n_neighbors=15)
mlp = MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=1000, random_state=42, early_stopping=True)

knn_cv_scores = cross_val_score(knn, X_all_s, y, cv=skf, scoring='roc_auc')
mlp_cv_scores = cross_val_score(mlp, X_all_s, y, cv=skf, scoring='roc_auc')

print(f"kNN 5-fold ROC-AUC: {knn_cv_scores}")
print(f"kNN mean: {knn_cv_scores.mean():.4f} (+/- {knn_cv_scores.std():.4f})")
print(f"\nNeural Net 5-fold ROC-AUC: {mlp_cv_scores}")
print(f"Neural Net mean: {mlp_cv_scores.mean():.4f} (+/- {mlp_cv_scores.std():.4f})")

# =========================================================
# PART 2.2: Learning curves
# =========================================================
print("\n" + "="*60)
print("2.2 LEARNING CURVES")
print("="*60)

train_sizes = np.linspace(0.1, 1.0, 8)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

for ax, model, name in zip(axes, [knn, mlp], ['kNN', 'Neural Network']):
    sizes, train_scores, test_scores = learning_curve(
        model, X_all_s, y, cv=skf, train_sizes=train_sizes,
        scoring='roc_auc', random_state=42, n_jobs=-1
    )
    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    test_mean = test_scores.mean(axis=1)
    test_std = test_scores.std(axis=1)

    ax.plot(sizes, train_mean, 'o-', color='#4C72B0', label='Training score')
    ax.fill_between(sizes, train_mean - train_std, train_mean + train_std, alpha=0.15, color='#4C72B0')
    ax.plot(sizes, test_mean, 'o-', color='#DD8452', label='Cross-validation score')
    ax.fill_between(sizes, test_mean - test_std, test_mean + test_std, alpha=0.15, color='#DD8452')
    ax.set_title(f'Learning Curve: {name}')
    ax.set_xlabel('Training set size')
    ax.set_ylabel('ROC-AUC')
    ax.legend(loc='lower right')
    ax.set_ylim(0.6, 1.0)

    print(f"\n{name} learning curve (train size -> CV ROC-AUC):")
    for s, m in zip(sizes, test_mean):
        print(f"  n={int(s):5d}  CV ROC-AUC={m:.4f}")

plt.tight_layout()
plt.savefig('learning_curves.png', dpi=150)
print("\nSaved learning_curves.png")

# =========================================================
# PART 2.3: Fairness analysis with Fairlearn
# =========================================================
print("\n" + "="*60)
print("2.3 FAIRNESS ANALYSIS (Fairlearn) — sensitive feature: gender")
print("="*60)

from fairlearn.metrics import MetricFrame, selection_rate, demographic_parity_difference, equalized_odds_difference

# Fit final models on train set for fairness eval on test set
knn.fit(X_train_s, y_train)
mlp.fit(X_train_s, y_train)

pred_knn = knn.predict(X_test_s)
pred_mlp = mlp.predict(X_test_s)

for name, preds in [('kNN', pred_knn), ('Neural Network', pred_mlp)]:
    print(f"\n--- {name} ---")
    mf = MetricFrame(
        metrics={'accuracy': accuracy_score, 'selection_rate': selection_rate},
        y_true=y_test, y_pred=preds, sensitive_features=gender_test
    )
    print(mf.by_group)

    dpd = demographic_parity_difference(y_test, preds, sensitive_features=gender_test)
    eod = equalized_odds_difference(y_test, preds, sensitive_features=gender_test)
    print(f"Demographic parity difference: {dpd:.4f}")
    print(f"Equalized odds difference: {eod:.4f}")

print("\nDone.")
