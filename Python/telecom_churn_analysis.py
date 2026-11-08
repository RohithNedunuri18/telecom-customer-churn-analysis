# %% Step 1: Import libraries
# pandas/numpy for data, seaborn for charts, scikit-learn for modelling.
import numpy as np, pandas as pd
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, ConfusionMatrixDisplay
sns.set_theme(style="whitegrid"); pd.set_option("display.width", 100); pd.set_option("display.max_columns", 12)
print("Libraries loaded")

# %% Step 2: Load the dataset
# %% Step 2: Load the REAL dataset

df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Convert TotalCharges into numeric format
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

# Display dataset information
print(df.head())
print("\nShape:", df.shape)
print("\nMissing values:")
print(df.isnull().sum())
# Remove rows with missing TotalCharges
df = df.dropna(subset=["TotalCharges"])

print("Cleaned dataset shape:", df.shape)

# %% Step 3: Overall churn rate and data quality
# Establish the baseline churn rate and confirm there are no gaps.
print("Missing values:", int(df.isna().sum().sum()))
rate = (df["Churn"] == "Yes").mean()
print(f"Overall churn rate: {rate:.1%}")
print(df["Churn"].value_counts().to_string())

# %% Step 4: Why do customers leave? Churn by key drivers
# Compare churn rate across contract, internet service, tech support and payment method.
fig, ax = plt.subplots(2, 2, figsize=(10, 7))
for a, col in zip(ax.ravel(), ["Contract", "InternetService", "TechSupport", "PaymentMethod"]):
    (df.groupby(col)["Churn"].apply(lambda s: (s == "Yes").mean() * 100)).sort_values().plot(kind="barh", ax=a, color="#E45756")
    a.set_title(f"Churn % by {col}"); a.set_xlabel("%"); a.set_ylabel("")
plt.tight_layout(); plt.show()
print(df.groupby("Contract")["Churn"].apply(lambda s: round((s == "Yes").mean() * 100, 1)).to_string())

# %% Step 5: Tenure and charges
# Churned customers tend to be newer and pay more per month.
fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
sns.kdeplot(data=df, x="tenure", hue="Churn", fill=True, ax=ax[0]); ax[0].set_title("Tenure (months)")
sns.kdeplot(data=df, x="MonthlyCharges", hue="Churn", fill=True, ax=ax[1]); ax[1].set_title("Monthly Charges")
plt.tight_layout(); plt.show()
print(df.groupby("Churn")[["tenure", "MonthlyCharges"]].mean().round(1).to_string())

# %% Step 6: Prepare features for modelling
# One-hot encode categorical columns and split into train and test sets.
y = (df["Churn"] == "Yes").astype(int)
X = pd.get_dummies(df.drop(columns=["customerID", "Churn"]), drop_first=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)
print("Train:", X_train.shape, "Test:", X_test.shape)

# %% Step 7: Train and compare models
# Logistic regression is the interpretable baseline; random forest captures interactions.
lr = LogisticRegression(max_iter=2000).fit(X_train, y_train)
rf = RandomForestClassifier(n_estimators=300, max_depth=8, random_state=42).fit(X_train, y_train)
for name, m in [("Logistic Regression", lr), ("Random Forest", rf)]:
    print(f"{name}: ROC-AUC = {roc_auc_score(y_test, m.predict_proba(X_test)[:, 1]):.3f}")
print("\n", classification_report(y_test, rf.predict(X_test), target_names=["Stayed", "Churned"]))

# %% Step 8: Confusion matrix and feature importance
# See where the model errs and which features matter most.
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ConfusionMatrixDisplay.from_estimator(rf, X_test, y_test, ax=ax[0], cmap="Blues", colorbar=False); ax[0].set_title("Confusion Matrix (RF)")
imp = pd.Series(rf.feature_importances_, X.columns).sort_values().tail(8)
imp.plot(kind="barh", ax=ax[1], color="#4C78A8", title="Top Churn Drivers")
plt.tight_layout(); plt.show()

# %% Step 9: Risk segmentation and retention strategies
# Score every customer, bucket by risk and attach an action to each bucket.
df["ChurnRisk"] = rf.predict_proba(X)[:, 1]
df["Risk"] = pd.cut(df["ChurnRisk"], [0, .3, .6, 1], labels=["Low", "Medium", "High"])
print(df["Risk"].value_counts().to_string())
hi = df[df["Risk"] == "High"]
print(f"\nHigh-risk customers: {len(hi)} | Monthly revenue at risk: {hi['MonthlyCharges'].sum():,.0f}")
print("\nRetention playbook:")
print("- Month-to-month + fibre: offer 12-month contract with a loyalty discount")
print("- No tech support: bundle free 3-month TechSupport trial")
print("- Electronic-check payers: incentivise auto-pay")
print("- Tenure < 12 months: onboarding call and early-life check-ins")
df.to_csv("telco_churn_cleaned.csv", index=False)