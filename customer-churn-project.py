
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score
from sklearn.metrics import recall_score, f1_score, confusion_matrix
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sns.set_theme(style="whitegrid")

# 1. DATA LOADING

df = pd.read_csv("C:/predictive analysis/WA_Fn-UseC_-Telco-Customer-Churn-dataset1.csv")

print("FIRST FIVE RECORDS")
print(df.head())

print("\nDATASET INFORMATION")
print(df.info())

print("\nDATASET SHAPE")
print(df.shape)

print("\nDESCRIPTIVE STATISTICS")
print(df.describe())

print("\nCATEGORICAL DESCRIPTION")
print(df.describe(include="object"))

print("\nMISSING VALUES")
print(df.isnull().sum())

print("\nDUPLICATE RECORDS")
print(df.duplicated().sum())

# 2. DATA PREPROCESSING

df.columns = df.columns.str.strip()

for col in df.select_dtypes(include="object").columns:
    df[col] = df[col].str.strip()

df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"], errors="coerce"
)

for col in df.select_dtypes(include="number").columns:
    df[col] = df[col].fillna(df[col].mean())

for col in df.select_dtypes(include="object").columns:
    if df[col].isnull().any():
        df[col] = df[col].fillna(df[col].mode()[0])

print("\nMISSING VALUES AFTER PREPROCESSING")
print(df.isnull().sum())

print("\nFINAL DATASET INFORMATION")
print(df.info())

print("\nFINAL DESCRIPTIVE STATISTICS")
print(df.describe())

# 3. FEATURE ENGINEERING

service_cols = [
    "PhoneService", "MultipleLines", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport",
    "StreamingTV", "StreamingMovies"
]

df["ServiceCount"] = sum(
    (df[col] == "Yes").astype(int) for col in service_cols
)

# OBJECTIVE 1: OVERALL CHURN RATE

churn_rate = (df["Churn"] == "Yes").mean() * 100

print("\nOBJECTIVE 1: OVERALL CHURN RATE")
print("Churn Rate:", round(churn_rate, 2), "%")

sns.countplot(data=df, x="Churn")
plt.title("Customer Churn Distribution")
plt.xlabel("Churn")
plt.ylabel("Number of Customers")
plt.show()

# OBJECTIVE 2: CONTRACT-WISE CHURN

contract_churn = pd.crosstab(
    df["Contract"], df["Churn"], normalize="index"
) * 100

print("\nOBJECTIVE 2: CHURN BY CONTRACT")
print(contract_churn.round(2))

contract_churn.plot(kind="bar", figsize=(8, 5))
plt.title("Contract Type vs Churn")
plt.ylabel("Churn Percentage")
plt.xticks(rotation=15)
plt.tight_layout()
plt.show()

# OBJECTIVE 3: TENURE AND MONTHLY CHARGES

print("\nOBJECTIVE 3: CUSTOMER TENURE AND CHARGES")
print(df.groupby("Churn")[["tenure", "MonthlyCharges"]].mean())

plt.figure(figsize=(7, 5))
sns.boxplot(data=df, x="Churn", y="tenure")
plt.title("Tenure vs Churn")
plt.show()

plt.figure(figsize=(7, 5))
sns.boxplot(data=df, x="Churn", y="MonthlyCharges")
plt.title("Monthly Charges vs Churn")
plt.show()

# CORRELATION ANALYSIS

numeric_df = df.select_dtypes(include="number")

plt.figure(figsize=(9, 6))
sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()

# OBJECTIVE 4: CUSTOMER CHARACTERISTICS

features = [
    "gender", "SeniorCitizen", "Partner", "Dependents",
    "InternetService", "OnlineSecurity", "TechSupport",
    "PaymentMethod"
]

for col in features:
    churn_analysis = pd.crosstab(
        df[col], df["Churn"], normalize="index"
    ) * 100

    print("\nCHURN ANALYSIS:", col)
    print(churn_analysis.round(2))

print("\nAVERAGE SERVICE COUNT BY CHURN")
print(df.groupby("Churn")["ServiceCount"].mean())

sns.boxplot(data=df, x="Churn", y="ServiceCount")
plt.title("Service Count vs Churn")
plt.show()

# 4. CLASSIFICATION DATA PREPARATION

X = df.drop(columns=["Churn", "customerID"])
y = df["Churn"].map({"No": 0, "Yes": 1})

X = pd.get_dummies(X, drop_first=True, dtype=int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# OBJECTIVE 5: FOUR MACHINE LEARNING MODELS

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(
        max_depth=8, random_state=42
    ),
    "Naive Bayes": GaussianNB(),
    "SVM": SVC()
}

results = []
trained_models = {}

for name, model in models.items():

    if name == "Decision Tree":
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
    else:
        model.fit(X_train_scaled, y_train)
        pred = model.predict(X_test_scaled)

    trained_models[name] = model

    accuracy = accuracy_score(y_test, pred)
    precision = precision_score(y_test, pred, zero_division=0)
    recall = recall_score(y_test, pred, zero_division=0)
    f1 = f1_score(y_test, pred, zero_division=0)

    results.append([
        name, accuracy, precision, recall, f1
    ])

    print("\nMODEL:", name)
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, pred))

# 5. MODEL COMPARISON

results = pd.DataFrame(results, columns=[
    "Model", "Accuracy", "Precision", "Recall", "F1-Score"
])

print("\nFINAL MODEL COMPARISON")
print(results.round(3).to_string(index=False))

best_name = results.loc[
    results["F1-Score"].idxmax(), "Model"
]

print("\nBEST MODEL:", best_name)

results.set_index("Model").plot(
    kind="bar", figsize=(10, 5)
)

plt.title("Comparison of Four ML Models")
plt.ylabel("Performance Score")
plt.ylim(0, 1)
plt.xticks(rotation=15)
plt.tight_layout()
plt.show()

# BEST MODEL CONFUSION MATRIX

best_model = trained_models[best_name]

if best_name == "Decision Tree":
    best_pred = best_model.predict(X_test)
else:
    best_pred = best_model.predict(X_test_scaled)

sns.heatmap(
    confusion_matrix(y_test, best_pred),
    annot=True, fmt="d", cmap="Blues"
)

plt.title("Best Model: " + best_name)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# FEATURE IMPORTANCE USING DECISION TREE

tree = trained_models["Decision Tree"]

importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": tree.feature_importances_
})

importance = importance.sort_values(
    "Importance", ascending=False
).head(10)

print("\nTOP 10 IMPORTANT FEATURES")
print(importance)

sns.barplot(data=importance, x="Importance", y="Feature")
plt.title("Important Features for Churn Prediction")
plt.show()

# 6. UNIT II: LINEAR REGRESSION

reg_X = df.drop(columns=[
    "MonthlyCharges", "TotalCharges",
    "Churn", "customerID"
])

reg_X = pd.get_dummies(reg_X, drop_first=True, dtype=int)
reg_y = df["MonthlyCharges"]

xr_train, xr_test, yr_train, yr_test = train_test_split(
    reg_X, reg_y, test_size=0.2, random_state=42
)

reg_model = LinearRegression()
reg_model.fit(xr_train, yr_train)

reg_pred = reg_model.predict(xr_test)

print("\nLINEAR REGRESSION RESULTS")
print("MAE:", mean_absolute_error(yr_test, reg_pred))
print("MSE:", mean_squared_error(yr_test, reg_pred))
print("RMSE:", np.sqrt(mean_squared_error(yr_test, reg_pred)))
print("R2 Score:", r2_score(yr_test, reg_pred))

plt.scatter(yr_test, reg_pred, alpha=0.5)
plt.xlabel("Actual Monthly Charges")
plt.ylabel("Predicted Monthly Charges")
plt.title("Linear Regression: Actual vs Predicted")
plt.show()

print("\nCUSTOMERGUARD AI PROJECT COMPLETED")
