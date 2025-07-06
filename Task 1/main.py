import pandas as pd

# Load the dataset
df = pd.read_csv("Delinquency_prediction_dataset(Delinquency_prediction_dataset).csv")

# Basic info
missing_values = df.isnull().sum()
missing_values = missing_values[missing_values > 0].sort_values(ascending=False)

# Basic statistics for numerical columns
numeric_desc = df.describe()

# Preview unique values in monthly repayment columns
monthly_cols = [col for col in df.columns if col.startswith("Month_")]
monthly_values = {col: df[col].unique() for col in monthly_cols}

# Correlation with target
correlation_with_target = df.corr(numeric_only=True)["Delinquent_Account"].sort_values(ascending=False)

# Top risk indicators (excluding the target itself)
top_risks = correlation_with_target.drop("Delinquent_Account").head(5)

# Identify anomalies
anomalies = df[(df["Account_Tenure"] == 0) | (df["Income"] < 5000)]

missing_values, numeric_desc, monthly_values, top_risks, anomalies[["Income", "Loan_Balance", "Account_Tenure", "Delinquent_Account"]].head()
