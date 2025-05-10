import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score

# 1. Load and Explore
# ---------------------

# Load dataset
df = pd.read_csv("house_prices.csv")

# Basic info
print("First 5 rows:\n", df.head())
print("\nData Summary:\n", df.describe())
print("\nMissing values:\n", df.isnull().sum())

# Handle missing values (basic approach: drop rows with any missing values)
df = df.dropna()

# Distribution plots
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.histplot(df['Size'], kde=True)
plt.title("Distribution of Size")
plt.subplot(1, 2, 2)
sns.histplot(df['Price'], kde=True)
plt.title("Distribution of Price")
plt.show()

# Outlier detection using boxplot
plt.figure(figsize=(12, 5))
sns.boxplot(data=df[['Size', 'Price']])
plt.title("Boxplot for Size and Price")
plt.show()

# 2. Data Preprocessing
# -----------------------

# Define features and target
X = df[['Size', 'Location', 'Number of Rooms']]
y = df['Price']

# Column types
numerical_features = ['Size', 'Number of Rooms']
categorical_features = ['Location']

# Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(drop='first'), categorical_features)
    ]
)

# 3. Feature Selection (via Correlation)
# ----------------------------------------

# Select only numeric columns for correlation heatmap
numeric_df = df.select_dtypes(include=[np.number])

# Correlation heatmap
plt.figure(figsize=(8, 5))
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# 4. Model Training
# -------------------

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Full pipeline
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# Train model
model.fit(X_train, y_train)

# 5. Model Evaluation
# ---------------------

# Predict
y_pred = model.predict(X_test)

# Metrics
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"\nModel Evaluation:")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R² Score: {r2:.4f}")

# 6. Predictions Comparison
# ---------------------------

results = pd.DataFrame({'Actual Price': y_test, 'Predicted Price': y_pred})
print("\nSample Predictions:\n", results.head())

# Plot actual vs predicted
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_pred)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')  # reference line
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Prices")
plt.show()

# 7. Feature Insights (coefficients)
# -------------------------------------

# Get feature names after encoding
ohe = model.named_steps['preprocessor'].named_transformers_['cat']
encoded_location = ohe.get_feature_names_out(['Location'])

all_features = numerical_features + list(encoded_location)
coefficients = model.named_steps['regressor'].coef_

coef_df = pd.DataFrame({'Feature': all_features, 'Coefficient': coefficients})
print("\nFeature Coefficients:\n", coef_df.sort_values(by='Coefficient', ascending=False))
