
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

DATA_FILE = "sales_dataset.csv"

df = pd.read_csv(DATA_FILE)
print("Dataset shape:", df.shape)
print("\nMissing values before preprocessing:")
print(df.isna().sum())

X = df.drop(columns=["Sales"])
y = df["Sales"]

numeric_features = ["Month", "Quantity", "Unit_Price", "Advertising_Spend",
                    "Discount_Percent", "Customers"]
categorical_features = ["Region", "Product_Category", "Sales_Channel"]

preprocessor = ColumnTransformer([
    ("num", SimpleImputer(strategy="median"), numeric_features),
    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ]), categorical_features)
])

models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(max_depth=8, random_state=42),
    "Random Forest": RandomForestRegressor(
        n_estimators=250, max_depth=12, random_state=42, n_jobs=-1
    )
}

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

results = []

for name, model in models.items():
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)

    r2 = r2_score(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))

    results.append([name, r2, mae, rmse])
    print(f"\n{name}")
    print(f"R²   : {r2:.4f}")
    print(f"MAE  : {mae:,.2f}")
    print(f"RMSE : {rmse:,.2f}")

results_df = pd.DataFrame(results, columns=["Model", "R2_Score", "MAE", "RMSE"])
results_df.to_csv("model_results.csv", index=False)
print("\nResults saved to model_results.csv")
