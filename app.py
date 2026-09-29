
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(page_title="Sales Prediction Dashboard", page_icon="📈", layout="wide")

@st.cache_resource
def train_model():
    df = pd.read_csv("sales_dataset.csv")
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

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("model", RandomForestRegressor(
            n_estimators=250, max_depth=12, random_state=42, n_jobs=-1
        ))
    ])
    model.fit(X, y)
    return model, df

model, df = train_model()

st.title("📈 Sales Prediction Dashboard")
st.write("Enter sales-related information to estimate expected sales.")

col1, col2, col3 = st.columns(3)

with col1:
    month = st.slider("Month", 1, 12, 6)
    region = st.selectbox("Region", sorted(df["Region"].unique()))
    product = st.selectbox("Product Category", sorted(df["Product_Category"].unique()))

with col2:
    channel = st.selectbox("Sales Channel", sorted(df["Sales_Channel"].unique()))
    quantity = st.number_input("Quantity", min_value=1, max_value=1000, value=100)
    unit_price = st.number_input("Unit Price (₹)", min_value=50.0, max_value=20000.0, value=2000.0)

with col3:
    advertising = st.number_input("Advertising Spend (₹)", min_value=0.0, max_value=500000.0, value=50000.0)
    discount = st.slider("Discount (%)", 0.0, 50.0, 10.0)
    customers = st.number_input("Customers", min_value=1, max_value=10000, value=500)

if st.button("🔮 Predict Sales", use_container_width=True):
    input_df = pd.DataFrame([{
        "Month": month,
        "Region": region,
        "Product_Category": product,
        "Sales_Channel": channel,
        "Quantity": quantity,
        "Unit_Price": unit_price,
        "Advertising_Spend": advertising,
        "Discount_Percent": discount,
        "Customers": customers
    }])

    prediction = model.predict(input_df)[0]
    st.success(f"Estimated Sales: ₹{prediction:,.2f}")

st.divider()
st.subheader("Dataset Preview")
st.dataframe(df.head(10), use_container_width=True)

st.subheader("Project Metrics")
results_path = "model_results.csv"
try:
    results = pd.read_csv(results_path)
    st.dataframe(results.style.format({
        "R2_Score": "{:.4f}",
        "MAE": "₹{:,.2f}",
        "RMSE": "₹{:,.2f}"
    }), use_container_width=True)
except FileNotFoundError:
    st.info("Run train_model.py first to create model_results.csv.")
