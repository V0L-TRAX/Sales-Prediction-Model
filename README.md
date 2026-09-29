# Sales Prediction Model

## 📌 Project Title

**End-to-End Sales Prediction and Forecasting Model in Python**

---

## 🎯 Project Objective

The primary objective of this project is to build a machine-learning based sales prediction pipeline using Python. The project demonstrates how historical sales-related data can be prepared, analyzed, modeled, evaluated, and used to generate predictions for future sales.

---

## ⚠️ Problem Statement

Historical sales data can contain numerical and categorical information that must be properly prepared before machine-learning models can be trained. Choosing suitable regression models and evaluating their predictions with reliable metrics is essential for producing meaningful sales forecasts.

This project addresses these requirements through a structured, step-by-step sales prediction workflow.

---

## 📊 Dataset Description

The project uses a **synthetic educational sales dataset** created for this assignment. The dataset contains **1,000 records** with numerical and categorical features and a **Sales** target variable.

### Dataset Contents

- Historical sales-related records
- Numerical features
- Categorical features
- Sales target variable
- Training and testing data for model evaluation

---

## 🔧 Project Workflow

The sales prediction pipeline follows these major steps:

1. Load the historical sales dataset
2. Inspect and prepare the data
3. Preprocess numerical and categorical features
4. Split the data into training and testing sets
5. Train multiple regression models
6. Generate sales predictions
7. Evaluate model performance
8. Visualize actual versus predicted sales
9. Compare model performance

---

## 🤖 Machine Learning Models

The project trains and compares the following regression models:

- **Linear Regression**
- **Decision Tree Regressor**
- **Random Forest Regressor**

---

## 📈 Model Evaluation

Model performance is evaluated using:

- **R² Score** — measures how well the model explains variation in sales
- **Mean Absolute Error (MAE)** — measures the average absolute prediction error
- **Root Mean Squared Error (RMSE)** — measures prediction error while giving greater weight to larger errors

The evaluation results are saved in `model_results.csv`.

---

## 📊 Visualizations

The project generates visual outputs to help understand model performance:

- **Actual vs Predicted Sales** — compares model predictions with actual test-set values
- **Model Comparison Chart** — compares the performance of the regression models

Generated files include:

- `actual_vs_predicted.png`
- `model_comparison.png`

---

## 📁 Project Files

- `sales_dataset.csv` — sample historical sales dataset
- `train_model.py` — preprocessing, model training, prediction, and evaluation
- `app.py` — Streamlit dashboard for live predictions
- `model_results.csv` — model evaluation results
- `actual_vs_predicted.csv` — test-set predictions
- `actual_vs_predicted.png` — actual versus predicted sales chart
- `model_comparison.png` — model comparison chart
- `requirements.txt` — required Python packages
- `Sales_Prediction_Task_03.pptx` — project presentation

---

## ▶️ How to Run

### Run with Python

Open a terminal in the project folder.

Install the required packages:

```bash
pip install -r requirements.txt
```

Train and evaluate the models:

```bash
python train_model.py
```

Launch the interactive dashboard:

```bash
streamlit run app.py
```

### Run with Jupyter Notebook

Install Jupyter if required:

```bash
pip install notebook
```

Start Jupyter:

```bash
python -m notebook
```

Open the included Jupyter notebook and select:

**Kernel → Restart Kernel and Run All**

---

## 📌 Expected Outcome

After running the project, you will have trained regression models, evaluation metrics, prediction results, comparison charts, and an interactive dashboard for exploring sales predictions.

---

## 📝 Conclusion

This project demonstrates an end-to-end machine-learning workflow for sales prediction, from historical data preparation and model training to evaluation, visualization, and interactive prediction.
