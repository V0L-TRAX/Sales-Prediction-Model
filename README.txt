# Sales Prediction Model

## Objective
Build a machine-learning model to forecast sales from historical sales-related data.

## Included
- `sales_dataset.csv` — sample historical dataset
- `train_model.py` — preprocessing, model training and evaluation
- `app.py` — Streamlit dashboard for live predictions
- `model_results.csv` — evaluation results
- `actual_vs_predicted.csv` — test-set predictions
- `actual_vs_predicted.png` — prediction chart
- `model_comparison.png` — model comparison chart
- `requirements.txt` — Python packages
- `Sales_Prediction_Task_03.pptx` — presentation

## How to run
1. Open a terminal in this folder.
2. Install packages:
   `pip install -r requirements.txt`
3. Train/evaluate:
   `python train_model.py`
4. Launch dashboard:
   `streamlit run app.py`

## Dataset
The included dataset is a synthetic educational dataset designed for this assignment. It contains 1,000 rows and includes numerical and categorical features plus a Sales target.

## Models
- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor

## Evaluation
- R² Score
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
