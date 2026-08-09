# Walmart Store Sales Forecasting

A machine learning project using Python to forecast weekly department-level sales across 45 Walmart stores, supporting better inventory and demand planning.

## Business Problem

The goal of this project was to predict weekly sales using historical sales, store characteristics, seasonal patterns, economic conditions, and promotional factors to support inventory planning and reduce stockouts and excess inventory.

## Tools Used

Python | Pandas | NumPy | Matplotlib | Seaborn | Scikit-learn | XGBoost | Streamlit

## Project Approach

- Cleaned, validated, and integrated 421K+ sales records
- Performed exploratory data analysis to identify sales patterns and key drivers
- Engineered seasonal and historical features including Lag52 and RollingMean4
- Used a time-based train-test split to prevent data leakage
- Compared a seasonal baseline, Linear Regression, Random Forest, and XGBoost
- Evaluated model performance using MAE, RMSE, and R²

## Key Results

**XGBoost** achieved the best overall performance:

- **MAE:** $1,278.39
- **RMSE:** $2,693.53
- **R²:** 0.9850

Historical sales features such as RollingMean4 and Lag52 were among the strongest predictors, while economic variables had lower predictive importance.

## Business Impact

The model can support more accurate demand forecasting, helping improve inventory planning, reduce stockouts and excess inventory, and support staffing and promotional decisions.

## Deployment

The final XGBoost model was packaged using Joblib and prepared for deployment through a Streamlit application for generating weekly sales forecasts.
