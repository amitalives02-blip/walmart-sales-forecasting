from pathlib import Path
from datetime import timedelta

import joblib
import numpy as np
import pandas as pd
import streamlit as st


# ---------------------------------------------------------
# PAGE SETTINGS
# ---------------------------------------------------------
st.set_page_config(
    page_title="Walmart Weekly Sales Forecast",
    page_icon="📊",
    layout="wide"
)


# ---------------------------------------------------------
# LOAD THE SAVED MODEL BUNDLE
# ---------------------------------------------------------
MODEL_PATH = Path(__file__).parent / "walmart_xgb_bundle.joblib"


@st.cache_resource
def load_model_bundle():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "The file 'walmart_xgb_bundle.joblib' was not found. "
            "Make sure it is in the same folder as app.py."
        )

    return joblib.load(MODEL_PATH)


try:
    bundle = load_model_bundle()
except Exception as error:
    st.error(f"Unable to load the model: {error}")
    st.stop()


model = bundle["model"]
feature_columns = bundle["feature_columns"]
sales_history = bundle["sales_history"].copy()
store_information = bundle["store_information"].copy()
store_dept_lookup = bundle["store_dept_lookup"].copy()
default_values = bundle["default_values"]
overall_training_average = bundle["overall_training_average"]
model_metrics = bundle.get("model_metrics", {})

sales_history["Date"] = pd.to_datetime(sales_history["Date"])


# ---------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------
def determine_store_type(store_row):
    """
    Walmart Type A is the reference category.
    Type_B = 1 means Type B.
    Type_C = 1 means Type C.
    """

    if int(store_row["Type_B"]) == 1:
        return "Type B"

    if int(store_row["Type_C"]) == 1:
        return "Type C"

    return "Type A"


def get_store_department_average(store, department):
    result = store_dept_lookup[
        (store_dept_lookup["Store"] == store)
        & (store_dept_lookup["Dept"] == department)
    ]

    if result.empty:
        return float(overall_training_average)

    return float(result["StoreDeptAvg"].iloc[0])


def get_historical_features(store, department, forecast_date):
    """
    Creates:
    - Lag52: sales from approximately 52 weeks before forecast date
    - RollingMean4: average sales from the latest four earlier weeks
    """

    history = sales_history[
        (sales_history["Store"] == store)
        & (sales_history["Dept"] == department)
        & (sales_history["Date"] < forecast_date)
    ].sort_values("Date")

    store_dept_average = get_store_department_average(
        store,
        department
    )

    if history.empty:
        return (
            store_dept_average,
            store_dept_average,
            store_dept_average
        )

    target_lag_date = forecast_date - timedelta(weeks=52)

    lag_candidates = history[
        history["Date"] <= target_lag_date
    ].sort_values("Date")

    if lag_candidates.empty:
        lag52 = store_dept_average
    else:
        lag52 = float(
            lag_candidates["Weekly_Sales"].iloc[-1]
        )

    recent_four_weeks = history.tail(4)

    if recent_four_weeks.empty:
        rolling_mean_4 = store_dept_average
    else:
        rolling_mean_4 = float(
            recent_four_weeks["Weekly_Sales"].mean()
        )

    return lag52, rolling_mean_4, store_dept_average


def create_prediction_dataframe(
    store,
    department,
    forecast_date,
    is_holiday,
    temperature,
    fuel_price,
    markdown1,
    markdown2,
    markdown3,
    markdown4,
    markdown5,
    cpi,
    unemployment
):
    store_row = store_information[
        store_information["Store"] == store
    ]

    if store_row.empty:
        raise ValueError(
            f"Store {store} was not found in the training data."
        )

    store_row = store_row.iloc[0]

    size = float(store_row["Size"])
    type_b = int(store_row["Type_B"])
    type_c = int(store_row["Type_C"])

    lag52, rolling_mean_4, store_dept_average = (
        get_historical_features(
            store,
            department,
            forecast_date
        )
    )

    total_markdown = (
        markdown1
        + markdown2
        + markdown3
        + markdown4
        + markdown5
    )

    iso_calendar = forecast_date.isocalendar()

    input_data = {
        "Store": store,
        "Dept": department,
        "IsHoliday": int(is_holiday),
        "Size": size,
        "Temperature": temperature,
        "Fuel_Price": fuel_price,
        "MarkDown1": markdown1,
        "MarkDown2": markdown2,
        "MarkDown3": markdown3,
        "MarkDown4": markdown4,
        "MarkDown5": markdown5,
        "CPI": cpi,
        "Unemployment": unemployment,
        "Year": forecast_date.year,
        "Month": forecast_date.month,
        "Week": int(iso_calendar.week),
        "Quarter": int(
            ((forecast_date.month - 1) // 3) + 1
        ),
        "DayOfYear": forecast_date.timetuple().tm_yday,
        "Total_MarkDown": total_markdown,
        "Lag52": lag52,
        "RollingMean4": rolling_mean_4,
        "StoreDeptAvg": store_dept_average,
        "Type_B": type_b,
        "Type_C": type_c
    }

    input_df = pd.DataFrame([input_data])

    missing_columns = [
        column
        for column in feature_columns
        if column not in input_df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing model features: {missing_columns}"
        )

    input_df = input_df[feature_columns]

    details = {
        "Store Type": determine_store_type(store_row),
        "Store Size": size,
        "Lag 52": lag52,
        "Rolling Mean 4": rolling_mean_4,
        "Store-Department Average": store_dept_average
    }

    return input_df, details


# ---------------------------------------------------------
# APP TITLE
# ---------------------------------------------------------
st.title("🛒 Walmart Weekly Sales Forecast")

st.write(
    """
    This app predicts weekly sales for a selected Walmart
    store and department using the trained XGBoost regression model.
    """
)


# ---------------------------------------------------------
# MODEL PERFORMANCE
# ---------------------------------------------------------
if model_metrics:
    st.subheader("Model Performance")

    metric_col1, metric_col2, metric_col3 = st.columns(3)

    metric_col1.metric(
        "MAE",
        f"${model_metrics.get('MAE', 0):,.2f}"
    )

    metric_col2.metric(
        "RMSE",
        f"${model_metrics.get('RMSE', 0):,.2f}"
    )

    metric_col3.metric(
        "R²",
        f"{model_metrics.get('R2', 0):.4f}"
    )


# ---------------------------------------------------------
# STORE AND DEPARTMENT SELECTION
# ---------------------------------------------------------
st.subheader("Forecast Inputs")

store_list = sorted(
    store_information["Store"].astype(int).unique()
)

selected_store = st.selectbox(
    "Select Store",
    options=store_list
)

department_list = sorted(
    store_dept_lookup[
        store_dept_lookup["Store"] == selected_store
    ]["Dept"].astype(int).unique()
)

if len(department_list) == 0:
    st.warning(
        "No departments were found for the selected store."
    )
    st.stop()

selected_department = st.selectbox(
    "Select Department",
    options=department_list
)


# ---------------------------------------------------------
# DATE AND HOLIDAY INPUTS
# ---------------------------------------------------------
latest_history_date = sales_history["Date"].max().date()
default_forecast_date = (
    latest_history_date + timedelta(weeks=1)
)

date_col1, date_col2 = st.columns(2)

with date_col1:
    forecast_date = st.date_input(
        "Forecast Date",
        value=default_forecast_date
    )

with date_col2:
    is_holiday = st.checkbox(
        "Holiday Week",
        value=False
    )

forecast_date = pd.Timestamp(forecast_date)


# ---------------------------------------------------------
# ECONOMIC INPUTS
# ---------------------------------------------------------
st.markdown("### Economic Conditions")

economic_col1, economic_col2 = st.columns(2)

with economic_col1:
    temperature = st.number_input(
        "Temperature",
        value=float(default_values["Temperature"]),
        format="%.2f"
    )

    fuel_price = st.number_input(
        "Fuel Price",
        value=float(default_values["Fuel_Price"]),
        format="%.3f"
    )

with economic_col2:
    cpi = st.number_input(
        "Consumer Price Index",
        value=float(default_values["CPI"]),
        format="%.3f"
    )

    unemployment = st.number_input(
        "Unemployment Rate",
        value=float(default_values["Unemployment"]),
        format="%.3f"
    )


# ---------------------------------------------------------
# MARKDOWN INPUTS
# ---------------------------------------------------------
st.markdown("### Promotional Markdown Values")

markdown_col1, markdown_col2, markdown_col3 = st.columns(3)

with markdown_col1:
    markdown1 = st.number_input(
        "MarkDown 1",
        min_value=0.0,
        value=float(default_values["MarkDown1"])
    )

    markdown2 = st.number_input(
        "MarkDown 2",
        min_value=0.0,
        value=float(default_values["MarkDown2"])
    )

with markdown_col2:
    markdown3 = st.number_input(
        "MarkDown 3",
        min_value=0.0,
        value=float(default_values["MarkDown3"])
    )

    markdown4 = st.number_input(
        "MarkDown 4",
        min_value=0.0,
        value=float(default_values["MarkDown4"])
    )

with markdown_col3:
    markdown5 = st.number_input(
        "MarkDown 5",
        min_value=0.0,
        value=float(default_values["MarkDown5"])
    )


# ---------------------------------------------------------
# PREDICTION BUTTON
# ---------------------------------------------------------
st.divider()

if st.button(
    "Generate Sales Forecast",
    type="primary",
    use_container_width=True
):
    try:
        prediction_df, forecast_details = (
            create_prediction_dataframe(
                store=selected_store,
                department=selected_department,
                forecast_date=forecast_date,
                is_holiday=is_holiday,
                temperature=temperature,
                fuel_price=fuel_price,
                markdown1=markdown1,
                markdown2=markdown2,
                markdown3=markdown3,
                markdown4=markdown4,
                markdown5=markdown5,
                cpi=cpi,
                unemployment=unemployment
            )
        )

        prediction = float(
            model.predict(prediction_df)[0]
        )

        prediction = max(prediction, 0)

        st.success("Forecast generated successfully.")

        st.metric(
            "Predicted Weekly Sales",
            f"${prediction:,.2f}"
        )

        st.subheader("Forecast Details")

        detail_col1, detail_col2, detail_col3 = st.columns(3)

        detail_col1.metric(
            "Store Type",
            forecast_details["Store Type"]
        )

        detail_col1.metric(
            "Store Size",
            f"{forecast_details['Store Size']:,.0f}"
        )

        detail_col2.metric(
            "Sales 52 Weeks Earlier",
            f"${forecast_details['Lag 52']:,.2f}"
        )

        detail_col2.metric(
            "Recent 4-Week Average",
            f"${forecast_details['Rolling Mean 4']:,.2f}"
        )

        detail_col3.metric(
            "Store-Department Average",
            f"${forecast_details['Store-Department Average']:,.2f}"
        )

        st.subheader("Model Input Features")

        feature_display = prediction_df.T.reset_index()
        feature_display.columns = [
            "Feature",
            "Value"
        ]

        st.dataframe(
            feature_display,
            use_container_width=True,
            hide_index=True
        )

    except Exception as error:
        st.error(f"Prediction failed: {error}")


# ---------------------------------------------------------
# APP EXPLANATION
# ---------------------------------------------------------
with st.expander("How does this forecast work?"):
    st.write(
        """
        The model uses store information, department information,
        economic indicators, markdown promotions, calendar features
        and historical sales behaviour.

        Lag52 represents sales from approximately the same period
        one year earlier.

        RollingMean4 represents average sales from the latest four
        available historical weeks.

        StoreDeptAvg represents the historical average sales for the
        selected store and department.
        """
    )