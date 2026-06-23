import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("fraud_model.pkl")
# with st.expander("Show model details"):
#     st.write(model.feature_names_in_)

# Page Title
st.title(" Credit Card Fraud Detection Dashboard")

# Sidebar
st.sidebar.title(" Project Details")

st.sidebar.info("""
Model: Random Forest

Dataset: Credit Card Fraud Dataset

Output: Fraud / Legitimate

Developer: Pami Gupta
""")

# File Upload
uploaded_file = st.file_uploader(
    "Upload Transaction CSV",
    type=["csv"]
)

if uploaded_file is not None:

    # Read CSV
    df = pd.read_csv(uploaded_file)

    # Data Preview
    st.subheader("📄 Uploaded Data Preview")
    st.dataframe(df.head())

    # Remove target column if present
    if "Class" in df.columns:
        df = df.drop("Class", axis=1)

    # Get expected model features
    expected_cols = list(model.feature_names_in_)

    # Check missing columns
    missing_cols = set(expected_cols) - set(df.columns)

    if missing_cols:
        st.error(f"Missing columns required by model: {list(missing_cols)}")
        st.stop()

    # Keep only required columns
    df_model = df[expected_cols]

    # Predictions
    predictions = model.predict(df_model)

    # Results dataframe
    result_df = df.copy()
    result_df["Prediction"] = predictions

    # Metrics
    fraud_count = int((predictions == 1).sum())
    legit_count = int((predictions == 0).sum())
    total_count = len(predictions)

    st.subheader("📊 Summary")

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Transactions", total_count)
    col2.metric("Fraudulent", fraud_count)
    col3.metric("Legitimate", legit_count)

    fraud_rate = (fraud_count / total_count) * 100 if total_count > 0 else 0

    st.metric("Fraud Rate", f"{fraud_rate:.2f}%")

    # Bar Chart
    st.subheader("📊 Transaction Counts")

    chart_data = pd.DataFrame({
        "Category": ["Legitimate", "Fraudulent"],
        "Count": [legit_count, fraud_count]
    })

    st.bar_chart(chart_data.set_index("Category"))

    # Convert prediction labels
    result_df["Prediction"] = result_df["Prediction"].map(
        {0: "Legitimate", 1: "Fraudulent"}
    )

    # Fraud Transactions
    fraud_df = result_df[result_df["Prediction"] == "Fraudulent"]

    st.subheader("🚨 Detected Fraud Transactions")

    if len(fraud_df) > 0:
        st.dataframe(fraud_df.head(20))
    else:
        st.success("No fraudulent transactions detected.")

    # Full Results
    st.subheader("📋 Prediction Results")
    st.dataframe(result_df)

    # Footer
    st.markdown("---")
    st.caption(
        "Built using Python, Pandas, Scikit-Learn and Streamlit"
    )

    # Download Button
    csv = result_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="📥 Download Results",
        data=csv,
        file_name="fraud_predictions.csv",
        mime="text/csv"
    )