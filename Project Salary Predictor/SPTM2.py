import streamlit as st
import pandas as pd
import numpy as np
import pickle

# ----------------- LOAD MODEL, ENCODERS & SCALER -----------------
with open("best_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("label_encoders.pkl", "rb") as f:
    label_encoders = pickle.load(f)

# ----------------- STREAMLIT UI SETTINGS -----------------
st.set_page_config(page_title="💼 Salary Estimator", layout="centered")
st.title("💼 Smart Salary Prediction System")
st.markdown("Predict employee salary based on job role, experience, and other factors using your trained ML model.")
mode = st.radio("Choose Mode", ["🌐 Individual Prediction", "📂 Bulk Upload"])

# ----------------- HELPER FUNCTION -----------------
def preprocess_input_bulk(df):
    df = df.copy()
    df.columns = df.columns.str.strip().str.lower()

    for col in df.columns:
        if col in label_encoders:  # Categorical
            df[col] = df[col].fillna("Unknown").astype(str)
            known_classes = set(label_encoders[col].classes_)
            df[col] = df[col].apply(lambda x: x if x in known_classes else "Unknown")
            df[col] = label_encoders[col].transform(df[col])
        else:  # Numerical
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].fillna(df[col].mean())
    return df

# ----------------- INDIVIDUAL PREDICTION -----------------
if mode == "🌐 Individual Prediction":
    st.subheader("🎯 Enter Employee Details")

    col1, col2 = st.columns(2)
    with col1:
        education = st.selectbox("🎓 Education Level", ["", "High School", "Bachelors", "Masters", "PhD"])
        experience = st.number_input("📈 Years of Experience", 0, 40, step=1)
        job_title = st.selectbox("💼 Job Title", ["", "Data Scientist", "Analyst", "Software Engineer", "Manager"])
        certs = st.slider("📜 Certifications", 0, 10, 1)
        working_hours = st.number_input("⏰ Weekly Working Hours", 0, 100, step=1)

    with col2:
        industry = st.selectbox("🏭 Industry", ["", "IT", "Healthcare", "Education", "Finance"])
        location = st.selectbox("📍 Location", ["", "New York", "San Francisco", "London", "Bangalore"])
        company_size = st.selectbox("🏢 Company Size", ["", "Small", "Medium", "Large"])
        age = st.number_input("👤 Age", 18, 70, step=1)

    if st.button("🚀 Predict Salary"):
        missing_fields = []

        # Validate for empty dropdowns
        if education == "":
            missing_fields.append("Education Level")
        if job_title == "":
            missing_fields.append("Job Title")
        if industry == "":
            missing_fields.append("Industry")
        if location == "":
            missing_fields.append("Location")
        if company_size == "":
            missing_fields.append("Company Size")

        if missing_fields:
            st.error(f"❌ The following fields are required: {', '.join(missing_fields)}")
        else:
            try:
                input_data = {
                    'education_level': [education],
                    'years_experience': [experience],
                    'job_title': [job_title],
                    'industry': [industry],
                    'location': [location],
                    'company_size': [company_size],
                    'certifications': [certs],
                    'age': [age],
                    'working_hours': [working_hours]
                }

                df_input = pd.DataFrame(input_data)

                # Encode categorical fields
                for col in label_encoders:
                    df_input[col] = label_encoders[col].transform(df_input[col])

                salary_pred = model.predict(df_input)[0]
                st.success(f"💰 Predicted Salary: ₹{salary_pred:,.2f}")

            except Exception as e:
                st.error(f"❌ Prediction failed: {str(e)}")

# ----------------- BULK UPLOAD -----------------
else:
    st.subheader("📂 Upload CSV File for Bulk Prediction")

    file = st.file_uploader("Upload your CSV or Excel file", type=["csv", "xlsx"])

    if file:
        try:
            # Load file
            if file.name.endswith(".csv"):
                df = pd.read_csv(file)
            else:
                df = pd.read_excel(file)

            required_columns = ['education_level', 'years_experience', 'job_title', 'industry',
                                'location', 'company_size', 'certifications', 'age', 'working_hours']

            if not all(col in df.columns for col in required_columns):
                st.error("❌ Missing one or more required columns.")
            else:
                df_input = df[required_columns].copy()
                df_input.columns = df_input.columns.str.strip().str.lower()

                remarks_list = []

                # Fill missing values with mean/mode and collect remarks
                for idx, row in df_input.iterrows():
                    row_remark = []
                    for col in df_input.columns:
                        val = row[col]

                        if pd.isna(val) or val == "":
                            if col in label_encoders:
                                mode_val = df_input[col].mode()[0]
                                df_input.at[idx, col] = mode_val
                                row_remark.append(f"{col} filled with mode '{mode_val}'")
                            else:
                                mean_val = df_input[col].mean()
                                df_input.at[idx, col] = mean_val
                                row_remark.append(f"{col} filled with mean {mean_val:.2f}")
                        else:
                            # No missing value
                            continue
                    remarks_list.append("; ".join(row_remark) if row_remark else "✅ No Imputation")

                df_input["remarks"] = remarks_list  # Add remarks before transformation

                # Encode categorical columns
                for col in label_encoders:
                    known_classes = set(label_encoders[col].classes_)
                    unseen = set(df_input[col].unique()) - known_classes
                    if unseen:
                        st.error(f"❌ Unseen values in column '{col}': {unseen}")
                        st.stop()
                    df_input[col] = label_encoders[col].transform(df_input[col])

                # Prepare data and predict
                X = df_input.drop(columns=["remarks"])
                df["Predicted_Salary"] = model.predict(X)
                df["Remarks"] = df_input["remarks"]

                st.success("✅ Salary prediction completed!")
                st.dataframe(df)

                @st.cache_data
                def convert_df(dframe):
                    return dframe.to_csv(index=False).encode("utf-8")

                st.download_button("📥 Download Results", convert_df(df), "predicted_salaries_with_remarks.csv", "text/csv")

        except Exception as e:
            st.error(f"❌ Error while processing file: {str(e)}")
