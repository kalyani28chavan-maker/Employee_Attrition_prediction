
import streamlit as st
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(
    page_title="Employee Attrition Prediction",
    page_icon="🏢",
    layout="wide"
)

st.title("🏢 Employee Attrition Prediction")
st.write("AI Powered HR Analytics Dashboard")
st.divider()

DATA_URL = "https://raw.githubusercontent.com/IamVasil/IBM-HR-Analytics-Employee-Attrition-Performance/main/WA_Fn-UseC_-HR-Employee-Attrition.csv"

@st.cache_resource
def train_model():
    df = pd.read_csv(DATA_URL)

    features = [
        "Age", "MonthlyIncome", "YearsAtCompany",
        "OverTime", "Department", "JobRole",
        "MaritalStatus", "DistanceFromHome"
    ]

    X = df[features]
    y = df["Attrition"]

    numeric_features = [
        "Age", "MonthlyIncome",
        "YearsAtCompany", "DistanceFromHome"
    ]

    categorical_features = [
        "OverTime", "Department",
        "JobRole", "MaritalStatus"
    ]

    preprocessor = ColumnTransformer([
        ("numeric", "passthrough", numeric_features),
        ("categorical", OneHotEncoder(handle_unknown="ignore"),
         categorical_features)
    ])

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", RandomForestClassifier(
            n_estimators=150,
            random_state=42,
            class_weight="balanced"
        ))
    ])

    pipeline.fit(X, y)

    return pipeline, df

try:
    with st.spinner("Training AI model... Please wait"):
        model, df = train_model()

    st.success("AI Model trained successfully!")

    st.subheader("Employee Details")

    col1, col2 = st.columns(2)

    with col1:
        age = st.slider("Employee Age", 18, 60, 30)
        income = st.number_input(
            "Monthly Income", 1000, 20000, 5000
        )
        years = st.slider("Years at Company", 0, 40, 5)
        overtime = st.selectbox("OverTime", ["No", "Yes"])

    with col2:
        department = st.selectbox(
            "Department",
            sorted(df["Department"].unique())
        )

        job_role = st.selectbox(
            "Job Role",
            sorted(df["JobRole"].unique())
        )

        marital = st.selectbox(
            "Marital Status",
            sorted(df["MaritalStatus"].unique())
        )

        distance = st.slider(
            "Distance From Home", 1, 29, 5
        )

    if st.button("Predict Employee Attrition"):
        employee = pd.DataFrame([{
            "Age": age,
            "MonthlyIncome": income,
            "YearsAtCompany": years,
            "OverTime": overtime,
            "Department": department,
            "JobRole": job_role,
            "MaritalStatus": marital,
            "DistanceFromHome": distance
        }])

        prediction = model.predict(employee)[0]
        probability = model.predict_proba(employee)[0]

        attrition_index = list(model.classes_).index("Yes")
        risk = probability[attrition_index] * 100

        st.divider()
        st.subheader("Prediction Result")

        st.metric("Employee Attrition Risk", f"{risk:.2f}%")

        if prediction == "Yes":
            st.error("Employee may leave the company.")
        else:
            st.success("Employee may stay in the company.")

        st.caption(
            "This is an AI-based estimate, not a guarantee."
        )

    st.divider()
    st.subheader("Dataset Overview")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Employees", len(df))
    col2.metric("Total Features", len(df.columns))
    col3.metric(
        "Attrition Cases",
        int((df["Attrition"] == "Yes").sum())
    )

except Exception as e:
    st.error("App could not start. Please check the error below.")
    st.exception(e)
st.divider()
st.header("📊 Attrition Analysis")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Department-wise Employees")
    dept_data = df["Department"].value_counts()
    st.bar_chart(dept_data)

with col2:
    st.subheader("Attrition Distribution")
    attrition_data = df["Attrition"].value_counts()
    st.bar_chart(attrition_data)

st.subheader("Job Role-wise Attrition")
role_data = pd.crosstab(df["JobRole"], df["Attrition"])
st.bar_chart(role_data)
st.divider()
st.header("💡 HR Business Insights")

total_employees = len(df)
left_employees = (df["Attrition"] == "Yes").sum()
stayed_employees = (df["Attrition"] == "No").sum()

attrition_rate = (left_employees / total_employees) * 100

col1, col2, col3 = st.columns(3)

col1.metric("Employees Left", left_employees)
col2.metric("Employees Stayed", stayed_employees)
col3.metric("Attrition Rate", f"{attrition_rate:.2f}%")

st.subheader("Business Summary")

st.write(
    f"Out of {total_employees} employees, "
    f"{left_employees} employees have left the company."
)

st.write(
    f"The company has retained {stayed_employees} employees."
)

st.info(
    f"The overall employee attrition rate is {attrition_rate:.2f}%."
)
