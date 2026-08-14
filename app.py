import streamlit as st
import pandas as pd
import joblib

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Insurance Fraud AI",
    page_icon="🔍",
    layout="wide"
)

# ============================================================
# LOAD SAVED MODEL
# ============================================================

MODEL_PATH = "models/fraud_model.pkl"
PREPROCESSOR_PATH = "models/preprocessor.pkl"
THRESHOLD_PATH = "models/fraud_threshold.pkl"

model = joblib.load(MODEL_PATH)
preprocessor = joblib.load(PREPROCESSOR_PATH)
threshold = joblib.load(THRESHOLD_PATH)
df = pd.read_csv("data/processed/fraud_oracle_clean.csv")

# ============================================================
# TITLE
# ============================================================

st.title("🔍 Insurance Claim Fraud Detection")
st.write("AI-powered insurance claim fraud prediction")

st.success("Model loaded successfully!")

st.write("Fraud Detection Threshold:", threshold)


# ============================================================
# CLAIM INPUTS
# ============================================================

st.header("📋 Insurance Claim Details")

st.subheader("Basic Claim Information")

col1, col2, col3 = st.columns(3)

with col1:
    month = st.selectbox(
        "Month",
        ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
         "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    )

with col2:
    week_of_month = st.number_input(
        "Week of Month",
        min_value=1,
        max_value=5,
        value=1
    )

with col3:
    day_of_week = st.selectbox(
        "Day of Week",
        ["Monday", "Tuesday", "Wednesday",
         "Thursday", "Friday", "Saturday", "Sunday"]
    )
    st.subheader("Vehicle Information")

col1, col2, col3 = st.columns(3)

with col1:
    make = st.selectbox(
        "Vehicle Make",
        sorted(df["Make"].unique()) if "Make" in df.columns else []
    )

with col2:
    accident_area = st.selectbox(
        "Accident Area",
        sorted(df["AccidentArea"].unique()) if "AccidentArea" in df.columns else []
    )

with col3:
    vehicle_category = st.selectbox(
        "Vehicle Category",
        sorted(df["VehicleCategory"].unique()) if "VehicleCategory" in df.columns else []
    )
    df = pd.read_csv("data/processed/fraud_oracle_clean.csv")
    model = joblib.load(MODEL_PATH)
preprocessor = joblib.load(PREPROCESSOR_PATH)
threshold = joblib.load(THRESHOLD_PATH)

df = pd.read_csv("data/processed/fraud_oracle_clean.csv")
st.subheader("👤 Claimant Information")

col1, col2, col3 = st.columns(3)

with col1:
    sex = st.selectbox(
        "Sex",
        sorted(df["Sex"].unique())
    )

with col2:
    marital_status = st.selectbox(
        "Marital Status",
        sorted(df["MaritalStatus"].unique())
    )

with col3:
    age = st.number_input(
        "Age",
        min_value=0,
        max_value=100,
        value=30
    )
    st.subheader("📄 Policy & Vehicle Details")

col1, col2, col3 = st.columns(3)

with col1:
    vehicle_price = st.selectbox(
        "Vehicle Price",
        sorted(df["VehiclePrice"].unique())
    )

with col2:
    age_of_vehicle = st.selectbox(
        "Age of Vehicle",
        sorted(df["AgeOfVehicle"].unique())
    )

with col3:
    age_of_policyholder = st.selectbox(
        "Age of Policy Holder",
        sorted(df["AgeOfPolicyHolder"].unique())
    )

col1, col2, col3 = st.columns(3)

with col1:
    policy_type = st.selectbox(
        "Policy Type",
        sorted(df["PolicyType"].unique())
    )

with col2:
    base_policy = st.selectbox(
        "Base Policy",
        sorted(df["BasePolicy"].unique())
    )

with col3:
    fault = st.selectbox(
        "Fault",
        sorted(df["Fault"].unique())
    )
    st.subheader("📝 Claim Processing Details")

col1, col2, col3 = st.columns(3)

with col1:
    police_report_filed = st.selectbox(
        "Police Report Filed",
        sorted(df["PoliceReportFiled"].unique())
    )

with col2:
    witness_present = st.selectbox(
        "Witness Present",
        sorted(df["WitnessPresent"].unique())
    )

with col3:
    agent_type = st.selectbox(
        "Agent Type",
        sorted(df["AgentType"].unique())
    )

col1, col2, col3 = st.columns(3)

with col1:
    number_of_supplements = st.selectbox(
        "Number of Supplements",
        sorted(df["NumberOfSuppliments"].unique())
    )

with col2:
    address_change_claim = st.selectbox(
        "Address Change Claim",
        sorted(df["AddressChange_Claim"].unique())
    )

with col3:
    number_of_cars = st.selectbox(
        "Number of Cars",
        sorted(df["NumberOfCars"].unique())
    )
    st.subheader("📅 Claim Details")

col1, col2, col3 = st.columns(3)

with col1:
    day_of_week_claimed = st.selectbox(
    "Day of Week Claimed",
    sorted(df["DayOfWeekClaimed"].unique()),
    index=1,
    key="day_of_week_claimed"
)

with col2:
    month_claimed = st.selectbox(
    "Month Claimed",
    sorted(df["MonthClaimed"].unique()),
    index=1,
    key="month_claimed"
)

with col3:
    week_of_month_claimed = st.number_input(
        "Week of Month Claimed",
        min_value=1,
        max_value=5,
        value=1
    )

col1, col2, col3 = st.columns(3)

with col1:
    year = st.number_input(
        "Year",
        min_value=int(df["Year"].min()),
        max_value=int(df["Year"].max()),
        value=int(df["Year"].median())
    )
    import streamlit as st
import pandas as pd
import joblib
# ============================================================
# ADDITIONAL MODEL INPUTS
# ============================================================

st.subheader("⚙️  Claim Information")

col1, col2, col3 = st.columns(3)

with col1:
    policy_number = st.number_input(
        "Policy Number",
        min_value=1,
        value=1,
        step=1,
        key="policy_number"
    )

with col2:
    rep_number = st.selectbox(
        "Representative Number",
        sorted(df["RepNumber"].unique()),
        key="rep_number"
    )

with col3:
    deductible = st.selectbox(
        "Deductible",
        sorted(df["Deductible"].unique()),
        key="deductible"
    )

col1, col2, col3 = st.columns(3)

with col1:
    driver_rating = st.selectbox(
        "Driver Rating",
        sorted(df["DriverRating"].unique()),
        key="driver_rating"
    )

with col2:
    days_policy_accident = st.selectbox(
        "Days Policy Accident",
        sorted(df["Days_Policy_Accident"].unique()),
        key="days_policy_accident"
    )

with col3:
    days_policy_claim = st.selectbox(
        "Days Policy Claim",
        sorted(df["Days_Policy_Claim"].unique()),
        key="days_policy_claim"
    )

col1, col2, col3 = st.columns(3)

with col1:
    past_number_of_claims = st.selectbox(
        "Past Number of Claims",
        sorted(df["PastNumberOfClaims"].unique()),
        key="past_number_of_claims"
    )

with col2:
    age_policyholder_clean = st.selectbox(
        "Age of Policy Holder (Clean)",
        sorted(df["AgeOfPolicyHolder_Clean"].unique()),
        key="age_policyholder_clean"
    )

with col3:
    number_cars_clean = st.selectbox(
        "Number of Cars (Clean)",
        sorted(df["NumberOfCars_Clean"].unique()),
        key="number_cars_clean"
    )
    # ============================================================
# CREATE MODEL INPUT
# ============================================================

claim_data = pd.DataFrame([{
    "Month": month,
    "WeekOfMonth": week_of_month,
    "DayOfWeek": day_of_week,
    "Make": make,
    "AccidentArea": accident_area,
    "DayOfWeekClaimed": day_of_week_claimed,
    "MonthClaimed": month_claimed,
    "WeekOfMonthClaimed": week_of_month_claimed,
    "Sex": sex,
    "MaritalStatus": marital_status,
    "Age": age,
    "Fault": fault,
    "PolicyType": policy_type,
    "VehicleCategory": vehicle_category,
    "VehiclePrice": vehicle_price,
    "PolicyNumber": policy_number,
    "RepNumber": rep_number,
    "Deductible": deductible,
    "DriverRating": driver_rating,
    "Days_Policy_Accident": days_policy_accident,
    "Days_Policy_Claim": days_policy_claim,
    "PastNumberOfClaims": past_number_of_claims,
    "AgeOfVehicle": age_of_vehicle,
    "AgeOfPolicyHolder": age_of_policyholder,
    "PoliceReportFiled": police_report_filed,
    "WitnessPresent": witness_present,
    "AgentType": agent_type,
    "NumberOfSuppliments": number_of_supplements,
    "AddressChange_Claim": address_change_claim,
    "NumberOfCars": number_of_cars,
    "Year": year,
    "BasePolicy": base_policy,
    "AgeOfPolicyHolder_Clean": age_policyholder_clean,
    "NumberOfCars_Clean": number_cars_clean
}])

st.subheader("🔎 Claim Data Preview")

st.dataframe(claim_data)
# ============================================================
# FRAUD PREDICTION
# ============================================================

st.subheader("🚨 Fraud Detection")

if st.button("🔍 Analyze Claim", key="analyze_claim"):

    # Transform the claim data using the saved preprocessor
    claim_encoded = preprocessor.transform(claim_data)

    # Get fraud probability
    fraud_probability = model.predict_proba(claim_encoded)[0][1]

    # Apply the saved threshold
    prediction = int(fraud_probability >= threshold)

    # Display probability
    st.metric(
        "Fraud Probability",
        f"{fraud_probability * 100:.2f}%"
    )

    if prediction == 1:
        st.error("🚨 FRAUDULENT CLAIM")
        st.warning(
            f"The model estimates a {fraud_probability * 100:.2f}% "
            "probability of fraud."
        )
    else:
        st.success("✅ LIKELY LEGITIMATE CLAIM")
        st.info(
            f"The model estimates a {fraud_probability * 100:.2f}% "
            "probability of fraud."
        )
