import json
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Flux Loan", page_icon="⚡", layout="wide")


@st.cache_resource
def load_assets():
    model = joblib.load("model.pkl")
    with open("meta.json") as f:
        meta = json.load(f)
    return model, meta


model, meta = load_assets()
FEATURES = meta["features"]
CATS = meta["categories"]
EDU = meta["education"]

st.title("⚡ Flux Loan")
st.subheader("Will your loan get approved?")
st.caption(
    "Student project made for learning. Not real financial advice, "
    "and no bank decision depends on it."
)

with st.form("loan_form"):
    st.markdown("### Enter your details")

    c1, c2, c3 = st.columns(3)
    with c1:
        applicant_income = st.number_input("Applicant income", min_value=0.0, value=50000.0, step=1000.0)
        coapplicant_income = st.number_input("Coapplicant income", min_value=0.0, value=0.0, step=1000.0)
        age = st.number_input("Age", min_value=18, max_value=80, value=30)
        dependents = st.number_input("Dependents", min_value=0, max_value=10, value=0)
        credit_score = st.number_input("Credit score", min_value=300, max_value=900, value=700)
        existing_loans = st.number_input("Existing loans", min_value=0, max_value=20, value=0)
    with c2:
        dti = st.number_input("DTI ratio", min_value=0.0, value=0.3, step=0.01, format="%.2f")
        savings = st.number_input("Savings", min_value=0.0, value=20000.0, step=1000.0)
        collateral = st.number_input("Collateral value", min_value=0.0, value=0.0, step=1000.0)
        loan_amount = st.number_input("Loan amount", min_value=0.0, value=200000.0, step=5000.0)
        loan_term = st.number_input("Loan term (months)", min_value=1, max_value=480, value=60)
        education = st.selectbox("Education level", EDU)
    with c3:
        employment = st.selectbox("Employment status", CATS["Employment_Status"])
        employer = st.selectbox("Employer category", CATS["Employer_Category"])
        marital = st.selectbox("Marital status", CATS["Marital_Status"])
        purpose = st.selectbox("Loan purpose", CATS["Loan_Purpose"])
        area = st.selectbox("Property area", CATS["Property_Area"])
        gender = st.selectbox("Gender", CATS["Gender"])

    submitted = st.form_submit_button("Check approval", type="primary")

if submitted:
    d = {
        "Applicant_Income": applicant_income,
        "Coapplicant_Income": coapplicant_income,
        "Age": age,
        "Dependents": dependents,
        "Credit_Score": credit_score,
        "Existing_Loans": existing_loans,
        "DTI_Ratio": dti,
        "Savings": savings,
        "Collateral_Value": collateral,
        "Loan_Amount": loan_amount,
        "Loan_Term": loan_term,
        "Education_Level": education,
        "Employment_Status": employment,
        "Marital_Status": marital,
        "Loan_Purpose": purpose,
        "Property_Area": area,
        "Gender": gender,
        "Employer_Category": employer,
    }

    # Same logic as the Flask /predict route
    row = {f: 0.0 for f in FEATURES}
    for f in FEATURES:
        if f in d and f != "Education_Level":
            try:
                row[f] = float(d[f])
            except (TypeError, ValueError):
                pass
    if d["Education_Level"] in EDU:
        row["Education_Level"] = EDU.index(d["Education_Level"])
    for col in CATS:
        name = f"{col}_{d.get(col)}"
        if name in row:
            row[name] = 1.0

    X = pd.DataFrame([row], columns=FEATURES)
    p = float(model.predict_proba(X)[0][1])
    approved = p >= 0.5

    st.markdown("### Result")
    if approved:
        st.success("✅ Accepted")
    else:
        st.error("❌ Rejected")

    m1, m2, m3 = st.columns(3)
    m1.metric("Approval chance", f"{p * 100:.1f}%")
    m2.metric("Model accuracy", "93.5%")
    m3.metric("DTI ratio", f"{dti:.2f}")
    st.progress(min(max(p, 0.0), 1.0))
