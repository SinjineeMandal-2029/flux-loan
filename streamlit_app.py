import json
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Flux Loan", page_icon="⚡", layout="wide")

ACCURACY = 93.5
LTI_HIGH = 15  # loan amount / total income above this = "too large"


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


def pick(options, preferred):
    return options.index(preferred) if preferred in options else 0


# ---------------- Reasons (ported from index.html) ----------------
def get_reasons(d):
    R = []

    def add(t, title, text, tip=None):
        R.append({"t": t, "title": title, "text": text, "tip": tip})

    income = d["Applicant_Income"] + d["Coapplicant_Income"]
    lti = d["Loan_Amount"] / max(income, 1)
    cs = d["Credit_Score"]
    dti = d["DTI_Ratio"]

    if cs >= 750:
        add("good", f"Strong credit score ({cs})",
            "A high credit score is the biggest factor in approvals.")
    elif cs < 650:
        add("bad", f"Low credit score ({cs})",
            "A low credit score is the biggest reason loans get rejected.",
            "Pay dues on time and keep card usage low to move above 700.")
    else:
        add("warn", f"Average credit score ({cs})",
            "Good, but not strong enough to stand out.",
            "Getting above 750 would improve your chances a lot.")

    if dti <= 0.30:
        add("good", f"Healthy DTI ratio ({dti})",
            "Your debt is small compared with your income.")
    elif dti > 0.40:
        add("bad", f"High DTI ratio ({dti})",
            "Too much of your income already goes to paying debts.",
            "Pay off some existing debt or ask for a smaller loan.")
    else:
        add("warn", f"Borderline DTI ratio ({dti})",
            "Close to the limit lenders prefer.",
            "Bring it below 0.30 if you can.")

    if lti > LTI_HIGH:
        add("bad", "Loan is large for your income",
            "The loan amount is high compared with total income.",
            "Ask for a smaller amount, or add a co-applicant income.")
    elif lti <= 8:
        add("good", "Loan fits your income",
            "The amount asked is comfortable for your income.")

    if d["Existing_Loans"] >= 3:
        add("bad", f"Many existing loans ({d['Existing_Loans']})",
            "Several running loans make a new one riskier.",
            "Close one or two loans before applying.")
    elif d["Existing_Loans"] == 0:
        add("good", "No existing loans",
            "A clean slate helps your application.")

    jobs = f"{d['Employment_Status']} {d['Employer_Category']}".lower()
    if "unemploy" in jobs:
        add("bad", "No steady employment income",
            "Lenders want proof of regular income.",
            "Steady employment or a working co-applicant improves your chances.")
    elif "salar" in d["Employment_Status"].lower():
        add("good", "Steady salaried income",
            "Regular salary gives lenders confidence.")

    if d["Savings"] >= 0.2 * d["Loan_Amount"]:
        add("good", "Good savings cushion",
            "Your savings cover a solid part of the loan.")
    elif d["Savings"] < 0.05 * d["Loan_Amount"]:
        add("bad", "Low savings compared with the loan",
            "Very little savings behind a large loan.",
            "Build up savings or plan a bigger down payment.")

    if d["Collateral_Value"] >= d["Loan_Amount"]:
        add("good", "Collateral covers the loan",
            "The loan is fully backed by your collateral.")
    elif d["Collateral_Value"] < 0.5 * d["Loan_Amount"]:
        add("warn", "Collateral covers less than half",
            "Only a small part of the loan is secured.",
            "Offer higher-value collateral if possible.")

    return R


def show_reason(x):
    body = f"**{x['title']}**\n\n{x['text']}"
    if x["tip"] and x["t"] != "good":
        body += f"\n\n💡 *Tip: {x['tip']}*"
    if x["t"] == "good":
        st.success(body, icon="✅")
    elif x["t"] == "bad":
        st.error(body, icon="❗")
    elif x["t"] == "warn":
        st.warning(body, icon="⚠️")
    else:
        st.info(body, icon="ℹ️")


# ---------------- UI ----------------
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
        st.markdown("**Income & credit**")
        applicant_income = st.number_input("Applicant income", min_value=0.0, value=60000.0, step=1000.0)
        coapplicant_income = st.number_input("Co-applicant income", min_value=0.0, value=0.0, step=1000.0)
        credit_score = st.number_input("Credit score", min_value=300, max_value=900, value=720)
        dti = st.number_input("DTI ratio (0 to 1)", min_value=0.0, value=0.3, step=0.01, format="%.2f")
        savings = st.number_input("Savings", min_value=0.0, value=150000.0, step=1000.0)
        existing_loans = st.number_input("Existing loans", min_value=0, max_value=20, value=1)
    with c2:
        st.markdown("**Loan details**")
        loan_amount = st.number_input("Loan amount", min_value=0.0, value=800000.0, step=5000.0)
        loan_term = st.number_input("Loan term", min_value=1, max_value=480, value=120)
        collateral = st.number_input("Collateral value", min_value=0.0, value=500000.0, step=1000.0)
        purpose = st.selectbox("Loan purpose", CATS["Loan_Purpose"],
                               index=pick(CATS["Loan_Purpose"], "Home"))
        area = st.selectbox("Property area", CATS["Property_Area"],
                            index=pick(CATS["Property_Area"], "Urban"))
    with c3:
        st.markdown("**About the applicant**")
        age = st.number_input("Age", min_value=18, max_value=80, value=32)
        dependents = st.number_input("Dependents", min_value=0, max_value=10, value=1)
        gender = st.selectbox("Gender", CATS["Gender"])
        marital = st.selectbox("Marital status", CATS["Marital_Status"])
        education = st.selectbox("Education level", EDU)
        employment = st.selectbox("Employment status", CATS["Employment_Status"],
                                  index=pick(CATS["Employment_Status"], "Salaried"))
        employer = st.selectbox("Employer category", CATS["Employer_Category"],
                                index=pick(CATS["Employer_Category"], "Private"))

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
    pc = round(p * 100)

    st.markdown("### Result")
    if approved:
        st.success("✅ Loan Accepted — Your profile looks strong for approval.")
    else:
        st.error("❌ Loan Rejected — Here are the likely reasons and what you can improve.")

    m1, m2, m3 = st.columns(3)
    m1.metric("Approval chance", f"{pc}%")
    m2.metric("Model accuracy", f"{ACCURACY}%")
    m3.metric("DTI ratio", f"{dti}")
    st.progress(min(max(p, 0.0), 1.0))

    # Reasons: same sorting and cut-off as the HTML version
    R = get_reasons(d)
    order = {"good": 0, "warn": 1, "bad": 2} if approved else {"bad": 0, "warn": 1, "good": 2}
    R = sorted(R, key=lambda x: order[x["t"]])[:6]
    if not approved and not any(x["t"] == "bad" for x in R):
        R.insert(0, {
            "t": "info",
            "title": "Mixed profile",
            "text": "No single weak point stands out. The combination of your details lowered the score.",
            "tip": "Improving your credit score or asking for a smaller loan is the best place to start.",
        })

    st.markdown("### " + ("Why it looks good" if approved else "Why it was rejected"))
    for x in R:
        show_reason(x)
    st.caption(
        "Reasons are based on common lending rules applied to your details. "
        "This is a student project, not financial advice."
    )
