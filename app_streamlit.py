import json
import boto3
import streamlit as st

st.set_page_config(page_title="Credit Score Prediction", layout="wide")

ENDPOINT_NAME = "credit-endpoint"
REGION        = "us-east-1"

# Urutan fitur harus sama persis dengan FEATURE_NAMES di inference.py
FEATURE_NAMES = [
    "Month", "Age", "Occupation", "Annual_Income", "Monthly_Inhand_Salary",
    "Num_Bank_Accounts", "Num_Credit_Card", "Interest_Rate", "Num_of_Loan",
    "Type_of_Loan", "Delay_from_due_date", "Num_of_Delayed_Payment",
    "Changed_Credit_Limit", "Num_Credit_Inquiries", "Credit_Mix",
    "Outstanding_Debt", "Credit_Utilization_Ratio", "Credit_History_Age",
    "Payment_of_Min_Amount", "Total_EMI_per_month", "Amount_invested_monthly",
    "Payment_Behaviour", "Monthly_Balance",
]

LABEL_DECODE = {0: "Poor", 1: "Standard", 2: "Good"}

OCCUPATIONS = [
    "Accountant", "Architect", "Developer", "Doctor", "Engineer", "Entrepreneur",
    "Journalist", "Lawyer", "Manager", "Mechanic", "Media_Manager", "Musician",
    "Scientist", "Teacher", "Writer",
]
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August"]
LOAN_TYPES = [
    "Auto Loan", "Credit-Builder Loan", "Debt Consolidation Loan", "Home Equity Loan",
    "Mortgage Loan", "Not Specified", "Payday Loan", "Personal Loan", "Student Loan",
]
PAYMENT_BEHAVIOURS = [
    "High_spent_Large_value_payments", "High_spent_Medium_value_payments",
    "High_spent_Small_value_payments", "Low_spent_Large_value_payments",
    "Low_spent_Medium_value_payments", "Low_spent_Small_value_payments",
]

RESULT_STYLE = {
    "Good": st.success,
    "Standard": st.warning,
    "Poor": st.error,
}


@st.cache_resource
def get_runtime_client():
    return boto3.client("sagemaker-runtime", region_name=REGION)


def predict_via_endpoint(raw_input: dict) -> dict:
    # Annual_Income ada di FEATURE_NAMES tapi di-drop oleh preprocessor — isi 0
    row = [raw_input.get(col, 0) for col in FEATURE_NAMES]
    payload = json.dumps({"instances": [row]})

    runtime = get_runtime_client()
    response = runtime.invoke_endpoint(
        EndpointName=ENDPOINT_NAME,
        ContentType="application/json",
        Accept="application/json",
        Body=payload,
    )
    result = json.loads(response["Body"].read().decode("utf-8"))

    # predict_fn returns: {"predictions": [int], "labels": [str], "probabilities": [[float,...]]}
    predicted_class = result["labels"][0]
    probs_list      = result["probabilities"][0]   # [prob_poor, prob_standard, prob_good]
    probabilities   = {LABEL_DECODE[i]: p for i, p in enumerate(probs_list)}

    return {"credit_score": predicted_class, "probabilities": probabilities}


def main():
    st.title("Credit Score Prediction")
    st.markdown("Predict a customer's **credit score class** (`Poor` / `Standard` / `Good`) from their profile.")

    with st.sidebar:
        st.header("About")
        st.info("Predicts credit score class for a financial institution's customers based on their financial profile.")
        st.markdown("**Model:** LightGBM — served via AWS SageMaker endpoint.")

    with st.form("prediction_form"):
        st.subheader("Profile")
        col1, col2, col3 = st.columns(3)

        with col1:
            month                  = st.selectbox("Month", MONTHS)
            age                    = st.number_input("Age", 14, 100, value=30)
            occupation             = st.selectbox("Occupation", OCCUPATIONS)
            monthly_inhand_salary  = st.number_input("Monthly Inhand Salary", 0.0, value=4000.0, step=100.0)
            num_bank_accounts      = st.number_input("Num Bank Accounts", 0, 20, value=4)
            num_credit_card        = st.number_input("Num Credit Card", 0, 20, value=5)
            credit_mix             = st.selectbox("Credit Mix", ["Bad", "Standard", "Good"])
            payment_of_min_amount  = st.selectbox("Payment of Min Amount", ["Yes", "No", "NM"])

        with col2:
            interest_rate          = st.number_input("Interest Rate (%)", 0, 50, value=12)
            num_of_loan            = st.number_input("Num of Loan", 0, 20, value=2)
            loan_types             = st.multiselect("Type of Loan", LOAN_TYPES, default=["Auto Loan"])
            delay_from_due_date    = st.number_input("Delay From Due Date (days)", 0, 100, value=10)
            num_of_delayed_payment = st.number_input("Num of Delayed Payment", 0, 50, value=8)
            changed_credit_limit   = st.number_input("Changed Credit Limit", -50.0, 50.0, value=5.5, step=0.5)
            num_credit_inquiries   = st.number_input("Num Credit Inquiries", 0, 30, value=3)

        with col3:
            outstanding_debt           = st.number_input("Outstanding Debt", 0.0, value=800.0, step=50.0)
            credit_utilization_ratio   = st.number_input("Credit Utilization Ratio (%)", 0.0, 100.0, value=30.0)
            history_years              = st.number_input("Credit History - Years", 0, 50, value=15)
            history_months             = st.number_input("Credit History - Months", 0, 11, value=6)
            total_emi_per_month        = st.number_input("Total EMI per Month", 0.0, value=100.0, step=10.0)
            amount_invested_monthly    = st.number_input("Amount Invested Monthly", 0.0, value=200.0, step=10.0)
            monthly_balance            = st.number_input("Monthly Balance", 0.0, value=350.0, step=10.0)

        payment_behaviour = st.selectbox("Payment Behaviour", PAYMENT_BEHAVIOURS)

        submitted = st.form_submit_button("Predict")

    if submitted:
        raw_input = {
            "Month":                   month,
            "Age":                     age,
            "Occupation":              occupation,
            "Monthly_Inhand_Salary":   monthly_inhand_salary,
            "Num_Bank_Accounts":       num_bank_accounts,
            "Num_Credit_Card":         num_credit_card,
            "Interest_Rate":           interest_rate,
            "Num_of_Loan":             num_of_loan,
            "Type_of_Loan":            ", ".join(loan_types) if loan_types else "Not Specified",
            "Delay_from_due_date":     delay_from_due_date,
            "Num_of_Delayed_Payment":  num_of_delayed_payment,
            "Changed_Credit_Limit":    changed_credit_limit,
            "Num_Credit_Inquiries":    num_credit_inquiries,
            "Credit_Mix":              credit_mix,
            "Outstanding_Debt":        outstanding_debt,
            "Credit_Utilization_Ratio": credit_utilization_ratio,
            "Credit_History_Age":      f"{history_years} Years and {history_months} Months",
            "Payment_of_Min_Amount":   payment_of_min_amount,
            "Total_EMI_per_month":     total_emi_per_month,
            "Amount_invested_monthly": amount_invested_monthly,
            "Payment_Behaviour":       payment_behaviour,
            "Monthly_Balance":         monthly_balance,
        }

        with st.spinner("Calling SageMaker endpoint..."):
            result = predict_via_endpoint(raw_input)

        st.markdown("---")
        st.subheader("Prediction Result")

        predicted_class = result["credit_score"]
        RESULT_STYLE[predicted_class](f"Predicted Credit Score: **{predicted_class}**")

        st.markdown("**Class probabilities:**")
        for label, prob in result["probabilities"].items():
            st.write(f"{label}: {prob * 100:.1f}%")
            st.progress(prob)

        st.markdown("---")
        st.subheader("Input Summary")
        st.dataframe(raw_input, use_container_width=True)


if __name__ == "__main__":
    main()
