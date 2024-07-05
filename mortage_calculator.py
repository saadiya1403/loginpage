import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import math

st.title("Mortage Repayments Calculator") 

st.write("### Input Data")
col1, col2 = st.columns(2)
home_value = col1.number_input("Home Value", min_value=0, value=500000,step=10000 )
Downpayment = col1.number_input("Downpayment", min_value=0, value=100000, step=10000)
interest_rate = col2.number_input("Interest rate (in %)", min_value=0.0, value=5.5)
loan_term = col2.number_input("Loan Term (in years)", min_value=1, value=30)

# calculate the repayments
loan_amount = home_value - Downpayment
monthly_interest_rate = (interest_rate / 100) / 12
number_of_payments = loan_term * 12
monthly_payment = (
    loan_amount
    * (monthly_interest_rate * (1 + monthly_interest_rate) ** number_of_payments)
    / ((1 + monthly_interest_rate) ** number_of_payments - 1)
)

# to display the repayments
total_payments = monthly_payment * number_of_payments
total_interest = total_payments - loan_amount

st.write("### Repayments")
col1, col2, col3 = st.columns(3)
col1.metric(label="Monthly Repayments", value=f"${monthly_payment:,.2f}")
col2.metric(label="Total Repayments", value=f"${total_payments:,.0f}")
col3.metric(label="Total Interest", value=f"${total_interest:,.0f}")

# using metric to pop out the elements

# now to create a data frame with payment schedule
schedule = []
remaining_balance = loan_amount

for i in range(1, number_of_payments + 1):
    interest_payment = remaining_balance * monthly_interest_rate
    principal_payment = monthly_payment- interest_payment
    remaining_balance-= principal_payment
    year = math.ceil(i / 12) # to calculate the year into loan
    # ceil() function takes in a value and returns its ceiling, the smallest integer greater than or equal to that value.
    schedule.append(
        [
            i,
            monthly_payment,
            principal_payment,
            interest_payment,
            remaining_balance,
            year,
        ]
    )

df = pd.DataFrame(
    schedule, 
    columns =["month", "payment", "principal", "Interest", "Remaining Balance", "Year"],
)

st.write("### Payment Schedule")
payment_df = df[["Year", "Remaining Balance"]].groupby("Year").min()
st.line_chart(payment_df)



