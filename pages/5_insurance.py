import streamlit as st
import pandas as pd

# Create a Streamlit app
st.title("Insurance Aggregator")
st.write("Enter your details to get recommended insurance policies")

# Get client details
name = st.text_input("Enter your name")
age = st.slider("Enter your age", min_value=0, max_value=100, value=25, step=1)
insurance_type = st.selectbox("Which insurance?", ["Life Insurance", "Car/Bike Insurance", "Home Insurance"])
sum_assured = st.select_slider("Amount of insurance (Sum assured)", ["50L", "1 cr", "2 cr"], value="1 cr")

# Load the insurance policies from the CSV file
insurance_policies = pd.read_csv('insurance_policies.csv')

# Create a submit button
if st.button("Submit"):
    # Filter the insurance policies based on the user's input
    filtered_policies = insurance_policies[(insurance_policies['Insurance Type'] == insurance_type) &
                                          (insurance_policies['Sum Assured'] == sum_assured)]
    
    # Display the recommended insurance policies
    st.write("Recommended Insurance Policies:")
    st.write("-------------------------------")
    for index, row in filtered_policies.iterrows():
        st.write(f"**{row['Company']}**: **{row['Monthly Premium']}** per month for **{row['Duration']}** with a sum assured of **{row['Sum Assured']}**")
    st.write("-------------------------------")