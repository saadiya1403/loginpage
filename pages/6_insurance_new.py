import pandas as pd
import streamlit as st

data = pd.read_csv('C:\datascience\streamlit\insurance_dataset.csv')

st.title('Insurance Aggregator')

st.subheader('Insurance Dataset')
st.write(data)

st.sidebar.header('User Input Features')

def user_input_features():
    age = st.sidebar.number_input('Age', min_value=18, max_value=100, value=30)
    gender = st.sidebar.selectbox('Gender', options=data['Gender'].unique())
    income = st.sidebar.number_input('Income', min_value=0, value=50000)
    marital_status = st.sidebar.selectbox('Marital Status', options=data['Marital_Status'].unique())
    education = st.sidebar.selectbox('Education', options=data['Education'].unique())
    occupation = st.sidebar.selectbox('Occupation', options=data['Occupation'].unique())
    
    return pd.DataFrame({
        'Age': [age],
        'Gender': [gender],
        'Income': [income],
        'Marital_Status': [marital_status],
        'Education': [education],
        'Occupation': [occupation]
    })

input_data = user_input_features()


st.subheader('User Input:')
st.write(input_data)

st.subheader('Claim Amount Statistics')
st.write(data['Claim_Amount'].describe())

st.subheader('Claim Amount Distribution')
st.bar_chart(data['Claim_Amount'])

