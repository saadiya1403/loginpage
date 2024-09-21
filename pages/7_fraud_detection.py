import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier #i used this for classification with 80% training and 20% testing
from sklearn.preprocessing import LabelEncoder #it basically transforms categorical data (like text labels) into numerical data

@st.cache_data
def load_data():
    df = pd.read_csv('insurance_dataset.csv')
    return df

df = load_data()

le = LabelEncoder() 
df['Gender'] = le.fit_transform(df['Gender']) #fit learns the unique categories in the 'Gender' column and transform converts these categories into numerical values.
df['Marital_Status'] = le.fit_transform(df['Marital_Status'])
df['Education'] = le.fit_transform(df['Education'])
df['Occupation'] = le.fit_transform(df['Occupation'])

#i will create a simple fraud flag which i use for the demonstration purposes
df['Fraud_Flag'] = np.where(df['Claim_Amount'] > df['Claim_Amount'].quantile(0.95), 1, 0) #so, i will create a new column called 'Fraud_Flag' in the data frame
#this calculates the 95th percentile or the 95% quantile of the 'Claim_Amount' column
#the 95th percentile is the value below which 95% of the claim amounts fall
#this means that the only top 5% of claim amounts are above this value

#now i will prepare features and target
X = df[['Age', 'Gender', 'Income', 'Marital_Status', 'Education', 'Occupation', 'Claim_Amount']]
y = df['Fraud_Flag']

#then i will train a Random Forest model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) #it basically means that the 20% data is for training
model = RandomForestClassifier(n_estimators=100, random_state=42) 
model.fit(X_train, y_train)

st.title('Insurance Claim Fraud Detection')

st.sidebar.header('Input Parameters')

age = st.sidebar.slider('Age', 18, 100, 30)
gender = st.sidebar.selectbox('Gender', ('Male', 'Female'))
income = st.sidebar.number_input('Income', min_value=0, max_value=500000, value=50000)
marital_status = st.sidebar.selectbox('Marital Status', ('Single', 'Married'))
education = st.sidebar.selectbox('Education', ('Bachelor\'s', 'Master\'s', 'PhD'))
occupation = st.sidebar.selectbox('Occupation', ('CEO', 'Doctor', 'Engineer', 'Teacher', 'Waiter'))
claim_amount = st.sidebar.number_input('Claim Amount', min_value=0, max_value=100000, value=5000)

#Now we will convert the selectedd data into a number
gender_encoded = le.fit_transform([gender])[0]
marital_status_encoded = le.fit_transform([marital_status])[0]
education_encoded = le.fit_transform([education])[0]
occupation_encoded = le.fit_transform([occupation])[0]

input_data = np.array([[age, gender_encoded, income, marital_status_encoded, 
                        education_encoded, occupation_encoded, claim_amount]])
prediction = model.predict(input_data)
prediction_proba = model.predict_proba(input_data)

st.subheader('Claim Fraud Prediction')
if prediction[0] == 1:
    st.error('⚠️ Potential Fraudulent Claim Detected')
else:
    st.success('✅ Claim Appears Legitimate')

st.write(f'Probability of Fraud: {prediction_proba[0][1]:.2%}')

st.subheader('Feature Importance')
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)

st.bar_chart(feature_importance.set_index('Feature'))

st.subheader('Dataset Overview')
st.write(df.head())

st.subheader('Dataset Statistics')
st.write(df.describe())