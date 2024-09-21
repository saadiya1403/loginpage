import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import streamlit as st

# Load the dataset
train_data = pd.read_csv('train.csv')  # Assuming you have a train.csv file
test_data = pd.read_csv('test.csv')

# Preprocess the data
X = train_data.drop(['Response'], axis=1)  # Replace 'target' with your target column name
y = train_data['Response']

# Identify categorical columns
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()

# Convert categorical columns to numerical using Label Encoding
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest Classifier
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Streamlit app
st.title('Risk Assessment Score Predictor')

# Input features
features = {}
for column in X.columns:
    features[column] = st.number_input(f'Enter value for {column}', value=0.0)

# Predicting the risk score
if st.button('Predict Risk Score'):
    input_data = pd.DataFrame(features, index=[0])
    
    # Convert categorical inputs using the same label encoders
    for col in categorical_cols:
        if col in input_data.columns:
            input_data[col] = label_encoders[col].transform(input_data[col].astype(str))

    prediction = model.predict(input_data)
    st.write(f'Predicted Risk Score: {prediction[0]}')