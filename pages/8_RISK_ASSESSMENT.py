import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

@st.cache_data
def load_data():
    data = pd.read_csv('train.csv')
    return data

data = load_data()

st.write("Dataset Overview")
st.write(data.head())

def preprocess_data(data):
    if 'Response' not in data.columns:
        st.error("Target column 'Response' not found in the training dataset.")
        return None, None, None
    
    X = data.drop(columns=['Response', 'Id'], errors='ignore')
    y = data['Response']
    
    for column in X.columns:
        if X[column].dtype == 'object':
            le = LabelEncoder()
            X[column] = le.fit_transform(X[column].astype(str))
    
    X = X.apply(lambda x: x.fillna(x.mean()) if x.dtype.kind in 'biufc' else x.fillna(x.mode()[0]))
    
    scaler = MinMaxScaler()
    X[X.select_dtypes(include=['int64', 'float64']).columns] = scaler.fit_transform(X.select_dtypes(include=['int64', 'float64']))
    
    return X, y, scaler

X, y, scaler = preprocess_data(data)

if X is None or y is None:
    st.stop()

# Check the distribution of the Response variable
st.write("Distribution of the Response variable:")
st.write(y.describe())

# Train the model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Model performance metrics
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
st.write("Mean Squared Error: ", mse)
st.write("R-squared: ", r2)

# Feature importances
feature_importances = pd.Series(model.feature_importances_, index=X.columns)
st.write("Feature Importances:")
st.write(feature_importances.sort_values(ascending=False))

# Dropdown for user input
st.write("Select Information for Risk Assessment:")

# Define a single dropdown with all columns
all_columns = X.columns.tolist()
selected_columns = st.multiselect("Select Features", all_columns)

# Input fields for the selected features
input_values = {}
for column in selected_columns:
    input_values[column] = st.number_input(f"Input a value for {column}", value=0.0)

# Prepare input data for prediction
if st.button("Assess Risk"):
    input_data = pd.DataFrame([X.mean()], columns=X.columns)  # Start with default mean values
    
    for column, value in input_values.items():
        input_data[column] = value  # Replace the selected features with user inputs

    # Apply the same scaling as used during training
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    input_data[numeric_features] = scaler.transform(input_data[numeric_features])
    
    st.write("Input Data:")
    st.write(input_data)
    
    prediction = model.predict(input_data)
    st.write("Predicted Risk Assessment (Continuous Value): ", prediction[0])

# Display predictions alongside the original data
data['Predicted_Risk'] = model.predict(X)
st.write("Predictions on the Dataset:")
st.write(data[['Id', 'Response', 'Predicted_Risk']])