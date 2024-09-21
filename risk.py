import streamlit as st
import pandas as pd
# import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from xgboost import XGBClassifier
from sklearn.preprocessing import OneHotEncoder


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
    
    # Encode the target variable if necessary
    if y.dtype == 'object' or y.dtype == 'category':
        le = LabelEncoder()
        y = le.fit_transform(y)

    return X, y, scaler

X, y, scaler = preprocess_data(data) 

if X is None or y is None:
    st.stop()

# Check the distribution of the Response variable
st.write("Distribution of the Response variable:")
st.write(y.value_counts())

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Encode the target variable after splitting
if y_train.dtype == 'object' or y_train.dtype == 'category':
    le = LabelEncoder()
    y_train = le.fit_transform(y_train)
    y_test = le.transform(y_test)

# Dropdown to select model
model_name = st.selectbox("Choose a Model", ("Random Forest", "Extra Trees", "Decision Tree", "Logistic Regression", "SVM", "XGBoost", "GBM"))

# Initialize model based on selection
if model_name == "Random Forest":
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
elif model_name == "Extra Trees":
    model = ExtraTreesClassifier(n_estimators=100, random_state=42)
elif model_name == "Decision Tree":
    model = DecisionTreeClassifier(random_state=42)
elif model_name == "Logistic Regression":
    model = LogisticRegression(max_iter=1000)
elif model_name == "SVM":
    model = SVC(probability=True)
elif model_name == "XGBoost":
    model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')  # Ensure eval_metric is set
elif model_name == "GBM":
    model = GradientBoostingClassifier()

# Train the model
model.fit(X_train, y_train)

# Model accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
st.write(f"Model Accuracy ({model_name}): ", accuracy)

# Confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
st.write("Confusion Matrix: ")
st.write(conf_matrix)

# Feature importances (if applicable)
if hasattr(model, 'feature_importances_'):
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
    input_values[column] = st.number_input(f"Input a value for {column}", value=0.0, key=column)

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
    st.write("Predicted Risk Assessment: ", prediction[0])

# Display predictions alongside the original data
data['Predicted_Risk'] = model.predict(X)
st.write("Predictions on the Dataset:")
st.write(data[['Id', 'Response', 'Predicted_Risk']])
