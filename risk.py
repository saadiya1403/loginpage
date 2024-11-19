import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

@st.cache_data
def load_data():
    data = pd.read_csv('train.csv')
    return data

data = load_data()

st.write("### Dataset Overview")
st.write(data.head())

def preprocess_data(data):
    # Checking if the target 'Response' exists
    if 'Response' not in data.columns:
        st.error("Target column 'Response' not found.")
        return None, None, None
    
    # Features and target
    X = data.drop(columns=['Response', 'Id'], errors='ignore')
    y = data['Response']
    
    # Encoding categorical variables
    for column in X.columns:
        if X[column].dtype == 'object':
            le = LabelEncoder()
            X[column] = le.fit_transform(X[column].astype(str))
    
    # Handling missing values
    X = X.apply(lambda x: x.fillna(x.mean()) if x.dtype.kind in 'biufc' else x.fillna(x.mode()[0]))
    
    # Scaling numerical data
    scaler = MinMaxScaler()
    X[X.select_dtypes(include=['int64', 'float64']).columns] = scaler.fit_transform(X.select_dtypes(include=['int64', 'float64']))
    
    # Label encoding for target
    if y.dtype == 'object' or y.dtype == 'category':
        le = LabelEncoder()
        y = le.fit_transform(y)

    return X, y, scaler

X, y, scaler = preprocess_data(data)

if X is None or y is None:
    st.stop()

# Apply SMOTE for balancing classes
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.4, random_state=42)

# Dropdown to select model
model_name = st.selectbox("Choose a Model", ("Random Forest", "Extra Trees", "Decision Tree", "Logistic Regression", "SVM", "GBM"))

# Initialize model with class weights if applicable
if model_name == "Random Forest":
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
elif model_name == "Extra Trees":
    model = ExtraTreesClassifier(n_estimators=100, random_state=42, class_weight='balanced')
elif model_name == "Decision Tree":
    model = DecisionTreeClassifier(random_state=42, class_weight='balanced')
elif model_name == "Logistic Regression":
    model = LogisticRegression(max_iter=1000, class_weight='balanced')
elif model_name == "SVM":
    model = SVC(probability=True, class_weight='balanced')
elif model_name == "GBM":
    model = GradientBoostingClassifier(learning_rate=0.1, max_depth=4, n_estimators=300, min_samples_leaf=5)

# Train the model
model.fit(X_train, y_train)

# Model predictions and evaluation
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

st.write(f"### Model Evaluation for {model_name}")
st.write(f"Accuracy: {accuracy:}")
st.write(f"Precision: {precision:}")
st.write(f"Recall: {recall:}")
st.write(f"F1 Score: {f1:}")

# Confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
st.write("### Confusion Matrix:")
st.write(conf_matrix)

# Feature importances (if applicable)
if hasattr(model, 'feature_importances_'):
    feature_importances = pd.Series(model.feature_importances_, index=X.columns)
    st.write("### Feature Importances:")
    st.bar_chart(feature_importances.sort_values(ascending=False))

# User input for prediction
st.write("### Select Features for Risk Assessment")
all_columns = X.columns.tolist()
selected_columns = st.multiselect("Select Features", all_columns)

input_values = {}
for column in selected_columns:
    input_values[column] = st.number_input(f"Input value for {column}", value=0.0)

if st.button("Assess Risk"):
    input_data = pd.DataFrame([X.mean()], columns=X.columns)
    for column, value in input_values.items():
        input_data[column] = value
    
    input_data[X.select_dtypes(include=['int64', 'float64']).columns] = scaler.transform(input_data[X.select_dtypes(include=['int64', 'float64']).columns])
    prediction = model.predict(input_data)
    st.write("### Predicted Risk Assessment: ", prediction[0])

data['Predicted_Risk'] = model.predict(X)
st.write("### Predictions on the Dataset:")
st.write(data[['Response', 'Predicted_Risk']])

