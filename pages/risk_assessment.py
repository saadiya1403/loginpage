import optuna
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from imblearn.over_sampling import SMOTE
from sklearn.utils.class_weight import compute_class_weight

@st.cache_data
def load_data():
    data = pd.read_csv('train.csv')
    return data

data = load_data()

st.write("### Dataset Overview")
st.write(data.head())

def preprocess_data(data):
    if 'Response' not in data.columns:
        st.error("Target column 'Response' not found.")
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
    
    if y.dtype == 'object' or y.dtype == 'category':
        le = LabelEncoder()
        y = le.fit_transform(y)
    
    y = y - y.min()
    
    return X, y, scaler

X, y, scaler = preprocess_data(data)

if X is None or y is None:
    st.stop()

st.write("### Distribution of the Response variable:")
st.bar_chart(pd.Series(y).value_counts())

# Apply SMOTE to balance the training data
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Calculate class weights
class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}

X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.4, random_state=42)

# Optuna optimization function for XGBoost with SMOTE and class weights
def optimize_xgb(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 250, step=50),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 10.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 10.0),
        'scale_pos_weight': class_weight_dict.get(1, 1),
        'random_state': 42
    }
    xgb = XGBClassifier(**params, use_label_encoder=False, eval_metric='logloss')
    xgb.fit(X_train, y_train)
    y_pred = xgb.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return 1.0 - accuracy

# Run Optuna study for XGB tuning
study = optuna.create_study(direction='minimize')
study.optimize(optimize_xgb, n_trials=50)
best_params = study.best_params

# Display the best parameters and accuracy
st.write("### Best Parameters from Optuna for XGB")
st.json(best_params)

# Train XGB with optimized parameters
model = XGBClassifier(**best_params, use_label_encoder=False, eval_metric='logloss')
model.fit(X_train, y_train)

# Model accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
st.write(f"### Optimized Model Accuracy (XGB with SMOTE and Class Weights): {accuracy}")

# Confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
st.write("### Confusion Matrix:")
st.write(conf_matrix)

# Feature importances
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
