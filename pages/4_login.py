import streamlit as st
import pandas as pd

# st.page_link("http://www.google.com", label="Home", icon="🏠")
# st.page_link("pages/3_simple_calculator.py", label="Calculator", icon="📱")
# st.page_link("pages/page_2.py", label="Page 2", icon="2️⃣", disabled=True)
# st.page_link("http://www.google.com", label="Google", icon="🌎")

def load_users(file_path):
    return pd.read_csv(file_path)

def validate_credentials(username, password, users_df):
    user_row = users_df[users_df['username'] == username]
    if not user_row.empty:
        stored_password = user_row.iloc[0]['password']
        return stored_password == password
    return False

def main():
    st.title("Login")

    users_df = load_users("C:\datascience\streamlit\login.csv")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if validate_credentials(username, password, users_df):
            st.success("Login successful!")
            st.write("Welcome to the celestia.")
        else:
            st.error("Invalid username or password. Please try again.")

if __name__ == "__main__":
    main()

