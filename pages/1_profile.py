import streamlit as st

st.write("hello!")
st.write("Welcome to Celestia")

st.write("Select any of the following!")
st.page_link("pages/3_simple_calculator.py", label="Calculator", icon="📱")
st.page_link("pages/4_login.py", label="Login", icon="🔐")


x=st.text_input("Favourite Color?")
st.write(f"Your Favourite Color is: {x}")

is_clicked =st.button("Click Me")