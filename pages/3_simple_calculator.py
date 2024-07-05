import streamlit as st

st.page_link("pages/4_login.py", label="Login", icon="🔐")

st.title("Simple Calculator") 

st.write("### Input Data")
col1, col2 = st.columns(2)
num_1 = col1.number_input("Number 1", min_value=1)
num_2 = col2.number_input("Number 2", min_value=1)

# calculate

add = num_1 + num_2
sub = num_1 - num_2
mult = num_1 * num_2
division = num_1 / num_2

# st.write(f"Addition of two numbers is: {add}")
# st.write(f"Subtraction of two numbers is: {sub}")
# st.write(f"Multiplication of two numbers is: {mult}")
# st.write(f"Division of two numbers is: {division}")

col3, col4 = st.columns(2)
with col3:
    if st.button("Addition"):
        st.write(f"Addition of two numbers is: {num_1 + num_2}")
    if st.button("Subtract"):
        st.write(f"Subtraction of two numbers is: {num_1 - num_2}")

with col4:
    if st.button("Multiply"):
        st.write(f"Multiplication of two numbers is: {num_1 * num_2}")
    if st.button("Divide"):
        st.write(f"Division of two numbers is: {num_1 / num_2}")


