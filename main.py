import streamlit as st
import pandas as pd
import numpy as np

st.write("hello world")
x=st.text_input("Favourite Movie?")
st.write(f"Your Favourite Movie is: {x}")

is_clicked =st.button("Click Me")
st.write("##this is a H2 title!")

data = pd.read_csv("C:\datascience\streamlit\login.csv")
st.write(data)

chart_data = pd.DataFrame(
    np.random.randn(20,3),
    columns=["a","b","c"]
)

st.bar_chart(chart_data)
st.line_chart(chart_data)