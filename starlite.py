# Streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# App Title
st.set_page_config(page_title="My Application", layout="wide")
st.title("My All-in-One ML Dashboard")

# Sidebar Inputs
st.sidebar.header("Controls")
num_rows = st.sidebar.slider("Rows of Random Data", 10, 100, 25)
columns = ["AmVals", "Beta", "CRUD", "Delta"]
feature_col = st.sidebar.selectbox("X Axis Feature", columns)
target_col = st.sidebar.selectbox("Y Axis Target", columns)
show_chart = st.sidebar.checkbox("Show Chart", value=True)

# Generate Random Data
df = pd.DataFrame(
    np.random.randn(num_rows, 4),
    columns=columns
)

st.subheader("Data Preview")
st.dataframe(df)

# Line Chart
if show_chart:
    st.subheader("Line Chart")
    st.line_chart(df)

# Matplotlib Chart
st.subheader("Scatter Plot with Matplotlib")
fig, ax = plt.subplots()
ax.scatter(df[feature_col], df[target_col])
ax.set_xlabel(feature_col)
ax.set_ylabel(target_col)
st.pyplot(fig)

# Train a Simple ML Model
st.subheader("Linear Regression Model")
model = LinearRegression()
model.fit(df[[feature_col]], df[target_col])
user_input = st.number_input(f"Enter value for {feature_col}", value=0.0)
pred = model.predict([[user_input]])
st.success(f"Prediction: {pred[0]:.2f}")

# File Upload
st.subheader("Upload CSV File")
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
if uploaded_file:
    uploaded_df = pd.read_csv(uploaded_file)
    st.write("Uploaded Data", uploaded_df)

# Session State Counter
st.subheader("Counter with Session State")
if 'counter' not in st.session_state:
    st.session_state.counter = 0

if st.button("Increase Counter"):
    st.session_state.counter += 1
st.write("Counter Value:", st.session_state.counter)

# Form Example
st.subheader("User Input Form")
with st.form("user_form"):
    name = st.text_input("Enter your name")
    email = st.text_input("Email")
    age = st.slider("Age", 15, 90, 25)
    submitted = st.form_submit_button("Submit")

if submitted:
    st.success(f"Welcome {name}, Age: {age}, Email: {email}")

# Downloadable File
st.subheader("Download Data as CSV")

def convert_df(df):
    return df.to_csv(index=False).encode('utf-8')

csv_data = convert_df(df)
st.download_button(
    label="Download Random Data",
    data=csv_data,
    file_name='random_data.csv',
    mime='text/csv'
)

