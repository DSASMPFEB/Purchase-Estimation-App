import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import gdown


st.set_page_config(page_title="Purchase Prediction App", layout="centered")
st.title("🛍️ Purchase Prediction App")
# Google Drive file ID
file_id = "1BtS1KyGA5dpgbQH2qGKoMoDp3xn4gZ33"  # Replace with actual ID
url = f"https://drive.google.com/uc?id={file_id}"

# Download model if not already present
model_path = "rf_model.pkl"
if not os.path.exists(model_path):
    gdown.download(url, model_path, quiet=False)

# Load model
with open(model_path, "rb") as f:
    rf_model = pickle.load(f)


# Sidebar inputs
st.sidebar.header("Customer & Product Info")

gender = st.sidebar.selectbox("Gender", ["M", "F"])
age = st.sidebar.selectbox(
    "Age Group", ["0-17", "18-25", "26-35", "36-45", "46-50", "51-55", "55+"]
)
city = st.sidebar.selectbox("City Category", ["A", "B", "C"])
stay_years = st.sidebar.selectbox("Years in Current City", ["0", "1", "2", "3", "4+"])
marital_status = st.sidebar.selectbox("Marital Status", ["Unmarried", "Married"])
occupation = st.sidebar.number_input(
    "Occupation Code", min_value=0, max_value=20, value=5
)
product_cat_1 = st.sidebar.number_input(
    "Product Category 1", min_value=1, max_value=20, value=5
)
product_cat_2 = st.sidebar.number_input(
    "Product Category 2", min_value=0, max_value=20, value=0
)
product_cat_3 = st.sidebar.number_input(
    "Product Category 3", min_value=0, max_value=20, value=0
)

# Encode categorical inputs (must match training encoding)
gender_map = {"M": 1, "F": 0}
age_map = {
    "0-17": 0,
    "18-25": 1,
    "26-35": 2,
    "36-45": 3,
    "46-50": 4,
    "51-55": 5,
    "55+": 6,
}
city_map = {"A": 0, "B": 1, "C": 2}
stay_map = {"0": 0, "1": 1, "2": 2, "3": 3, "4+": 4}
marital_map = {"Unmarried": 0, "Married": 1}

# Prepare feature array
features = np.array(
    [
        [
            gender_map[gender],
            age_map[age],
            city_map[city],
            stay_map[stay_years],
            marital_map[marital_status],
            occupation,
            product_cat_1,
            product_cat_2,
            product_cat_3,
        ]
    ]
)

# Prediction
if st.button("Predict Purchase Amount"):
    pred = rf_model.predict(features)
    st.markdown(f"### 💰 Estimated Purchase Amount: ₹{round(pred[0], 2)}")

# Footer
st.markdown("---")
st.caption("Built by Anna • Powered by Streamlit & Random Forest")
