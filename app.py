
import streamlit as st
import pandas as pd
import pickle

# Load trained model
with open("house_price_model.pkl", "rb") as file:
    model = pickle.load(file)

# Website settings
st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠"
)

st.title("🏠 House Price Prediction")
st.write("Predict house prices using Machine Learning")

st.divider()

st.header("Enter House Details")

# User inputs
area = st.number_input(
    "Area (Square Feet)",
    min_value=100,
    max_value=10000,
    value=1200
)

bedrooms = st.number_input(
    "Number of Bedrooms",
    min_value=1,
    max_value=10,
    value=3
)

bathrooms = st.number_input(
    "Number of Bathrooms",
    min_value=1,
    max_value=10,
    value=2
)

age = st.number_input(
    "House Age (Years)",
    min_value=0,
    max_value=100,
    value=5
)

# Prediction
if st.button("Predict House Price"):

    input_data = pd.DataFrame(
        [[area, bedrooms, bathrooms, age]],
        columns=["Area", "Bedrooms", "Bathrooms", "Age"]
    )

    prediction = model.predict(input_data)

    st.success("Prediction Completed!")

    st.metric(
        "Estimated House Price",
        f"₹{prediction[0]:.2f} Lakhs"
    )

    st.info(
        "This is an educational estimate "
        "based on a small synthetic dataset."
    )

st.divider()

st.caption("Developed by Parth Surani")
