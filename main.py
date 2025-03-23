import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestRegressor

# Load dataset (for similar results section)
def load_data():
    data = {
        'Gender': ['male', 'male', 'female', 'male', 'female'],
        'Age': [52, 21, 24, 30, 75],
        'Height': [183, 180, 145, 179, 164],
        'Weight': [87, 85, 49, 82, 64],
        'Duration': [11, 14, 12, 13, 9],
        'Heart_Rate': [90, 86, 88, 89, 89],
        'Body_Temp': [39.7, 40.3, 40.2, 40, 39.7],
        'Calories': [53, 37, 51, 43, 46]
    }
    return pd.DataFrame(data)

# Dummy model for calorie prediction
def train_model():
    X = np.random.rand(100, 5) * [90, 25, 35, 70, 6]  # Fake training data
    y = X[:, 0] * 0.5 + X[:, 1] * 0.7 + X[:, 2] * 1.5 + X[:, 3] * 0.3 + X[:, 4] * 1.2  # Fake target
    model = RandomForestRegressor()
    model.fit(X, y)
    return model

model = train_model()

# Streamlit UI
st.title("Personal Fitness Tracker")
st.write("In this WebApp, you can observe your predicted calories burned based on your inputs.")

# User Input
st.sidebar.header("User Input Parameters:")
age = st.sidebar.slider("Age:", 10, 100, 30)
bmi = st.sidebar.slider("BMI:", 15, 40, 20)
duration = st.sidebar.slider("Duration (min):", 0, 35, 15)
heart_rate = st.sidebar.slider("Heart Rate:", 60, 130, 80)
body_temp = st.sidebar.slider("Body Temperature (C):", 36, 42, 38)
gender = st.sidebar.radio("Gender:", ("Male", "Female"))
gender_value = 1 if gender == "Male" else 0

# Convert input into DataFrame
input_data = pd.DataFrame([[age, bmi, duration, heart_rate, body_temp, gender_value]],
                          columns=["Age", "BMI", "Duration", "Heart_Rate", "Body_Temp", "Gender_Male"])
st.subheader("Your Parameters:")
st.table(input_data)

# Make Prediction
prediction = model.predict(input_data.iloc[:, :-1])[0]
st.subheader("Prediction:")
st.write(f"### {round(prediction, 2)} kilocalories")

# Display Similar Results
data = load_data()
st.subheader("Similar Results:")
st.dataframe(data)

# General Information
general_info = [
    f"You are older than {np.random.uniform(20, 80):.1f}% of other people.",
    f"Your exercise duration is higher than {np.random.uniform(30, 70):.1f}% of other people.",
    f"You have a higher heart rate than {np.random.uniform(10, 60):.1f}% of other people during exercise.",
    f"You have a higher body temperature than {np.random.uniform(5, 50):.1f}% of other people during exercise."
]

st.subheader("General Information:")
for info in general_info:
    st.write(f"- {info}")
