import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load the trained model pipeline and the processed dataframe
try:
    pipe = pickle.load(open('artifacts/pipe.pkl', 'rb'))
    df = pickle.load(open('artifacts/df.pkl', 'rb'))
except FileNotFoundError:
    st.error("Model or data files not found. Make sure 'pipe.pkl' and 'df.pkl' are in the 'artifacts' directory.")
    st.stop()


# --- App Title and Description ---
st.set_page_config(page_title="Laptop Price Predictor", layout="wide")
st.title("💻 Laptop Price Predictor")
st.markdown("Fill in the details below to get an estimated price for your desired laptop configuration.")
st.markdown("---")


# --- User Input Fields in Columns for a Modern Layout ---

# -- Row 1: Brand, Type, OS --
col1, col2, col3 = st.columns(3)

with col1:
    company = st.selectbox('**Brand**', df['Company'].unique())

with col2:
    type = st.selectbox('**Type**', df['TypeName'].unique())

with col3:
    os = st.selectbox('**Operating System**', df['os'].unique())


# -- Row 2: RAM, Weight, Touchscreen, IPS --
col4, col5 = st.columns(2)

with col4:
    ram = st.selectbox('**RAM (in GB)**', [2, 4, 6, 8, 12, 16, 24, 32, 64])
    weight = st.number_input('**Weight (in kg)**', min_value=0.5, max_value=4.5, value=1.5, step=0.1)

with col5:
    touchscreen = st.radio('**Touchscreen**', ['No', 'Yes'], horizontal=True)
    ips = st.radio('**IPS Display**', ['No', 'Yes'], horizontal=True)


# -- Row 3: Screen Size, Resolution --
st.markdown("---")
st.subheader("Display Configuration")
col6, col7 = st.columns(2)

with col6:
    screen_size = st.number_input('**Screen Size (in inches)**', min_value=10.0, max_value=20.0, value=15.6, step=0.1)

with col7:
    resolution = st.selectbox('**Screen Resolution**', [
        '1920x1080', '1366x768', '1600x900', '3840x2160', '3200x1800',
        '2880x1800', '2560x1600', '2560x1440', '2304x1440'
    ])


# -- Row 4: CPU, GPU --
st.markdown("---")
st.subheader("Hardware Configuration")
col8, col9 = st.columns(2)

with col8:
    cpu = st.selectbox('**CPU Brand**', df['Cpu_brand'].unique())

with col9:
    gpu = st.selectbox('**GPU Brand**', df['Gpu_brand'].unique())

# -- Row 5: HDD, SSD --
col10, col11 = st.columns(2)
with col10:
    hdd = st.selectbox('**HDD (in GB)**', [0, 128, 256, 512, 1024, 2048])

with col11:
    ssd = st.selectbox('**SSD (in GB)**', [0, 8, 128, 256, 512, 1024])


# --- Prediction Logic and Display ---
st.markdown("---")
if st.button('**Predict Price**', type="primary", use_container_width=True):
    # Convert categorical inputs to numerical format
    touchscreen = 1 if touchscreen == 'Yes' else 0
    ips = 1 if ips == 'Yes' else 0

    # Calculate PPI (Pixels Per Inch)
    try:
        X_res = int(resolution.split('x')[0])
        Y_res = int(resolution.split('x')[1])
        ppi = ((X_res**2) + (Y_res**2))**0.5 / screen_size
    except ZeroDivisionError:
        st.error("Screen size cannot be zero. Please enter a valid value.")
        st.stop()

    # Create the query array for the model
    # Important: The order must match the training data columns
    query = np.array([
        company, type, ram, weight, touchscreen, ips, ppi, cpu, hdd, ssd, gpu, os
    ], dtype=object)

    query = query.reshape(1, 12)

    # Make the prediction
    prediction_log = pipe.predict(query)[0]
    predicted_price = int(np.exp(prediction_log))

    # Display the result in a styled box
    st.success(f"### Estimated Laptop Price: ₹ {predicted_price:,}")
    st.balloons()