import streamlit as st
import pickle
import numpy as np
import pandas as pd
import plotly.express as px

# --- Configuration and File Loading ---
st.set_page_config(page_title="Laptop Price Predictor", layout="wide", initial_sidebar_state="expanded")

# --- Load Model and Data ---
@st.cache_data
def load_data():
    """Loads the pickled model and dataframe."""
    try:
        with open('artifacts/pipe.pkl', 'rb') as f:
            pipe = pickle.load(f)
        with open('artifacts/df.pkl', 'rb') as f:
            df = pickle.load(f)
        return pipe, df
    except FileNotFoundError:
        st.error("Error: Model or data files not found in the 'artifacts' directory.")
        st.stop()

pipe, df = load_data()


# --- Prediction Function ---
def predict_price(company, type_name, ram, weight, touchscreen, ips, screen_size, resolution, cpu_brand, gpu_brand, hdd, ssd, os):
    """Calculates the predicted price based on user inputs."""
    try:
        touchscreen_val = 1 if touchscreen == 'Yes' else 0
        ips_val = 1 if ips == 'Yes' else 0

        X_res = int(resolution.split('x')[0])
        Y_res = int(resolution.split('x')[1])
        ppi = ((X_res**2) + (Y_res**2))**0.5 / screen_size

        query_df = pd.DataFrame([{
            'Company': company, 'TypeName': type_name, 'Ram': ram, 'Weight': weight,
            'Touchscreen': touchscreen_val, 'Ips': ips_val, 'ppi': ppi,
            'Cpu_brand': cpu_brand, 'HDD': hdd, 'SSD': ssd, 'Gpu_brand': gpu_brand, 'os': os
        }])

        prediction_log = pipe.predict(query_df)[0]
        return int(np.exp(prediction_log))
    except (ZeroDivisionError, ValueError):
        return 0

# --- Main App Layout ---
st.title("💻 NextGen Laptop Price Predictor")
st.markdown("An interactive tool to estimate laptop prices and explore market trends.")

tab1, tab2 = st.tabs(["**Price Predictor**", "**Market Insights**"])

# --- Tab 1: The Predictor ---
with tab1:
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Configure Your Laptop")
        # --- Core Specs ---
        company = st.selectbox('Brand', sorted(df['Company'].unique()), key='company')
        type_name = st.selectbox('Type', sorted(df['TypeName'].unique()), key='type')
        ram = st.selectbox('RAM (in GB)', sorted(df['Ram'].unique()), key='ram')
        os = st.selectbox('Operating System', sorted(df['os'].unique()), key='os')
        weight = st.slider('Weight (in kg)', 0.5, 4.5, 1.5, 0.1, key='weight')
        
        # --- Hardware Specs ---
        cpu_brand = st.selectbox('CPU Brand', sorted(df['Cpu_brand'].unique()), key='cpu')
        gpu_brand = st.selectbox('GPU Brand', sorted(df['Gpu_brand'].unique()), key='gpu')
        hdd = st.select_slider('HDD (in GB)', options=[0, 128, 256, 512, 1024, 2048], key='hdd')
        ssd = st.select_slider('SSD (in GB)', options=[0, 8, 128, 256, 512, 1024], key='ssd')

        # --- Display Specs ---
        screen_size = st.slider('Screen Size (in inches)', 10.0, 20.0, 15.6, 0.1, key='screen_size')
        resolution = st.selectbox('Screen Resolution', ['1920x1080', '1366x768', '1600x900', '3840x2160', '3200x1800', '2880x1800', '2560x1600', '2560x1440', '2304x1440'], key='resolution')
        touchscreen = st.radio('Touchscreen', ['No', 'Yes'], horizontal=True, key='touchscreen')
        ips = st.radio('IPS Display', ['No', 'Yes'], horizontal=True, key='ips')

    with col2:
        st.subheader("Price Estimation")
        
        predicted_price = predict_price(
            st.session_state.company, st.session_state.type, st.session_state.ram,
            st.session_state.weight, st.session_state.touchscreen, st.session_state.ips,
            st.session_state.screen_size, st.session_state.resolution, st.session_state.cpu,
            st.session_state.gpu, st.session_state.hdd, st.session_state.ssd, st.session_state.os
        )

        if predicted_price > 0:
            # Model's typical error margin (from Mean Absolute Error in notebook)
            mae_log_scale = 0.15 
            price_lower_bound = predicted_price * np.exp(-mae_log_scale)
            price_upper_bound = predicted_price * np.exp(mae_log_scale)
            
            # Create columns for the metrics
            metric_col1, metric_col2, metric_col3 = st.columns(3)
            
            metric_col1.metric("Plausible Low", f"₹ {int(price_lower_bound):,}")
            metric_col2.metric("Predicted Price", f"₹ {predicted_price:,}")
            metric_col3.metric("Plausible High", f"₹ {int(price_upper_bound):,}")
            
            st.caption("The prediction is flanked by a plausible price range based on the model's average error margin.")
        else:
            st.warning("Could not calculate a valid price. Please check the inputs.")

# --- Tab 2: Market Insights ---
with tab2:
    st.header("Explore Laptop Market Trends")
    st.markdown("Select a feature to see how it correlates with the average market price.")
    
    feature_options = ['Company', 'TypeName', 'Ram', 'Cpu_brand', 'Gpu_brand', 'os']
    selected_feature = st.selectbox("Choose a feature to analyze", feature_options)
    
    avg_price_df = df.groupby(selected_feature)['Price'].mean().round(0).sort_values(ascending=False).reset_index()
    
    fig = px.bar(
        avg_price_df,
        x=selected_feature,
        y='Price',
        title=f"Average Laptop Price by {selected_feature}",
        labels={'Price': 'Average Price (₹)', selected_feature: selected_feature},
        text_auto=True
    )
    fig.update_layout(title_x=0.5, xaxis_title=None)
    st.plotly_chart(fig, use_container_width=True)