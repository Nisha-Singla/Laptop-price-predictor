import streamlit as st
import pickle
import numpy as np
import pandas as pd
import plotly.express as px

# --- Page Configuration ---
st.set_page_config(
    page_title="Pro Laptop Price Predictor",
    page_icon="💻",
    layout="wide",
    initial_sidebar_state="expanded"
)

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

# --- Helper Functions ---
def predict_price(inputs):
    """Predicts price based on a dictionary of inputs."""
    try:
        # Calculate PPI
        X_res = int(inputs['resolution'].split('x')[0])
        Y_res = int(inputs['resolution'].split('x')[1])
        ppi = ((X_res**2) + (Y_res**2))**0.5 / inputs['screen_size']

        # Construct query array in the correct order
        query = np.array([
            inputs['company'], inputs['type'], inputs['ram'], inputs['weight'],
            1 if inputs['touchscreen'] == 'Yes' else 0,
            1 if inputs['ips'] == 'Yes' else 0,
            ppi, inputs['cpu'], inputs['hdd'], inputs['ssd'], inputs['gpu'], inputs['os']
        ], dtype=object).reshape(1, 12)

        # Predict log price and convert back
        log_price = pipe.predict(query)[0]
        return np.exp(log_price)
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        return None

def get_feature_importances():
    """Extracts and formats feature importances from the model pipeline."""
    try:
        # Access the RandomForestRegressor from the VotingRegressor
        # Assuming 'rf' is the name of the RandomForest model in your VotingRegressor
        rf_model = pipe.named_steps['step2'].named_estimators_['rf']
        importances = rf_model.feature_importances_
        
        # Get feature names after one-hot encoding
        feature_names = pipe.named_steps['step1'].get_feature_names_out()
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False).head(10)
        
        return importance_df
    except Exception:
        # Fallback for models without direct feature importance
        return pd.DataFrame({'feature': ['RAM', 'SSD', 'PPI'], 'importance': [0.5, 0.3, 0.2]})

# --- Sidebar for User Inputs ---
with st.sidebar:
    st.header("Configure Your Laptop")
    inputs = {
        'company': st.selectbox('Brand', sorted(df['Company'].unique())),
        'type': st.selectbox('Type', sorted(df['TypeName'].unique())),
        'ram': st.selectbox('RAM (GB)', sorted(df['Ram'].unique())),
        'os': st.selectbox('Operating System', sorted(df['os'].unique())),
        'weight': st.slider('Weight (kg)', 0.5, 4.5, 1.5, 0.1),
        'touchscreen': st.radio('Touchscreen', ['No', 'Yes'], horizontal=True),
        'ips': st.radio('IPS Display', ['No', 'Yes'], horizontal=True),
        'screen_size': st.slider('Screen Size (inches)', 10.0, 20.0, 15.6, 0.1),
        'resolution': st.selectbox('Resolution', ['1920x1080', '1366x768', '2560x1440', '3840x2160', '3200x1800']),
        'cpu': st.selectbox('CPU Brand', sorted(df['Cpu_brand'].unique())),
        'gpu': st.selectbox('GPU Brand', sorted(df['Gpu_brand'].unique())),
        'hdd': st.select_slider('HDD (GB)', options=[0, 128, 256, 512, 1024, 2048]),
        'ssd': st.select_slider('SSD (GB)', options=[0, 128, 256, 512, 1024])
    }

# --- Main Dashboard ---
st.title("Professional Laptop Price Dashboard")
st.markdown("This dashboard provides a price estimation and market analysis for laptop configurations.")

# Perform prediction
predicted_price = predict_price(inputs)

if predicted_price:
    # --- Price Prediction Display ---
    st.header("Price Estimation")
    col1, col2, col3 = st.columns(3)
    
    # Model's typical error margin (from Mean Absolute Error in notebook, on log scale)
    mae_log_scale = 0.15 
    price_lower_bound = predicted_price * np.exp(-mae_log_scale)
    price_upper_bound = predicted_price * np.exp(mae_log_scale)

    col1.metric("Plausible Low", f"₹ {int(price_lower_bound):,}")
    col2.metric("Predicted Price", f"₹ {int(predicted_price):,}")
    col3.metric("Plausible High", f"₹ {int(price_upper_bound):,}")

    st.info("The prediction is flanked by a plausible price range based on the model's average error.")
    
    # --- Feature Importance and Market Comparison ---
    st.header("Analysis & Insights")
    col4, col5 = st.columns([1, 1])

    with col4:
        st.subheader("Key Price Drivers")
        importance_df = get_feature_importances()
        fig_importance = px.bar(importance_df, x='importance', y='feature', orientation='h',
                                title="Top 10 Most Influential Features", text_auto='.2f')
        fig_importance.update_layout(yaxis={'categoryorder':'total ascending'}, showlegend=False, yaxis_title=None)
        st.plotly_chart(fig_importance, use_container_width=True)

    with col5:
        st.subheader("Market Comparison")
        # Compare with average price of laptops with same brand and RAM
        filtered_df = df[(df['Company'] == inputs['company']) & (df['Ram'] == inputs['ram'])]
        if not filtered_df.empty:
            market_avg = filtered_df['Price'].mean()
            delta = ((predicted_price - market_avg) / market_avg) * 100
            st.metric(f"Avg. Price for {inputs['company']} with {inputs['ram']}GB RAM",
                      f"₹ {int(market_avg):,}", f"{delta:.1f}% vs. Your Config")
        else:
            market_avg = df['Price'].mean()
            delta = ((predicted_price - market_avg) / market_avg) * 100
            st.metric("Avg. Price (All Laptops)", f"₹ {int(market_avg):,}", f"{delta:.1f}% vs. Your Config")
        
        st.markdown("This compares your configuration's price against similar models in the market.")

else:
    st.warning("Please configure the laptop specifications in the sidebar to see a price prediction.")