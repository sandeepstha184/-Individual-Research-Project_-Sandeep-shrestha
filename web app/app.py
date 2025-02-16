import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow.keras as keras
import matplotlib.pyplot as plt
import plotly.express as px


# 🌍 Title of the Web App
st.title("🚦🌍 AQI and Traffic Prediction Web App")

st.write("Welcome! Upload a CSV file to predict AQI levels using LSTM and Random Forest models.")

# 📂 Upload CSV File
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    st.success("✅ File uploaded successfully!")

    # 📊 Load Data
    df = pd.read_csv(uploaded_file)

    # 📋 Show Data Preview
    st.subheader("📊 Data Preview:")
    st.write(df.head())

    try:
        # 🔍 Load trained models
        rf_model = joblib.load("rf_tuned_model.pkl")  # Load RF Model
        lstm_model = keras.models.load_model("lstm_tuned_model.keras")  # Load LSTM Model
        
        # 🔍 Get expected feature names from the trained model
        expected_features = rf_model.feature_names_in_

        # ✅ **Feature Transformation: Convert 'indicator' column into separate columns**
        if "indicator" in df.columns and "value" in df.columns:
            df_pivot = df.pivot(index=["year", "month", "date", "time", "location", "latitude", "longitude"], 
                                columns="indicator", values="value").reset_index()

            # 🔹 Fill missing feature columns with 0
            for feature in expected_features:
                if feature not in df_pivot.columns:
                    df_pivot[feature] = 0  

            # ✅ Select only the expected features
            X = df_pivot[expected_features]

            # 🔮 Make Predictions
            rf_prediction = rf_model.predict(X)
            lstm_prediction = lstm_model.predict(X)

            # ✅ Store Predictions
            df_pivot["RF Prediction"] = rf_prediction
            df_pivot["LSTM Prediction"] = lstm_prediction.flatten()

            st.subheader("📉 Model Predictions:")
            st.write(df_pivot[["location", "RF Prediction", "LSTM Prediction"]])

            # 🏷️ **Classify AQI Levels**
            def classify_aqi(value):
                if value <= 50:
                    return "✅ Good"
                elif value <= 100:
                    return "🟡 Moderate"
                elif value <= 150:
                    return "🟠 Unhealthy (Sensitive Groups)"
                elif value <= 200:
                    return "🔴 Unhealthy"
                elif value <= 300:
                    return "🟣 Very Unhealthy"
                else:
                    return "⚫ Hazardous"

            df_pivot["RF AQI Category"] = df_pivot["RF Prediction"].apply(classify_aqi)
            df_pivot["LSTM AQI Category"] = df_pivot["LSTM Prediction"].apply(classify_aqi)

            # ✅ Display results
            st.subheader("📌 Air Quality Classification:")
            st.write(df_pivot[["location", "RF Prediction", "RF AQI Category", "LSTM Prediction", "LSTM AQI Category"]])

            # 📊 **AQI Distribution - Histogram**
            st.subheader("📊 AQI Distribution")

            fig, ax = plt.subplots()
            ax.hist(df_pivot["RF Prediction"], bins=20, alpha=0.6, color="blue", label="RF Prediction")
            ax.hist(df_pivot["LSTM Prediction"], bins=20, alpha=0.6, color="red", label="LSTM Prediction")

            ax.set_xlabel("AQI Levels")
            ax.set_ylabel("Frequency")
            ax.set_title("Histogram of Predicted AQI Levels")
            ax.legend()

            # Show in Streamlit
            st.pyplot(fig)

            # **🌍 AQI Map Visualization**
            if all(col in df_pivot.columns for col in ["latitude", "longitude", "RF Prediction"]):
                st.subheader("🌍 AQI Map Visualization")

                # Convert latitude, longitude, and AQI value to numeric (if needed)
                df_pivot["latitude"] = pd.to_numeric(df_pivot["latitude"], errors="coerce")
                df_pivot["longitude"] = pd.to_numeric(df_pivot["longitude"], errors="coerce")
                df_pivot["RF Prediction"] = pd.to_numeric(df_pivot["RF Prediction"], errors="coerce")

                # Drop any rows with missing values
                df_cleaned = df_pivot.dropna(subset=["latitude", "longitude", "RF Prediction"])

                if df_cleaned.empty:
                    st.error("❌ No valid data to display on the map. Check for missing values in latitude, longitude, or AQI values.")
                else:
                    # Create an interactive AQI map
                    fig = px.scatter_mapbox(
                        df_cleaned, 
                        lat="latitude", 
                        lon="longitude", 
                        size="RF Prediction",  
                        color="RF Prediction",  
                        hover_name="location", 
                        hover_data=["RF Prediction"], 
                        color_continuous_scale=px.colors.sequential.Plasma, 
                        size_max=15, 
                        zoom=10, 
                        title="Air Quality Index (AQI) Visualization"
                    )

                    fig.update_layout(
                        mapbox_style="open-street-map",  
                        margin={"r":0,"t":30,"l":0,"b":0}
                    )

                    st.plotly_chart(fig, use_container_width=True)

            else:
                st.error("❌ Latitude, Longitude, or RF Prediction column not found. Please check your dataset.")

        else:
            st.error("❌ Missing 'indicator' or 'value' column in the uploaded CSV.")

    except Exception as e:
        st.error(f"❌ Error processing the file: {e}")




