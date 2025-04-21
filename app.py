# import streamlit as st
# import pandas as pd
# import xgboost as xgb
# from sklearn.preprocessing import MinMaxScaler
# import joblib

# # Define the path to your model file
# model_path = '/Users/kothavamsi/Desktop/jaundice_vs/xgb_model.pkl'

# # Load the saved model using joblib
# try:
#     loaded_model = joblib.load(model_path)
#     st.write("Model loaded successfully!")
# except Exception as e:
#     st.error(f"Error loading model: {e}")

# st.title("Jaundice Risk Prediction")

# st.write("""
#     Enter the clinical parameters below to predict the risk of jaundice.
# """)

# # Input fields
# total_bilirubin = st.number_input("Total Bilirubin", min_value=0.0, max_value=20.0, step=0.01)
# direct_bilirubin = st.number_input("Direct Bilirubin", min_value=0.0, max_value=10.0, step=0.01)
# indirect_bilirubin = st.number_input("Indirect Bilirubin", min_value=0.0, max_value=15.0, step=0.01)
# a_g_ratio = st.number_input("A/G Ratio", min_value=0.5, max_value=2.5, step=0.01)
# age = st.number_input("Age", min_value=1, max_value=80, step=1)
# sex = st.selectbox("Sex", options=["Male", "Female"])

# # Preprocessing input data
# input_data = pd.DataFrame({
#     'Total_bilirubin': [total_bilirubin],
#     'Direct_bilirubin': [direct_bilirubin],
#     'Indirect_bilirubin': [indirect_bilirubin],
#     'A_G_ratio': [a_g_ratio],
#     'Age': [age],
#     'Sex_Male': [1 if sex == 'Male' else 0]
# })

# # Scaling continuous features
# scaler = MinMaxScaler()
# scaled_features = scaler.fit_transform(input_data[['Total_bilirubin', 'Direct_bilirubin', 'Indirect_bilirubin', 'A_G_ratio']])
# input_data[['Total_bilirubin', 'Direct_bilirubin', 'Indirect_bilirubin', 'A_G_ratio']] = scaled_features

# # Prediction
# if st.button('Predict'):
#     try:
#         # Use the loaded_model for prediction
#         prediction = loaded_model.predict(input_data)
#         if prediction == 1:
#             st.write("The patient is at risk of jaundice.")
#         else:
#             st.write("The patient is not at risk of jaundice.")
#     except Exception as e:
#         st.error(f"Prediction failed: {e}")





# import streamlit as st
# import numpy as np
# import pandas as pd
# import xgboost as xgb
# from sklearn.preprocessing import StandardScaler
# import joblib

# # Load the trained model and scaler
# xgb_model = joblib.load("xgb_model.pkl")  # Ensure the model is saved as .pkl
# try:
#     scaler = joblib.load("/Users/kothavamsi/Desktop/jaundice_vs/scaler.pkl")
#     st.write("Scaler loaded successfully.")
# except Exception as e:
#     st.error(f"Error loading scaler: {e}")

# # Define feature names (Ensure these match your dataset)
# feature_names = [
#     "Age_of_the_patient", "Gender_of_the_patient", "Total_Bilirubin", "Direct_Bilirubin",
#     "Alkphos_Alkaline_Phosphotase", "Sgpt_Alamine_Aminotransferase", "Sgot_Aspartate_Aminotransferase",
#     "Total_Protiens", "ALB_Albumin", "A/G_Ratio_Albumin_and_Globulin_Ratio"
# ]

# # Streamlit UI
# st.title("Jaundice Risk Prediction")

# # User input form
# with st.form("user_input_form"):
#     age = st.number_input("Age of the Patient", min_value=0, step=1)
#     gender = st.radio("Gender of the Patient", ["Male", "Female"])
#     total_bilirubin = st.number_input("Total Bilirubin", min_value=0.0, format="%.2f")
#     direct_bilirubin = st.number_input("Direct Bilirubin", min_value=0.0, format="%.2f")
#     alkphos = st.number_input("Alkaline Phosphotase (Alkphos)", min_value=0.0, format="%.2f")
#     sgpt = st.number_input("Alamine Aminotransferase (SGPT)", min_value=0.0, format="%.2f")
#     sgot = st.number_input("Aspartate Aminotransferase (SGOT)", min_value=0.0, format="%.2f")
#     total_protiens = st.number_input("Total Protiens", min_value=0.0, format="%.2f")
#     albumin = st.number_input("Albumin (ALB)", min_value=0.0, format="%.2f")
#     ag_ratio = st.number_input("A/G Ratio (Albumin and Globulin Ratio)", min_value=0.0, format="%.2f")

#     # Convert Gender to numerical (Ensure this matches training data encoding)
#     gender = 1 if gender == "Male" else 0

#     submitted = st.form_submit_button("Predict")

# # If user submits, preprocess and predict
# if submitted:
#     # Prepare input for prediction
#     user_input = np.array([[
#         age, gender, total_bilirubin, direct_bilirubin, alkphos, sgpt, sgot,
#         total_protiens, albumin, ag_ratio
#     ]])

#     # Scale input
#     try:
#         user_input_scaled = scaler.transform(user_input)
#         st.write("Input scaled successfully.")
#     except Exception as e:
#         st.error(f"Error scaling input: {e}")

#     # Predict
#     try:
#         prediction = xgb_model.predict(user_input_scaled)[0]
#         prediction_proba = xgb_model.predict_proba(user_input_scaled)[0][1]  # Probability of having jaundice

#         # Display results
#         st.subheader("Prediction Result")
#         if prediction == 1:
#             st.error(f"⚠️ High risk of jaundice! ")
#         else:
#             st.success(f"Low risk of jaundice. ")
#     except Exception as e:
#         st.error(f"Prediction failed: {e}")

import streamlit as st
import numpy as np
import pandas as pd
import joblib
import xgboost as xgb

# Load the trained model and scaler
try:
    xgb_model = joblib.load("xgb_model.pkl")
    scaler = joblib.load("scaler.pkl")
    st.write("Model and Scaler loaded successfully.")
except Exception as e:
    st.error(f"Error loading model or scaler: {e}")

# Define feature names (Ensure these match your dataset)
feature_names = [
    "Age_of_the_patient", "Gender_of_the_patient", "Total_Bilirubin", "Direct_Bilirubin",
    "Alkphos_Alkaline_Phosphotase", "Sgpt_Alamine_Aminotransferase", "Sgot_Aspartate_Aminotransferase",
    "Total_Protiens", "ALB_Albumin", "A/G_Ratio_Albumin_and_Globulin_Ratio"
]

st.title("Jaundice Risk Prediction")

# File uploader
uploaded_file = st.file_uploader("Upload a CSV file containing patient data", type=["csv"])

if uploaded_file:
    try:
        # Read CSV
        df = pd.read_csv(uploaded_file)
        st.write("File uploaded successfully!")
        
        # Check if required columns exist
        if not set(feature_names).issubset(df.columns):
            st.error("Uploaded file is missing required columns. Please check the format.")
        else:
            # Convert gender to numerical
            df["Gender_of_the_patient"] = df["Gender_of_the_patient"].map({"Male": 1, "Female": 0})
            
            # Scale input features
            df_scaled = scaler.transform(df[feature_names])
            
            # Predict
            predictions = xgb_model.predict(df_scaled)
            prediction_proba = xgb_model.predict_proba(df_scaled)[:, 1]
            
            # Add predictions to dataframe
            df["Jaundice_Risk"] = np.where(predictions == 1, "High Risk", "Low Risk")
            df["Risk_Probability"] = prediction_proba
            
            # Display results
            st.subheader("Prediction Results")
            st.write(df[["Age_of_the_patient","Jaundice_Risk", "Risk_Probability"]])
            
            # Provide download option
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download Predictions as CSV",
                data=csv,
                file_name="jaundice_predictions.csv",
                mime="text/csv"
            )
    except Exception as e:
        st.error(f"Error processing file: {e}")