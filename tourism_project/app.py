import streamlit as st
import pandas as pd
import joblib
from huggingface_hub import hf_hub_download
import os

# ✅ Fix permissions by redirecting caches
os.environ["STREAMLIT_HOME"] = os.path.join(os.getcwd(), ".streamlit")
os.environ["HF_HOME"] = os.path.join(os.getcwd(), ".hf_home")
os.environ["TRANSFORMERS_CACHE"] = os.path.join(os.getcwd(), ".hf_home", "transformers")

os.makedirs(os.environ["STREAMLIT_HOME"], exist_ok=True)
os.makedirs(os.environ["HF_HOME"], exist_ok=True)
os.makedirs(os.environ["TRANSFORMERS_CACHE"], exist_ok=True)

# Define the model repository ID and filenames on Hugging Face
repo_id = "Neethu2718/Visit_with_us"
model_filename = "logistic_regression_model.joblib"
scaler_filename = "scaler.joblib"

# Local download directory
download_dir = os.path.join(os.getcwd(), "downloaded_model")
os.makedirs(download_dir, exist_ok=True)

# ✅ Download model + scaler (repo_type fixed)
model_path = hf_hub_download(
    repo_id=repo_id, filename=model_filename,
    local_dir=download_dir, local_dir_use_symlinks=False
)
scaler_path = hf_hub_download(
    repo_id=repo_id, filename=scaler_filename,
    local_dir=download_dir, local_dir_use_symlinks=False
)

# Load trained model & scaler
model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

# ---------------------- Streamlit UI ----------------------
st.title("Wellness Tourism Package Purchase Prediction")
st.write("Enter customer details to predict the likelihood of purchasing the Wellness Tourism Package.")

# Input fields
age = st.number_input("Age", min_value=0, max_value=120, value=30)
typeofcontact = st.selectbox("Type of Contact", ['Company Invited', 'Self Inquiry'])
citytier = st.selectbox("City Tier", [1, 2, 3])
durationofpitch = st.number_input("Duration of Pitch (minutes)", min_value=0.0, value=10.0)
occupation = st.selectbox("Occupation", ['Salaried', 'Small Business', 'Large Business', 'Free Lancer'])
gender = st.selectbox("Gender", ['Male', 'Female'])
numberofpersonvisiting = st.number_input("Number of Persons Visiting", min_value=1, value=1)
preferredpropertystar = st.selectbox("Preferred Property Star", [1, 2, 3, 4, 5])
maritalstatus = st.selectbox("Marital Status", ['Single', 'Married', 'Divorced'])
numberoftrips = st.number_input("Number of Trips (annually)", min_value=0.0, value=1.0)
passport = st.selectbox("Passport", [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
pitchsatisfactionscore = st.slider("Pitch Satisfaction Score", 1, 5, 3)
owncar = st.selectbox("Own Car", [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
numberofchildrenvisiting = st.number_input("Number of Children Visiting", min_value=0.0, value=0.0)
designation = st.selectbox("Designation", ['Executive', 'Manager', 'Senior Manager', 'AVP', 'VP'])
monthlyincome = st.number_input("Monthly Income", min_value=0.0, value=20000.0)
# ✅ Added missing inputs
numberoffollowups = st.number_input("Number of Followups", min_value=0, value=0)
productpitched = st.selectbox("Product Pitched", ['Basic', 'Standard', 'Deluxe', 'Super Deluxe', 'King'])

if st.button("Predict"):
    # Create a DataFrame from inputs
    input_data = pd.DataFrame({
        'Age': [age],
        'TypeofContact': [typeofcontact],
        'CityTier': [citytier],
        'DurationOfPitch': [durationofpitch],
        'Occupation': [occupation],
        'Gender': [gender],
        'NumberOfPersonVisiting': [numberofpersonvisiting],
        'PreferredPropertyStar': [preferredpropertystar],
        'MaritalStatus': [maritalstatus],
        'NumberOfTrips': [numberoftrips],
        'Passport': [passport],
        'PitchSatisfactionScore': [pitchsatisfactionscore],
        'OwnCar': [owncar],
        'NumberOfChildrenVisiting': [numberofchildrenvisiting],
        'Designation': [designation],
        'MonthlyIncome': [monthlyincome],
        'NumberOfFollowups': [numberoffollowups],
        'ProductPitched': [productpitched]
    })

    # Categorical columns
    categorical_cols = ['TypeofContact', 'Occupation', 'Gender', 'ProductPitched', 'MaritalStatus', 'Designation']

    # Dummy row to guarantee all categories exist
    dummy_data = pd.DataFrame({
        'Age': [0], 'TypeofContact': ['Company Invited'], 'CityTier': [0], 'DurationOfPitch': [0],
        'Occupation': ['Salaried'], 'Gender': ['Male'], 'NumberOfPersonVisiting': [0],
        'PreferredPropertyStar': [0], 'MaritalStatus': ['Single'], 'NumberOfTrips': [0],
        'Passport': [0], 'PitchSatisfactionScore': [0], 'OwnCar': [0],
        'NumberOfChildrenVisiting': [0], 'Designation': ['Executive'], 'MonthlyIncome': [0],
        'NumberOfFollowups': [0], 'ProductPitched': ['Basic']
    })

    # One-hot encoding
    combined_data = pd.concat([dummy_data, input_data], ignore_index=True)
    combined_data = pd.get_dummies(combined_data, columns=categorical_cols, drop_first=True)
    input_data_processed = combined_data.iloc[1:]  # drop dummy

    # Columns from training (spelling fixed → "Self Inquiry")
    training_columns = [
        'Age', 'CityTier', 'DurationOfPitch', 'NumberOfPersonVisiting', 'NumberOfFollowups',
        'PreferredPropertyStar', 'NumberOfTrips', 'Passport', 'PitchSatisfactionScore',
        'OwnCar', 'NumberOfChildrenVisiting', 'MonthlyIncome',
        'TypeofContact_Self Inquiry',
        'Occupation_Large Business', 'Occupation_Salaried', 'Occupation_Small Business',
        'Occupation_Free Lancer',
        'Gender_Female',  # Male is dropped because of drop_first=True
        'ProductPitched_Deluxe', 'ProductPitched_King', 'ProductPitched_Standard',
        'ProductPitched_Super Deluxe',
        'MaritalStatus_Married', 'MaritalStatus_Single', 'MaritalStatus_Divorced',
        'Designation_Manager', 'Designation_Senior Manager', 'Designation_VP', 'Designation_AVP'
    ]

    # Ensure all training columns exist
    for col in training_columns:
        if col not in input_data_processed.columns:
            input_data_processed[col] = 0

    # Reorder
    input_data_processed = input_data_processed[training_columns]

    # Apply scaling
    numerical_cols_initial = [
        'Age', 'CityTier', 'DurationOfPitch', 'NumberOfPersonVisiting',
        'NumberOfFollowups', 'PreferredPropertyStar', 'NumberOfTrips', 'Passport',
        'PitchSatisfactionScore', 'OwnCar', 'NumberOfChildrenVisiting', 'MonthlyIncome'
    ]
    scaled_numerical_cols = [col for col in numerical_cols_initial if col in input_data_processed.columns]
    input_data_processed[scaled_numerical_cols] = scaler.transform(input_data_processed[scaled_numerical_cols])

    # Prediction
    prediction = model.predict(input_data_processed)
    prediction_proba = model.predict_proba(input_data_processed)[:, 1]

    if prediction[0] == 1:
        st.success(f"✅ The customer is likely to purchase the Wellness Tourism Package (Probability: {prediction_proba[0]:.2f})")
    else:
        st.info(f"❌ The customer is unlikely to purchase the Wellness Tourism Package (Probability: {prediction_proba[0]:.2f})")
