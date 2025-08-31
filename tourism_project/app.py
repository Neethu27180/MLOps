import streamlit as st
import pandas as pd
import joblib
from huggingface_hub import hf_hub_download
import os
from sklearn.preprocessing import StandardScaler # Import StandardScaler for the temporary scaler

# ✅ Fix permissions by redirecting caches
os.environ["STREAMLIT_HOME"] = os.path.join(os.getcwd(), ".streamlit")
os.environ["HF_HOME"] = os.path.join(os.getcwd(), ".hf_home")
os.environ["TRANSFORMERS_CACHE"] = os.path.join(os.getcwd(), ".hf_home", "transformers")

os.makedirs(os.environ["STREAMLIT_HOME"], exist_ok=True)
os.makedirs(os.environ["HF_HOME"], exist_ok=True)
os.makedirs(os.environ["TRANSFORMERS_CACHE"], exist_ok=True)


# Define the model repository ID and filename on Hugging Face
repo_id = "Neethu2718/Visit_with_us"
model_filename = "logistic_regression_model.joblib"
scaler_filename = "scaler.joblib" # Define scaler filename

# Define the local directory to download files to
# Although HF_HOME is set, explicitly defining a local_dir in hf_hub_download is good practice
download_dir = os.path.join(os.getcwd(), "downloaded_model") # Download within the working directory
os.makedirs(download_dir, exist_ok=True) # Explicitly create the download directory

# Download the model file from Hugging Face
model_path = hf_hub_download(repo_id=repo_id, filename=model_filename, repo_type="space", local_dir=download_dir, local_dir_use_symlinks=False)

# Load the trained model
model = joblib.load(model_path)

# Download the scaler file from Hugging Face
scaler_path = hf_hub_download(repo_id=repo_id, filename=scaler_filename, repo_type="space", local_dir=download_dir, local_dir_use_symlinks=False)

# Load the fitted scaler
scaler = joblib.load(scaler_path)


st.title("Wellness Tourism Package Purchase Prediction")

st.write("Enter customer details to predict the likelihood of purchasing the Wellness Tourism Package.")

# Add input fields for the features
age = st.number_input("Age", min_value=0, max_value=120, value=30)
typeofcontact = st.selectbox("Type of Contact", ['Company Invited', 'Self Inquiry'])
citytier = st.selectbox("City Tier", [1, 2, 3])
durationofpitch = st.number_input("Duration of Pitch (minutes)", min_value=0.0, value=10.0)
occupation = st.selectbox("Occupation", ['Salaried', 'Small Business', 'Large Business', 'Free Lancer'])
gender = st.selectbox("Gender", ['Male', 'Female'])
numberofpersonvisiting = st.number_input("NumberOfPersonsVisiting", min_value=1, value=1)
preferredpropertystar = st.selectbox("Preferred Property Star", [1, 2, 3, 4, 5])
maritalstatus = st.selectbox("Marital Status", ['Single', 'Married', 'Divorced'])
numberoftrips = st.number_input("Number of Trips (annually)", min_value=0.0, value=1.0)
passport = st.selectbox("Passport", [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
pitchsatisfactionscore = st.slider("Pitch Satisfaction Score", 1, 5, 3)
owncar = st.selectbox("Own Car", [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
numberofchildrenvisiting = st.number_input("Number of Children Visiting", min_value=0.0, value=0.0)
designation = st.selectbox("Designation", ['Executive', 'Manager', 'Senior Manager', 'AVP', 'VP'])
monthlyincome = st.number_input("Monthly Income", min_value=0.0, value=20000.0)


if st.button("Predict"):
    # Create a DataFrame from the input values
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
        'MonthlyIncome': [monthlyincome]
    })

    # The following preprocessing steps should match the training preprocessing
    # Apply one-hot encoding to categorical columns - ensure consistency with training
    categorical_cols = ['TypeofContact', 'Occupation', 'Gender', 'ProductPitched', 'MaritalStatus', 'Designation'] # Include ProductPitched here as it was in training data
    # Need a dummy row with all possible categories from the training data to ensure all columns are created during one-hot encoding
    # This is a simplified approach; in a real-world scenario, you would save the list of columns from training
    dummy_data = pd.DataFrame({
        'Age': [0], 'TypeofContact': ['Company Invited'], 'CityTier': [0], 'DurationOfPitch': [0],
        'Occupation': ['Salaried'], 'Gender': ['Male'], 'NumberOfPersonVisiting': [0],
        'PreferredPropertyStar': [0], 'MaritalStatus': ['Single'], 'NumberOfTrips': [0],
        'Passport': [0], 'PitchSatisfactionScore': [0], 'OwnCar': [0],
        'NumberOfChildrenVisiting': [0], 'Designation': ['Executive'], 'MonthlyIncome': [0],
        'ProductPitched': ['Basic'] # Include a value for ProductPitched
    })
    combined_data = pd.concat([dummy_data, input_data], ignore_index=True)
    combined_data = pd.get_dummies(combined_data, columns=categorical_cols, drop_first=True)
    input_data_processed = combined_data.iloc[1:] # Remove the dummy row

    # Ensure the order of columns and presence of all training columns
    # In a real application, save the list of columns from X_train after preprocessing and use it here
    # For this example, we'll infer from the loaded model's expected features (requires model to have feature names)
    # If the model doesn't have feature names, you'd need to save and load them separately
    # Assuming the model was trained on X_train which had columns like:
    # 'Age', 'CityTier', 'DurationOfPitch', 'NumberOfPersonVisiting', 'NumberOfFollowups',
    # 'PreferredPropertyStar', 'NumberOfTrips', 'Passport', 'PitchSatisfactionScore',
    # 'OwnCar', 'NumberOfChildrenVisiting', 'MonthlyIncome', 'TypeofContact_Self Inquiry',
    # 'Occupation_Large Business', 'Occupation_Salaried', 'Occupation_Small Business',
    # 'Gender_Female', 'Gender_Male', 'ProductPitched_Deluxe', 'ProductPitched_King',
    # 'ProductPitched_Standard', 'ProductPitched_Super Deluxe', 'MaritalStatus_Married',
    # 'MaritalStatus_Single', 'MaritalStatus_Unmarried', 'Designation_Executive',
    # 'Designation_Manager', 'Designation_Senior Manager', 'Designation_VP']

    # Create a list of the columns that were in X_train after preprocessing
    # This is a placeholder and should be replaced with the actual column names from your trained model's features
    training_columns = ['Age', 'CityTier', 'DurationOfPitch', 'NumberOfPersonVisiting', 'NumberOfFollowups',
                       'PreferredPropertyStar', 'NumberOfTrips', 'Passport', 'PitchSatisfactionScore',
                       'OwnCar', 'NumberOfChildrenVisiting', 'MonthlyIncome', 'TypeofContact_Self Enquiry',
                       'Occupation_Large Business', 'Occupation_Salaried', 'Occupation_Small Business',
                       'Gender_Female', 'Gender_Male', 'ProductPitched_Deluxe', 'ProductPitched_King',
                       'ProductPitched_Standard', 'ProductPitched_Super Deluxe', 'MaritalStatus_Married',
                       'MaritalStatus_Single', 'MaritalStatus_Unmarried', 'Designation_Executive',
                       'Designation_Manager', 'Designation_Senior Manager', 'Designation_VP']

    # Add missing columns with default value 0 (for one-hot encoded columns not present in input)
    for col in training_columns:
        if col not in input_data_processed.columns:
            input_data_processed[col] = 0

    # Ensure the order of columns is the same as during training
    input_data_processed = input_data_processed[training_columns]

    # Apply StandardScaler to numerical columns - use the loaded scaler
    numerical_cols_initial = ['Age', 'CityTier', 'DurationOfPitch', 'NumberOfPersonVisiting',
                              'NumberOfFollowups', 'PreferredPropertyStar', 'NumberOfTrips', 'Passport',
                              'PitchSatisfactionScore', 'OwnCar', 'NumberOfChildrenVisiting', 'MonthlyIncome']

    # Identify the scaled numerical columns in the processed data
    scaled_numerical_cols_processed = [col for col in numerical_cols_initial if col in input_data_processed.columns]

    # Use the loaded scaler to transform the input data
    input_data_processed[scaled_numerical_cols_processed] = scaler.transform(input_data_processed[scaled_numerical_cols_processed])


    # Make prediction
    prediction = model.predict(input_data_processed)
    prediction_proba = model.predict_proba(input_data_processed)[:, 1]

    if prediction[0] == 1:
        st.success(f"Prediction: The customer is likely to purchase the Wellness Tourism Package (Probability: {prediction_proba[0]:.2f})")
    else:
        st.info(f"Prediction: The customer is unlikely to purchase the Wellness Tourism Package (Probability: {prediction_proba[0]:.2f})")
