
import streamlit as st
import pandas as pd
import joblib
import os

# Define the paths to the model and the training data for preprocessing
MODEL_PATH = "model_building/logistic_regression_model.joblib"
X_TRAIN_PATH = "data_registration/X_train.csv"

# Load the trained model
@st.cache_resource
def load_model(model_path):
    try:
        model = joblib.load(model_path)
        return model
    except FileNotFoundError:
        st.error(f"Model file not found at {model_path}. Please ensure the model is in the correct location.")
        return None
    except Exception as e:
        st.error(f"Error loading the model: {e}")
        return None

# Load the training data for column alignment during preprocessing
@st.cache_data
def load_training_data(data_path):
    try:
        df = pd.read_csv(data_path)
        return df
    except FileNotFoundError:
        st.error(f"Training data file not found at {data_path}. Please ensure the data is in the correct location.")
        return None
    except Exception as e:
        st.error(f"Error loading the training data: {e}")
        return None

# Data preprocessing function (must match the preprocessing in model training)
def preprocess_input(input_df, X_train_df):
    # Identify categorical and numerical columns from the training data
    categorical_cols = X_train_df.select_dtypes(include=['object', 'bool']).columns
    numerical_cols = X_train_df.select_dtypes(include=['float64', 'int64']).columns

    # Handle boolean columns in input_df by converting them to int
    for col in categorical_cols:
        if input_df[col].dtype == 'bool':
            input_df[col] = input_df[col].astype(int)

    # Apply one-hot encoding to categorical columns in the input data
    input_df = pd.get_dummies(input_df, columns=categorical_cols, drop_first=True)

    # Align columns - crucial for consistent feature sets between training and input
    train_cols = X_train_df.columns
    input_cols = input_df.columns
    missing_in_input = set(train_cols) - set(input_cols)

    # Add missing columns to the input DataFrame and set their values to 0
    for c in missing_in_input:
        input_df[c] = 0

    # Ensure the order of columns is the same as the training data
    input_df = input_df[train_cols]

    # Apply StandardScaler to numerical columns (using the scaler fitted on training data - not explicitly saved, so fitting on X_train_df again for demonstration)
    # In a real MLOps scenario, the scaler should be saved and loaded as well.
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    # Fit on training data and transform input data
    scaler.fit(X_train_df[numerical_cols])
    input_df[numerical_cols] = scaler.transform(input_df[numerical_cols])

    return input_df


# Load the model and training data
model = load_model(MODEL_PATH)
X_train_df = load_training_data(X_TRAIN_PATH)

# Streamlit app layout
st.title("Wellness Tourism Package Purchase Prediction")
st.write("Enter customer details to predict the likelihood of purchasing the Wellness Tourism Package.")

if model is not None and X_train_df is not None:
    # Create input fields for customer details
    st.header("Customer Details")
    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    typeofcontact = st.selectbox("Type of Contact", ['Company Invited', 'Self Enquiry'])
    citytier = st.selectbox("City Tier", [1, 2, 3])
    durationofpitch = st.number_input("Duration of Pitch (minutes)", min_value=0.0, value=10.0)
    occupation = st.selectbox("Occupation", ['Salaried', 'Free Lancer', 'Large Business', 'Small Business'])
    gender = st.selectbox("Gender", ['Male', 'Female'])
    numberofpersonvisiting = st.number_input("Number of People Visiting", min_value=1, value=1)
    preferredpropertystar = st.selectbox("Preferred Property Star", [1.0, 2.0, 3.0, 4.0, 5.0])
    maritalstatus = st.selectbox("Marital Status", ['Single', 'Married', 'Divorced'])
    numberoftrips = st.number_input("Number of Trips Annually", min_value=0.0, value=1.0)
    passport = st.selectbox("Passport", [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
    pitchsatisfactionscore = st.slider("Pitch Satisfaction Score", 1, 5, 3)
    owncar = st.selectbox("Own Car", [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
    numberofchildrenvisiting = st.number_input("Number of Children Visiting (under 5)", min_value=0.0, value=0.0)
    designation = st.selectbox("Designation", ['Manager', 'Executive', 'Senior Manager', 'AVP', 'VP', 'Director'])
    monthlyincome = st.number_input("Monthly Income", min_value=0.0, value=25000.0)

    # Create a button to make predictions
    if st.button("Predict Purchase"):
        # Create a DataFrame from the input data
        input_data = {
            'Age': age,
            'TypeofContact': typeofcontact,
            'CityTier': citytier,
            'DurationOfPitch': durationofpitch,
            'Occupation': occupation,
            'Gender': gender,
            'NumberOfPersonVisiting': numberofpersonvisiting,
            'PreferredPropertyStar': preferredpropertystar,
            'MaritalStatus': maritalstatus,
            'NumberOfTrips': numberoftrips,
            'Passport': passport,
            'PitchSatisfactionScore': pitchsatisfactionscore,
            'OwnCar': owncar,
            'NumberOfChildrenVisiting': numberofchildrenvisiting,
            'Designation': designation,
            'MonthlyIncome': monthlyincome
        }
        input_df = pd.DataFrame([input_data])

        # Preprocess the input data
        processed_input_df = preprocess_input(input_df.copy(), X_train_df.copy()) # Use copy to avoid modifying cached data

        # Make a prediction
        prediction = model.predict(processed_input_df)
        prediction_proba = model.predict_proba(processed_input_df)[:, 1]

        # Display the prediction
        st.subheader("Prediction Result")
        if prediction[0] == 1:
            st.success(f"The model predicts that the customer is LIKELY to purchase the Wellness Tourism Package.")
        else:
            st.info(f"The model predicts that the customer is UNLIKELY to purchase the Wellness Tourism Package.")

        st.write(f"Confidence Score: {prediction_proba[0]:.2f}")

else:
    st.warning("Model or training data could not be loaded. Please check the file paths.")
