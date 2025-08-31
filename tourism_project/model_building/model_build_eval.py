
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import mlflow
import joblib
from huggingface_hub import HfApi
import os

# Define the paths to the data and the model
local_data_path = 'tourism_project/data_registration'
local_model_path = 'tourism_project/model_building/logistic_regression_model.joblib'
repo_id = 'Neethu2718/MLOps' # Replace with your Hugging Face username and repo name

# Load data from local files (downloaded in a previous workflow step)
try:
    X_train = pd.read_csv(f'{local_data_path}/X_train.csv')
    X_test = pd.read_csv(f'{local_data_path}/X_test.csv')
    y_train = pd.read_csv(f'{local_data_path}/y_train.csv')
    y_test = pd.read_csv(f'{local_data_path}/y_test.csv')
except FileNotFoundError:
    print('Train/test data not found locally. Ensure the Data Preparation step ran successfully.')
    exit(1)

# Data preprocessing and transformation
categorical_cols = X_train.select_dtypes(include=['object']).columns
numerical_cols = X_train.select_dtypes(include=['float64', 'int64']).columns

X_train = pd.get_dummies(X_train, columns=categorical_cols, drop_first=True)
X_test = pd.get_dummies(X_test, columns=categorical_cols, drop_first=True)

# Align columns after one-hot encoding
train_cols = X_train.columns
test_cols = X_test.columns
missing_in_test = set(train_cols) - set(test_cols)
for c in missing_in_test:
    X_test[c] = 0
missing_in_train = set(test_cols) - set(train_cols)
for c in missing_in_train:
    X_train[c] = 0
X_test = X_test[train_cols]

scaler = StandardScaler()
X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])

# Define and train model
model = LogisticRegression(random_state=42, max_iter=5000)

# Start MLflow run
with mlflow.start_run(run_name='Logistic Regression Training'):
    model.fit(X_train, y_train.values.ravel())

    # Evaluate model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

    # Log parameters and metrics
    mlflow.log_param('solver', 'lbfgs')
    mlflow.log_param('random_state', 42)
    mlflow.log_param('max_iter', 5000)
    mlflow.log_metric('accuracy', accuracy)
    mlflow.log_metric('precision', precision)
    mlflow.log_metric('recall', recall)
    mlflow.log_metric('f1', f1)
    mlflow.log_metric('roc_auc', roc_auc)

    print('Model training and evaluation completed.')
    print(f'Accuracy: {accuracy}')
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'F1-score: {f1}')
    print(f'ROC AUC: {roc_auc}')

    # Save the trained model locally
    os.makedirs('tourism_project/model_building', exist_ok=True)
    joblib.dump(model, local_model_path)
    print(f'Model saved locally at {local_model_path}')

    # Upload the saved model file to the Hugging Face Model Hub
    api = HfApi(token=os.environ['HF_TOKEN']) # Use environment variable in GitHub Actions
    api.upload_file(
        path_or_fileobj=local_model_path,
        path_in_repo='logistic_regression_model.joblib',
        repo_id=repo_id,
        repo_type='space',
    )
    print(f'Model uploaded to Hugging Face Model Hub: {repo_id}')

