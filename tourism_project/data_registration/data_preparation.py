
import pandas as pd
from sklearn.model_selection import train_test_split
from huggingface_hub import HfApi
import os

# Create the data_registration directory if it doesn't exist
os.makedirs('tourism_project/data_registration', exist_ok=True)

# Load data (assuming tourism.csv is already in the repo or can be downloaded)
# For now, let's assume it's in the repo at tourism_project/data_registration
try:
    df = pd.read_csv('tourism_project/data_registration/tourism.csv')
except FileNotFoundError:
    print('tourism.csv not found. Please ensure it is in the tourism_project/data_registration directory in the repo.')
    exit(1)

# Remove duplicate rows
df.drop_duplicates(inplace=True)

# Remove unnecessary columns
irrelevant_columns = ['Unnamed: 0', 'CustomerID']
df = df.drop(columns=irrelevant_columns)

# Split data
X = df.drop('ProdTaken', axis=1)
y = df['ProdTaken']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Save data locally
X_train.to_csv('tourism_project/data_registration/X_train.csv', index=False)
X_test.to_csv('tourism_project/data_registration/X_test.csv', index=False)
y_train.to_csv('tourism_project/data_registration/y_train.csv', index=False)
y_test.to_csv('tourism_project/data_registration/y_test.csv', index=False)

# Upload data to Hugging Face (using the HF_TOKEN secret from GitHub Actions)
# Replace with your Hugging Face username and dataset name
repo_id = 'Neethu2718/MLOps'

api = HfApi(token=os.environ['HF_TOKEN']) # Use environment variable in GitHub Actions

api.upload_file(
    path_or_fileobj='tourism_project/data_registration/X_train.csv',
    path_in_repo='X_train.csv',
    repo_id=repo_id,
    repo_type='space',
)

api.upload_file(
    path_or_fileobj='tourism_project/data_registration/X_test.csv',
    path_in_repo='X_test.csv',
    repo_id=repo_id,
    repo_type='space',
)

api.upload_file(
    path_or_fileobj='tourism_project/data_registration/y_train.csv',
    path_in_repo='y_train.csv',
    repo_id=repo_id,
    repo_type='space',
)

api.upload_file(
    path_or_fileobj='tourism_project/data_registration/y_test.csv',
    path_in_repo='y_test.csv',
    repo_id=repo_id,
    repo_type='space',
)

print('Data preparation and upload completed.')
