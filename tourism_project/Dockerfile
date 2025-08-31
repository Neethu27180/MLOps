# Use a lightweight Python base image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Set environment variable for Hugging Face Hub cache
ENV HF_HUB_CACHE=/app/.cache/huggingface/hub

# Create the cache directory with appropriate permissions
RUN mkdir -p /app/.cache/huggingface/hub && chmod -R 777 /app/.cache

# Copy the requirements file and install dependencies
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application files
COPY . .

# Expose the port the Streamlit app will run on
EXPOSE 7860

# Define the command to run the Streamlit app
CMD ["streamlit", "run", "app.py"]
