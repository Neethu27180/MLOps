#!/bin/bash

# Define variables
SPACE_NAME="Neethu2718/Visit_with_us" # Replace with your Hugging Face Space name
IMAGE_NAME="tourism_app"
HF_TOKEN="" # This will be set from GitHub Actions Secrets

# Login to Hugging Face Container Registry
echo "Logging in to Hugging Face Container Registry..."
echo $HF_TOKEN | docker login https://ghcr.io -u $HF_TOKEN --password-stdin
if [ $? -ne 0 ]; then
    echo "Docker login failed."
    exit 1
fi
echo "Docker login successful."

# Build the Docker image
echo "Building Docker image..."
docker build -t $IMAGE_NAME .
if [ $? -ne 0 ]; then
    echo "Docker build failed."
    exit 1
fi
echo "Docker image built successfully."

# Tag the Docker image for the Hugging Face Space
echo "Tagging Docker image..."
docker tag $IMAGE_NAME ghcr.io/$SPACE_NAME:$IMAGE_NAME
if [ $? -ne 0 ]; then
    echo "Docker tagging failed."
    exit 1
fi
echo "Docker image tagged successfully."

# Push the Docker image to the Hugging Face Space
echo "Pushing Docker image to Hugging Face Space..."
docker push ghcr.io/$SPACE_NAME:$IMAGE_NAME
if [ $? -ne 0 ]; then
    echo "Docker push failed."
    exit 1
fi
echo "Docker image pushed to Hugging Face Space successfully."

echo "Deployment script finished."
