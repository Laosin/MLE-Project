# Dockerfile for training
# Start from a base Python image
FROM python:3.10

# Set the working directory in the container
WORKDIR /app

# Copy the dependencies file to the working directory
COPY requirements_train.txt .

# Install any dependencies
RUN pip install --no-cache-dir -r requirements_train.txt

# Copy the content of the local src directory to the working directory
COPY train_script.py .

# Command to run the training script
CMD ["python", "train_script.py"]