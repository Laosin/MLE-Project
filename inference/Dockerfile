# Dockerfile for training
# Start from a base Python image
FROM python:3.10

# Set the working directory in the container to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements/requirements_infer.txt

# Define environment variable (if needed)
# ENV NAME Value

# Run infer_script.py when the container launches
CMD ["python", "infer_script.py"]