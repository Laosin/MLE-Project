
# MLE-Project (Iris)

## Description
This project involves the training and inference of a machine learning model using the Iris dataset. The repository is structured to separate the training and inference environments, ensuring a clear workflow and easy reproducibility.

## Repository Structure
```
project_repository/
│
├── datafiles/
│   ├── train_set.csv      # Training data file
│   └── inference_set.csv  # Inference data file
│
├── training/
│   ├── Dockerfile         # Dockerfile for training environment
│   ├── train_script.py    # Python script for training the model
│   └── requirements_train.txt  # Dependencies for training
│
├── inference/
│   ├── Dockerfile         # Dockerfile for inference environment
│   ├── infer_script.py    # Python script for model inference
│   └── requirements_infer.txt  # Dependencies for inference
│
├── unittests/
│   └── unittests.py
│
└── README.md              # README file
```

## Setup and Installation

### Prerequisites
- Docker
- Python 3.x

### Installation
1. Clone the repository:
   ```
   git clone [repository URL]
   ```
2. Navigate to the cloned directory:
   ```
   cd your_project_repository
   ```

### Training the Model
1. Build the Docker image for training:
   ```
   docker build -t iris-train -f training/Dockerfile .
   ```
2. Run the Docker container:
   ```
   docker run iris-train
   ```

### Running Inference
1. Build the Docker image for inference:
   ```
   docker build -t iris-infer -f inference/Dockerfile .
   ```
2. Run the Docker container:
   ```
   docker run iris-infer
   ```

## Usage

### Training
- The training script (`train_script.py`) is located in the `training/` directory. This script trains the model using `train_set.csv` and saves the trained model for inference.

### Inference
- The inference script (`infer_script.py`) is located in the `inference/` directory. This script loads the trained model and performs inference on `inference_set.csv`.

## Authors
- O. K.
