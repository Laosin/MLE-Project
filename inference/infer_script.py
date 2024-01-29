import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Define the neural network model (should be the same as used in training)
class IrisNet(nn.Module):
    def __init__(self):
        super(IrisNet, self).__init__()
        self.fc1 = nn.Linear(4, 100)  # 4 input features, hidden layer with 100 neurons
        self.fc2 = nn.Linear(100, 3)  # Output layer with 3 classes

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def main():
    # Load the inference dataset
    inference_data = pd.read_csv('datafiles/inference_set.csv')

    # Preprocess the data (ensure this matches training preprocessing)
    scaler = StandardScaler()
    X_infer = scaler.fit_transform(inference_data.iloc[:, :-1])
    X_tensor_infer = torch.tensor(X_infer, dtype=torch.float32)

    # Create a TensorDataset and DataLoader
    infer_dataset = TensorDataset(X_tensor_infer)
    infer_loader = DataLoader(infer_dataset, batch_size=16)

    # Load the trained model
    model = IrisNet()
    model.load_state_dict(torch.load('iris_model.pth'))
    model.eval()

    # Perform inference
    predictions = []
    with torch.no_grad():
        for inputs in infer_loader:
            outputs = model(inputs[0])
            _, predicted = torch.max(outputs.data, 1)
            predictions.extend(predicted.numpy())

    # Print or save the predictions
    print("Predictions:", predictions)

if __name__ == '__main__':
    main()
