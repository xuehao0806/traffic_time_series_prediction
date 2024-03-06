import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math
from models import LSTMModel, RNNModel, GRUModel, SimpleTransformerModel
import pandas as pd
import matplotlib.pyplot as plt


def train_model(model, train_inputs, train_labels, learning_rate, epochs, batch_size=64):
    # Wrap training data and labels into TensorDataset and use DataLoader
    train_dataset = TensorDataset(train_inputs, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    # Train the model
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # Print average loss after each epoch
        print(f'Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}')


def evaluate_model(model, test_inputs, test_labels, means, stds):
    model.eval()
    with torch.no_grad():
        predictions = model(test_inputs)
        predictions = predictions.squeeze().detach().numpy()
        test_labels = test_labels.squeeze().detach().numpy()

        predictions = predictions * stds.reshape(-1) + means.reshape(-1)
        # Initialize lists to store evaluation metrics
        rmses = []
        maes = []
        mapes = []

        # Define a threshold to filter out samples with near-zero true values
        threshold = 1e-6

        # Calculate metrics for each dimension of test_labels
        for i in range(test_labels.shape[1]):
            true = test_labels[:, i]
            pred = predictions[:, i]
            rmse = math.sqrt(mean_squared_error(true, pred))
            mae = mean_absolute_error(true, pred)

            # Calculate MAPE only for samples with true values above the threshold
            valid_indices = true > threshold
            valid_true = true[valid_indices]
            valid_pred = pred[valid_indices]
            if len(valid_true) > 0:
                mape = np.mean(np.abs((valid_true - valid_pred) / valid_true)) * 100
            else:
                mape = float('nan')

            rmses.append(rmse)
            maes.append(mae)
            mapes.append(mape)

        # Create a DataFrame to store and display evaluation metrics
        metrics_df = pd.DataFrame({
            'RMSE': rmses,
            'MAE': maes,
            'MAPE': mapes
        }, index=[f'Time Step {i + 1}' for i in range(test_labels.shape[1])])

        print(metrics_df)
        return metrics_df


def plot_predictions(model, inputs, labels, means, stds, vis_len, vis_step):
    model.eval()
    with torch.no_grad():
        # Obtain the predictions
        predictions = model(inputs[:vis_len]).squeeze().detach().numpy()
        # Denormalize the predictions and actual values
        predictions = predictions * stds.reshape(-1) + means.reshape(-1)
        predictions = predictions[:, vis_step - 1]
        actuals = labels[:vis_len, vis_step - 1].squeeze().detach().numpy()

    # Plot the graph
    plt.figure(figsize=(10, 6))
    plt.plot(predictions, label=f'Prediction (Step {vis_step})', color='red', linewidth=2, linestyle='-')
    plt.plot(actuals, label='Actual', color='black', linewidth=2, linestyle='--')
    plt.title('Predictions vs Actual')
    plt.xlabel('Time')
    plt.ylabel('Number of vehicles')
    plt.legend()
    plt.savefig('visualisation/predictions_vs_actual.png', dpi=300)


if __name__ == "__main__":
    # a. Load data
    file_path = 'data/'
    month = 1
    target = "Observations"
    input_seq_len = 12
    output_seq_len = 6
    file_name = f'traffic_data_M{month}_{target}_IN{input_seq_len}_OUT{output_seq_len}.npz'
    data = np.load(file_path + file_name)
    train_inputs = torch.Tensor(data['train_inputs'])
    train_labels = torch.Tensor(data['train_labels'])
    test_inputs = torch.Tensor(data['test_inputs'])
    test_labels = torch.Tensor(data['test_labels'])
    means = data['means']
    stds = data['stds']
    # b. Construct LSTM model
    input_dim = 1  # Default number of features is 1
    hidden_dim = 32  # hidden layer dimension
    output_seq_len = 6  # Output dimension
    model = LSTMModel(input_dim, hidden_dim, output_seq_len)
    vis_len = 100 # Length of time for visualization
    vis_step = 6 # Prediction step number for visualization
    # c. Train the model
    learning_rate = 0.005
    epochs = 20
    train_model(model, train_inputs, train_labels, learning_rate, epochs)

    # d. Evaluate the model
    evaluate_model(model, test_inputs, test_labels, means, stds)

    # e. Plot model prediction results
    plot_predictions(model, test_inputs, test_labels, means, stds, vis_len, vis_step)
