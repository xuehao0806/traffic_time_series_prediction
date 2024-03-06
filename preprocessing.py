import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def create_and_save_seq2seq_dataset(file_path, file_name, month, target, input_seq_len, output_seq_len):
    """
    Create and save a seq2seq dataset for traffic time series prediction.

    Parameters:
    - file_path: Path to the CSV file containing the traffic data.
    - month: The month (1-6) for which to filter the data.
    - target: The target variable for prediction ('Duration' or 'Observations').
    - input_seq_len: The length of the input sequences.
    - output_seq_len: The length of the output sequences (labels).

    The function reads the specified CSV file, filters data for the given month, constructs
    input and output sequences using sliding window method, splits the data into training
    and testing sets, and saves these datasets into a single .npz file with a name that
    reflects the function parameters.
    """

    # Read the CSV file
    df = pd.read_csv(file_path + file_name)

    # Convert CaptureDate to datetime format and filter data for the specified month
    df['CaptureDate'] = pd.to_datetime(df['CaptureDate'])
    df = df[df['CaptureDate'].dt.month == month]

    # Select the target column based on the prediction target
    target_column = df[target].values

    # Initialize lists to hold input and output sequences
    input_sequences = []
    output_sequences = []

    # Build input and output sequences using sliding window technique
    total_len = input_seq_len + output_seq_len
    for i in range(len(target_column) - total_len + 1):
        input_seq = target_column[i:i + input_seq_len]
        output_seq = target_column[i + input_seq_len:i + total_len]
        input_sequences.append(input_seq)
        output_sequences.append(output_seq)

    # Convert lists to numpy arrays and reshape for seq2seq modeling
    input_sequences = np.array(input_sequences).reshape(-1, input_seq_len, 1)
    output_sequences = np.array(output_sequences).reshape(-1, output_seq_len, 1)

    # Split the dataset into training and testing sets
    test_size = int(len(input_sequences) * 0.2)
    train_inputs = input_sequences[:-test_size]
    test_inputs = input_sequences[-test_size:]
    train_labels = output_sequences[:-test_size]
    test_labels = output_sequences[-test_size:]

    # Standardize the datasets
    scaler_inputs = StandardScaler()
    scaler_labels = StandardScaler()

    # Fit and transform the training data
    train_inputs = scaler_inputs.fit_transform(train_inputs.reshape(-1, input_seq_len)).reshape(-1, input_seq_len, 1)
    train_labels = scaler_labels.fit_transform(train_labels.reshape(-1, output_seq_len))
    # Transform the testing inputs
    test_inputs = scaler_inputs.transform(test_inputs.reshape(-1, input_seq_len)).reshape(-1, input_seq_len, 1)

    means = test_labels.mean(axis=0)
    stds = test_labels.std(axis=0)
    # Generate a file name based on function parameters

    output_name = f'traffic_data_M{month}_{target}_IN{input_seq_len}_OUT{output_seq_len}.npz'
    # Save the datasets into a single .npz file
    np.savez(file_path + output_name, train_inputs=train_inputs, test_inputs=test_inputs,
             train_labels=train_labels, test_labels=test_labels, means=means, stds=stds)
    return print(f"Data saved as {output_name}")


# Example usage of the function
create_and_save_seq2seq_dataset("data/", "raw_data.csv", 1, "Observations", 12, 6)