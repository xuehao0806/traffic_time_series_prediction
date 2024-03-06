import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class RNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_seq_len):
        super(RNNModel, self).__init__()
        self.rnn = nn.RNN(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_seq_len)

    def forward(self, x):
        # x shape: (batch_size, seq_length, input_dim)
        x, _ = self.rnn(x)
        # Select the last time step's output
        x = x[:, -1, :]
        x = self.fc(F.relu(x))
        return x


class GRUModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_seq_len):
        super(GRUModel, self).__init__()
        self.rnn = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_seq_len)

    def forward(self, x):
        # x shape: (batch_size, seq_length, input_dim)
        x, _ = self.rnn(x)
        # Select the last time step's output
        x = x[:, -1, :]
        x = self.fc(F.relu(x))
        return x


class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_seq_len):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_seq_len)  # Output dim remains 1 since we're predicting one step at a time

    def forward(self, x):
        x, (hn, cn) = self.lstm(x)
        # Assuming x is of shape (batch_size, seq_length, hidden_dim)
        # We take the output of the last time step and repeat it output_seq_len times
        x = x[:, -1, :]
        # Now pass each time step output through the fully connected layer
        x = self.fc(F.relu(x))
        return x


class TransformerModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_seq_len, nhead=2, num_encoder_layers=1):
        super(TransformerModel, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=nhead, dim_feedforward=hidden_dim)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_encoder_layers)
        self.fc = nn.Linear(input_dim, output_seq_len)

    def forward(self, x):
        # x shape: (seq_length, batch_size, input_dim)
        x = x.permute(1, 0, 2)  # Transformer expects (seq_length, batch_size, input_dim)
        x = self.transformer_encoder(x)
        # Select the last time step's output
        x = x[-1, :, :]
        x = self.fc(x)
        return x


class SimpleTransformerModel(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_encoder_layers, output_seq_len):
        super(SimpleTransformerModel, self).__init__()
        self.d_model = d_model
        self.input_embedding = nn.Linear(input_dim, d_model)  # 将输入映射到较高维度
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layers = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer=encoder_layers, num_layers=num_encoder_layers)
        self.fc_out = nn.Linear(d_model, output_seq_len)  # 假设输出序列长度固定

    def forward(self, src):
        src = self.input_embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        src = src.permute(1, 0, 2)  # 转换为Transformer期望的格式
        output = self.transformer_encoder(src)
        output = output.permute(1, 0, 2)  # 转换回(batch_size, seq_len, features)
        output = self.fc_out(output[:, -1, :])  # 只取序列的最后一步
        return output

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)