import torch.nn as nn

class TaxiDriverClassifier(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(TaxiDriverClassifier, self).__init__()

        hidden_dim = 256
        num_layers = 2

        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.5)  # Increased dropout
        self.batch_norm = nn.BatchNorm1d(hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(0.5)  # Added dropout before FC layer

        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = x.to(next(self.parameters()).device)
        lstm_out, _ = self.lstm(x)
        x = lstm_out[:, -1, :]
        x = self.batch_norm(x)
        x = self.layer_norm(x)
        x = self.dropout(x)  # Apply dropout before classification
        x = self.fc(x)
        return x
