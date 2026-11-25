import torch
import torch.nn as nn

class FlexGloveRNN(nn.Module):
    def __init__(self, num_sensors: int, window_size: int, num_classes: int):
        super().__init__()
        self.num_sensors = num_sensors
        self.window_size = window_size


        self.lstm = nn.LSTM(
            input_size=num_sensors,
            hidden_size=64,
            num_layers=2,
            batch_first=True,
            dropout=0.3
        )


        self.classifier = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):


        lstm_out, (h_n, c_n) = self.lstm(x)


        final_hidden = h_n[-1]


        return self.classifier(final_hidden)
