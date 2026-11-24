import torch
import torch.nn as nn

class FlexGloveMLP(nn.Module):
    def __init__(self, num_sensors: int, window_size: int, num_classes: int):
        super().__init__()
        self.num_sensors = num_sensors
        self.window_size = window_size


        input_dim = window_size * num_sensors


        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.network(x)
