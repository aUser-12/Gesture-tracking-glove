import torch
import torch.nn as nn

class FlexGloveTransformer(nn.Module):
    def __init__(self, num_sensors: int, window_size: int, num_classes: int):
        super().__init__()
        self.num_sensors = num_sensors
        self.window_size = window_size

        d_model = 64  # Hidden dimension size for the transformer

        # 1. Project raw sensor values up to d_model size
        self.sensor_projection = nn.Linear(num_sensors, d_model)

        # 2. Learnable positional encodings so the model knows the order of time steps
        self.pos_embedding = nn.Parameter(torch.randn(1, window_size, d_model))

        # 3. Transformer Encoder Layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=4,               # 4 attention heads
            dim_feedforward=128,
            dropout=0.3,
            batch_first=True       # Expects (batch, time, features)
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # 4. Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # x shape arrives as: (batch, window_size, num_sensors) -> (1, 30, 5)

        # Project sensors to hidden dimension -> (1, 30, 64)
        x = self.sensor_projection(x)

        # Add positional data so time sequence matters -> (1, 30, 64)
        x = x + self.pos_embedding[:, :x.size(1), :]

        # Pass through self-attention layers -> (1, 30, 64)
        x = self.transformer_encoder(x)

        # Global Average Pooling across the time dimension to get a single 64-d vector -> (1, 64)
        x = x.mean(dim=1)

        # Map to class logits -> (1, 50)
        return self.classifier(x)
