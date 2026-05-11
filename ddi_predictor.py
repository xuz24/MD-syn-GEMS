import torch
import torch.nn as nn

class DDI_Predictor(nn.Module):
    def __init__(self, input_dim=2048, output_dim=2, dropout=0.3):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(32, output_dim)
        )

    def forward(self, x):
        return self.net(x)
        