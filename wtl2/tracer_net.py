import torch.nn as nn
import torch.nn.functional as F

class TracerNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(4, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(16),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
        )

        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(576, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 2)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.regressor(x)
        x = F.normalize(x, dim=1)
        return x