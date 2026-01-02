import torch.nn as nn

class CornerNet(nn.Module):
    def __init__(self, in_channels=1024, out_channels=4):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)  # [B, 256, 1, 1]
        self.fc = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(128, out_channels)
        )

    def forward(self, x):
        x = self.conv_block(x)
        x = self.pool(x).view(x.size(0), -1)
        out = self.fc(x)  
        return out
