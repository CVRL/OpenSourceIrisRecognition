import torch
import torch.nn as nn


class RegressionHead(nn.Module):

    def __init__(self, in_channels=1024, out_channels=8, dropout=0.3, use_sigmoid=False):
        super().__init__()
        self.out_channels = out_channels
        self.use_sigmoid = use_sigmoid

        self.reduce = nn.Sequential(
            nn.Conv2d(in_channels, 512, kernel_size=1),
            nn.BatchNorm2d(512),
            nn.GELU(),
        )
        self.refine = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1, groups=8),
            nn.BatchNorm2d(512),
            nn.GELU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
        )
        self.refine_act = nn.GELU()
        self.pool = nn.AdaptiveAvgPool2d(1)

        layers = [
            nn.Linear(512, 256), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(256, 64), nn.GELU(), nn.Dropout(dropout * 0.5),
            nn.Linear(64, out_channels),
        ]
        if use_sigmoid:
            layers.append(nn.Sigmoid())
        self.head = nn.Sequential(*layers)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
   
        final_linear = [m for m in self.head.modules() if isinstance(m, nn.Linear)][-1]
        nn.init.normal_(final_linear.weight, std=1e-3)

    def forward(self, x):
        x = self.reduce(x)
        x = self.refine_act(self.refine(x) + x)
        x = self.pool(x).flatten(1)
        return self.head(x)
