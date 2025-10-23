import torch
import torch.nn as nn


class YOLOCustom(nn.Module):
    def __init__(self, num_classes=58, input_channels=3):
        super(YOLOCustom, self).__init__()
        self.num_classes = num_classes

        # Backbone
        self.backbone = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )

        # Neck
        self.neck = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )

        # Head
        self.head = nn.Conv2d(256, (num_classes + 5) * 3, kernel_size=1)  # 3 anchors

    def forward(self, x):
        x = self.backbone(x)
        x = self.neck(x)
        x = self.head(x)
        return x


def build_yolo(num_classes):
    return YOLOCustom(num_classes=num_classes)
