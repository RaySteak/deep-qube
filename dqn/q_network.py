import torch
import torch.nn as nn
from torchsummary import summary


class QNetwork2DConv(nn.Module):
    """this architecture has a size of 6.01 MB"""

    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()

        self.bn0 = nn.BatchNorm2d(6)
        self.conv1 = nn.Conv2d(6, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(256)
        self.dropout = nn.Dropout(0.1)
        self.fc1 = nn.Linear(256 * 3 * 3, 256)
        self.fc2 = nn.Linear(256, 12)

    def forward(self, x):
        x = self.bn0(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.conv4(x)
        x = self.relu(x)
        x = self.bn1(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class QNetwork3DConv(nn.Module):
    """this architecture has a size of 4.67 MB"""

    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv3d(1, 32, 3, padding=(1, 1, 1))  # B, 32, 3, 3, 48
        self.average_pool1 = nn.AvgPool3d(
            kernel_size=(1, 1, 1), stride=(1, 1, 1)
        )  # B, 32, 3, 3, 24
        self.conv2 = nn.Conv3d(32, 64, 3, padding=1)  # B, 64, 3, 3, 24
        self.average_pool2 = nn.AvgPool3d(
            kernel_size=(1, 1, 1), stride=(1, 1, 1)
        )  # B, 64, 3, 3, 12
        self.conv3 = nn.Conv3d(64, 64, 3, padding=1)  # B, 128, 3, 3, 12
        self.average_pool3 = nn.AvgPool3d(
            kernel_size=(1, 1, 1), stride=(1, 1, 1)
        )  # B, 128, 3, 3, 6
        self.conv4 = nn.Conv3d(64, 64, 3, padding=1)  # B, 256, 3, 3, 6
        self.bn1 = nn.BatchNorm3d(64)
        self.dropout = nn.Dropout(0.1)
        self.fc1 = nn.Linear(64 * 6 * 3 * 3, 256)
        self.fc2 = nn.Linear(256, 12)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.average_pool1(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.average_pool2(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.average_pool3(x)
        x = self.conv4(x)
        x = self.relu(x)
        x = self.bn1(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


if __name__ == "__main__":
    q_net = QNetwork3DConv()
    print(q_net(torch.rand(10, 1, 6, 3, 3)).shape)

    print(summary(q_net, (1, 6, 3, 3)))
    print(q_net)
    q_net = QNetwork2DConv()
    # from torchsummary import summary

    print(summary(q_net, (6, 3, 3)))
