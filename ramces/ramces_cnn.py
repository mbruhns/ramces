import torch.nn as nn


class SimpleCNN(nn.Module):
    def __init__(self, image_dim):
        super(SimpleCNN, self).__init__()

        # First convolutional layer
        self.conv1 = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Second convolutional layer
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Third convolutional layer
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Flattened size after conv layers
        self.flattened_size = 128 * (image_dim[0] // 8) * (image_dim[1] // 8)

        # Dropout and fully connected layer for binary classification
        self.dropout = nn.Dropout(p=0.3)
        self.fc1 = nn.Linear(self.flattened_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(-1, self.flattened_size)
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.sigmoid(x)
        return x
