
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class NeuralPaintStroke(nn.Module):

    def __init__(self, action_size):
        super(NeuralPaintStroke, self).__init__()

        self.dim = 4
        self.chn = [
            64, 32, 16, 1
        ]

        self.fc1 = nn.Linear(action_size, self.dim * self.dim * self.chn[0])
        self.bn1 = nn.BatchNorm2d(self.chn[0])
        self.conv1 = nn.ConvTranspose2d(self.chn[0], self.chn[1], 4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(self.chn[1])
        self.conv2 = nn.ConvTranspose2d(self.chn[1], self.chn[2], 4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(self.chn[2])
        # self.conv3 = nn.ConvTranspose2d(self.chn[2], self.chn[3], 4, stride=2, padding=1)
        # self.bn4 = nn.BatchNorm2d(self.chn[3])
        self.conv4 = nn.ConvTranspose2d(self.chn[2], self.chn[3], 4, stride=2, padding=1)

        # self.up1 = nn.Upsample(scale_factor=(2, 2), mode="bilinear")
        # self.up2 = nn.Upsample(scale_factor=(2, 2), mode="bilinear")

    def forward(self, x):

        x = self.fc1(x)
        x = x.view(-1, self.chn[0], self.dim, self.dim)
        x = F.relu(self.bn1(x))
        x = F.relu(self.bn2(self.conv1(x)))
        x = F.relu(self.bn3(self.conv2(x)))
        # x = F.relu(self.bn4(self.conv3(x)))
        x = torch.sigmoid(self.conv4(x))
        return x


class NeuralUpscale(nn.Module):

    def __init__(self):
        super(NeuralUpscale, self).__init__()

        self.output_size = 256

        self.conv1 = nn.ConvTranspose2d(1, 1, 4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(1)
        self.conv2 = nn.ConvTranspose2d(1, 1, 4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(1)
        self.conv3 = nn.ConvTranspose2d(1, 1, 4, stride=2, padding=1)

    def forward(self, x):
        x = self.bn1(x)
        x = F.relu(self.conv1(x))
        x = self.bn2(x)
        x = F.relu(self.conv2(x))
        x = torch.sigmoid(self.conv3(x))
        return x

