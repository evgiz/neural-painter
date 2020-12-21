
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class NeuralPaintStroke(nn.Module):

    def __init__(self, action_size):
        super(NeuralPaintStroke, self).__init__()

        self.dim = 16
        self.chn = [
            32, 16, 12, 8, 1
        ]

        self.fc1 = nn.Linear(action_size, self.dim * self.dim * self.chn[0])
        self.bn1 = nn.BatchNorm2d(self.chn[0])
        self.conv1 = nn.ConvTranspose2d(self.chn[0], self.chn[1], 4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(self.chn[1])
        self.conv2 = nn.ConvTranspose2d(self.chn[1], self.chn[2], 4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(self.chn[2])
        self.conv3 = nn.ConvTranspose2d(self.chn[2], self.chn[3], 4, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(self.chn[3])
        self.conv4 = nn.ConvTranspose2d(self.chn[3], self.chn[4], 4, stride=2, padding=1)


    def forward(self, x):

        x = self.fc1(x)
        x = x.view(-1, self.chn[0], self.dim, self.dim)
        x = F.relu(self.bn1(x))
        x = F.relu(self.bn2(self.conv1(x)))
        x = F.relu(self.bn3(self.conv2(x)))
        x = F.relu(self.bn4(self.conv3(x)))
        x = torch.tanh(self.conv4(x))
        return x


class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()

        self.dim = 16
        self.chn = [
            1, 4, 8, 12, 16
        ]

        self.conv1 = nn.Conv2d(self.chn[0], self.chn[1], 4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(self.chn[1], self.chn[2], 4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(self.chn[2], self.chn[3], 4, stride=2, padding=1)
        self.conv4 = nn.Conv2d(self.chn[3], self.chn[4], 4, stride=2, padding=1)
        self.fc1 = nn.Linear(self.dim * self.dim * self.chn[-1], 1)

    def forward(self, x):

        x = F.relu(self.conv1(x))
        print(x.shape)
        x = F.relu(self.conv2(x))
        print(x.shape)
        x = F.relu(self.conv3(x))
        print(x.shape)
        x = F.relu(self.conv4(x))
        print(x.shape)
        x = x.view(-1)
        print(x.shape)
        x = self.fc1(x)
        print(x.shape)

        return x