
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision

from brush_simulator import BrushPaint
from neural_brush import NeuralPaintStroke


class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()

        self.dim = 4
        self.chn = [
            1, 4, 8, 16, 32, 64, 64
        ]

        self.bn1 = nn.BatchNorm2d(self.chn[0])
        self.conv1 = nn.Conv2d(self.chn[0], self.chn[1], 4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(self.chn[1])
        self.conv2 = nn.Conv2d(self.chn[1], self.chn[2], 4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(self.chn[2])
        self.conv3 = nn.Conv2d(self.chn[2], self.chn[3], 4, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(self.chn[3])
        self.conv4 = nn.Conv2d(self.chn[3], self.chn[4], 4, stride=2, padding=1)
        self.bn5 = nn.BatchNorm2d(self.chn[4])
        self.conv5 = nn.Conv2d(self.chn[4], self.chn[5], 4, stride=2, padding=1)

        self.fc1 = nn.Linear(4096, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = x.flatten(start_dim=1)
        x = self.fc1(x)
        x = torch.sigmoid(x)
        return x


if __name__ == "__main__":

    import os

    if not os.path.exists("brush_train"):
        os.mkdir("brush_train")
    if not os.path.exists("brush_train/checkpoint_dsc"):
        os.mkdir("brush_train/checkpoint_dsc")

    n_epochs = 100

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    discriminator = Discriminator().to(device)
    generator = NeuralPaintStroke(5).to(device)
    generator.load_state_dict(torch.load("goodmodel/paint256a", map_location=device))

    brush = BrushPaint(256)

    loss_f = nn.BCELoss()
    s_optim = optim.Adam(discriminator.parameters(), lr=1e-3)

    for epoch in range(n_epochs):

        acts, real = brush.batch(64)
        real = torch.tensor(real, dtype=torch.float, device=device)
        acts = torch.tensor(acts, dtype=torch.float, device=device)
        fake = generator.forward(acts)

        labels = torch.tensor([[1] if i < 64 else [0] for i in range(128)], dtype=torch.float, device=device)
        data = torch.tensor(torch.cat([real, fake]), dtype=torch.float, device=device)

        pred = discriminator.forward(data)

        s_optim.zero_grad()
        loss = loss_f(pred, labels)
        loss.backward()
        s_optim.step()

        print(f"Epoch {epoch}, loss = {loss}")

        if epoch % 10 == 0:
            torch.save(discriminator.state_dict(), "brush_train/checkpoint_dsc/dsc_{:05d}".format(epoch))