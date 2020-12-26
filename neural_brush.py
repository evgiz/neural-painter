
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import brush_simulator

class NeuralPaintStroke(nn.Module):

    def __init__(self, action_size):
        super(NeuralPaintStroke, self).__init__()

        self.dim = 4
        self.chn = [
            64, 32, 16, 8, 4, 2, 1
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
        self.bn5 = nn.BatchNorm2d(self.chn[4])
        self.conv5 = nn.ConvTranspose2d(self.chn[4], self.chn[5], 4, stride=2, padding=1)
        self.bn6 = nn.BatchNorm2d(self.chn[5])
        self.conv6 = nn.ConvTranspose2d(self.chn[5], self.chn[6], 4, stride=2, padding=1)

        # self.up1 = nn.Upsample(scale_factor=(2, 2), mode="bilinear")
        # self.up2 = nn.Upsample(scale_factor=(2, 2), mode="bilinear")

    def forward(self, x):

        x = self.fc1(x)
        x = x.view(-1, self.chn[0], self.dim, self.dim)
        x = F.relu(self.bn1(x))
        x = F.relu(self.bn2(self.conv1(x)))
        x = F.relu(self.bn3(self.conv2(x)))
        x = F.relu(self.bn4(self.conv3(x)))
        x = F.relu(self.bn5(self.conv4(x)))
        x = F.relu(self.bn6(self.conv5(x)))
        # x = F.relu(self.bn4(self.conv3(x)))
        x = torch.sigmoid(self.conv6(x))
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


if __name__ == "__main__":

    import os
    if not os.path.exists("brush_train"):
        os.mkdir("brush_train")
    if not os.path.exists("brush_train/checkpoint"):
        os.mkdir("brush_train/checkpoint")

    # Hyper parameters
    n_epochs = 1024
    checkpoint_every = 32
    output_every = 8

    minibatch_count = 1
    minibatch_size = 32

    # Brush sim and model
    brush = brush_simulator.BrushBezier(256)
    model = NeuralPaintStroke(brush.action_size())

    if torch.cuda.is_available():
        print("Running on CUDA")
        model.cuda()

    loss_f = nn.MSELoss()
    s_optim = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(n_epochs):
        tot_loss = 0

        for _ in range(minibatch_count):
            s_optim.zero_grad()

            x, y = brush.batch(minibatch_size)
            x = torch.tensor(x, dtype=torch.float)
            y = torch.tensor(y, dtype=torch.float)

            if torch.cuda.is_available():
                x = x.cuda()
                y = y.cuda()

            p = model.forward(x)
            loss = loss_f(y, p)
            tot_loss += loss.item()
            loss.backward()
            s_optim.step()

        print("Epoch", epoch, "loss", tot_loss)

        if i % output_every == 0:
            x_test = torch.rand((32, brush.action_size()), dtype=torch.float, device=device)
            p_test = model.forward(x_test)
            torchvision.utils.save_image(p_test, "brush_train/{:05d}_y.png".format(epoch))
        if i % checkpoint_every == 0:
            torch.save(model.state_dict(), "brush_train/checkpoint/epoch_{:05d}".format(name, i))

    torch.save(model.state_dict(), "brush_train/checkpoint/epoch_{:05d}_done".format(name, n_epochs))
