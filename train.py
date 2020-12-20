
import torch
import numpy as np
from paint import Painting, Stroke
from neural_painter import NeuralPaintStroke
import torch.optim as optim
import torch.nn as nn
import torchvision
from data import Batch


def create_stroke_samples(n=1000):
    actions = []
    outputs = []
    print(f"Generating {n} samples...")
    for i in range(n):
        p = Painting(256, 256)
        pos = np.random.rand(3, 2)
        pres = np.random.rand(2) / 10.0
        stroke = Stroke(pos, np.zeros(3), pres)
        p.stroke(stroke)
        actions.append(np.concatenate([pos.reshape(-1), pres]))

        canvas = p.canvas / 255.0
        canvas = np.moveaxis(canvas, 2, 0)
        outputs.append([canvas[0]])
        print(f"{i}/{n}")
    return actions, outputs


def train_stroke(model, epoch_size, refresh, batch_size=32, epochs=1, save=1, name="stroke_model", draw=1):
    if torch.cuda.is_available():
        print("Running on CUDA")
        model.cuda()

    s_optim = optim.Adam(model.parameters(), lr=1e-2)

    print("Generating initial dataset...")
    batch = Batch(epoch_size)

    for i in range(epochs):
        tot_loss = 0

        if refresh > -1 and i % refresh == 0 and i > 0:
            print("Generating new dataset...")
            batch = Batch(epoch_size)

        while batch.has_next():
            s_optim.zero_grad()

            x, y = batch.next_batch(batch_size)
            x = torch.tensor(x, dtype=torch.float)
            y = torch.tensor(y, dtype=torch.float)

            if torch.cuda.is_available():
                x = x.cuda()
                y = y.cuda()

            p = model.forward(x)

            loss = (p - y).pow(2).mean()
            tot_loss += loss.item()
            loss.backward()
            s_optim.step()

        print("Epoch", i, "loss", tot_loss)

        if i % draw == 0:
            torchvision.utils.save_image(p, "out/{:05d}_p.png".format(i))
            torchvision.utils.save_image(y, "out/{:05d}_y.png".format(i))
        if i % save == 0:
            torch.save(model.state_dict(), "{}_{:05d}".format(name, i))

    torch.save(model.state_dict(), "{}_{:05d}_done".format(name, epochs))
