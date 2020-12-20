
import torch
import numpy as np
from paint import Painting, Stroke
from neural_painter import NeuralPaintStroke
import torch.optim as optim
import torch.nn as nn
import torchvision
import pickle


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


def train_neural_stroke(stroke, minibatch=10, epochs=1):

    s_optim = optim.Adam(stroke.parameters(), lr=1e-2)
    # actions = torch.tensor(np.load("data/1k_actions.npy")[0:100], dtype=torch.float)
    # outputs = torch.tensor(np.load("data/1k_outputs.npy")[0:100], dtype=torch.float)

    actions, outputs = create_stroke_samples(n=10)
    actions = torch.tensor(actions, dtype=torch.float)
    outputs = torch.tensor(outputs, dtype=torch.float)

    for i in range(epochs):
        tot_loss = []
        for j in range(len(actions) // min(minibatch, len(actions))):
            acts = actions[j * minibatch : (j + 1) * minibatch]
            outs = outputs[j * minibatch : (j + 1) * minibatch]
            preds = stroke.forward(acts)

            loss = (preds - outs).pow(2).mean()
            tot_loss.append(loss.item())

            # Train
            s_optim.zero_grad()
            loss.backward()
            s_optim.step()

        print("Epoch", i, "loss", np.mean(tot_loss))

        torchvision.utils.save_image(preds, "out/{:05d}_gen.png".format(i))
        torchvision.utils.save_image(outs, "out/{:05d}_real.png".format(i))
        torch.save(stroke.state_dict(), "model/stroke_model_{:05d}".format(i))



if __name__ == "__main__":

    # actions, outputs = create_stroke_samples(n=1000)
    # np.save("data/1k_actions", actions)
    # np.save("data/1k_outputs", outputs)

    neural_stroke = NeuralPaintStroke(6 + 2)
    #train_neural_stroke(neural_stroke, epochs=100)


    # Test
    neural_stroke.load_state_dict(torch.load("model/stroke_model_00099"))

    actions, outputs = create_stroke_samples(n=10)
    actions = torch.tensor(actions, dtype=torch.float)
    outputs = torch.tensor(outputs, dtype=torch.float)
    preds = neural_stroke.forward(actions)
    fc1test = neural_stroke.fc1(actions)
    fc1test.view(-1, 16, 16, 16)

    torchvision.utils.save_image(fc1test, "fc1.png")
