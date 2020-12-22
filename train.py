
import torch, cv2
import numpy as np
from paint import Painting, Stroke
from neural_painter import NeuralPaintStroke
import torch.optim as optim
import torch.nn as nn
import torchvision
from data import Batch, generate_from_painter, generate
import torch.functional as F


def train_stroke(model, epoch_size, refresh, batch_size=100, epochs=1, learning_rate=None, save=1, name="stroke_model", draw=1):
    if torch.cuda.is_available():
        print("Running on CUDA")
        model.cuda()

    loss_f = nn.MSELoss()
    s_optim = optim.Adam(model.parameters(), lr=learning_rate or 1e-3)

    print("Generating initial dataset...")
    batch = Batch(epoch_size)

    for i in range(epochs):
        tot_loss = 0

        if refresh > -1 and i % refresh == 0 and i > 0:
            print("Generating new dataset...")
            batch = Batch(epoch_size)
        else:
            batch.reset()

        while batch.has_next():
            s_optim.zero_grad()

            x, y = batch.next_batch(batch_size)
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

        print("Epoch", i, "loss", tot_loss / batch_size)

        if i % draw == 0:
            x, y = generate(16)
            x = torch.tensor(x, dtype=torch.float)
            y = torch.tensor(y, dtype=torch.float)
            if torch.cuda.is_available():
                x = x.cuda()
                y = y.cuda()
            pp = model.forward(x)
            print("validation loss: ", loss_f(y, pp).item() / 16)
            torchvision.utils.save_image(pp, "out/{:05d}_p.png".format(i))
            torchvision.utils.save_image(y, "out/{:05d}_y.png".format(i))
        if i % save == 0:
            torch.save(model.state_dict(), "{}_{:05d}".format(name, i))

    torch.save(model.state_dict(), "{}_{:05d}_done".format(name, epochs))


def forward_paint(canvas, model, actions, colors):

    result = canvas.clone().detach()
    strokes = model.forward(actions)

    for stroke, color in zip(strokes, colors):
        color_action = color.view(-1, 3, 1, 1)
        color_action = color_action.repeat(1, 1, 64, 64)

        col = stroke * color_action
        result = col + (1 - stroke) * result

    return result


def train_painting(target, model, epochs=1000, strokes=10, simultaneous=1, background=None, learning_rate=None, verbose=True):

    actions = torch.rand(simultaneous, 5, requires_grad=True)
    colors = torch.rand(simultaneous, 3, requires_grad=True)
    target_mean = background if background is not None else target.mean().item()

    canvas = torch.ones(3, 64, 64) * target_mean

    paint_optimizer = optim.Adam([
        actions,
        colors
    ], lr=learning_rate or 1e-3)

    priority_test = 3

    for i in range(strokes):

        for _ in range(epochs):
            paint_optimizer.zero_grad()
            pred = forward_paint(canvas, model, torch.sigmoid(actions), torch.sigmoid(colors))

            loss = (target - pred).pow(2).mean()
            loss.backward(retain_graph=True)
            paint_optimizer.step()

        print(f"Stroke {i} reconstruction loss", loss.item())

        canvas.data = pred.data
        actions.data = torch.rand(simultaneous, 5).data
        colors.data = torch.rand(simultaneous, 3).data

        pred = forward_paint(canvas, model, torch.sigmoid(actions), torch.sigmoid(colors))
        best_loss = (target - pred).pow(2).mean().item()

        # Priority test (find best of n new strokes to commit to)
        for _ in range(priority_test):
            a_test = torch.rand(simultaneous, 5)
            c_test = torch.rand(simultaneous, 3)
            pp = forward_paint(canvas, model, torch.sigmoid(a_test), torch.sigmoid(c_test))
            p_loss = (target - pp).pow(2).mean().item()
            if p_loss < best_loss:
                actions.data = a_test.data
                colors.data = c_test.data
                best_loss = p_loss

        # Reset optimizer parameters
        paint_optimizer = optim.Adam([
            actions,
            colors
        ], lr=learning_rate or 1e-3)

        if verbose and i % 1 == 0:
            torchvision.utils.save_image(canvas, "out_paint/{:05d}.png".format(i))

    if verbose:
        torchvision.utils.save_image(canvas, "out_paint/done.png")

    return canvas


def paint_chunked(target_name, model, chunks=16, strokes=4):

    if torch.cuda.is_available():
        print("Running on CUDA")
        model.cuda()

    # Preprocess chunks
    print("Generating target chunks...")

    target = cv2.imread(target_name, cv2.IMREAD_COLOR)
    target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

    chunk_size = int(target.shape[0] / chunks)
    chunks_y = int(target.shape[1] / chunk_size + 0.5)

    targets = [[] for _ in range(chunks)]
    outputs = []

    # Crop chunks
    for x in range(chunks):
        for y in range(chunks_y):
            xend = target.shape[0] - 1 if x == chunks-1 else x * chunk_size + chunk_size
            yend = target.shape[1] - 1 if y == chunks_y - 1 else y * chunk_size + chunk_size

            chunk = target[y*chunk_size:yend, x*chunk_size:xend]
            chunk = cv2.resize(chunk, (64, 64), interpolation=cv2.INTER_AREA)

            chunk = torch.tensor([chunk / 255.0], dtype=torch.float)
            print(x, y, chunk.shape)
            chunk = chunk.permute(0, 3, 1, 2)

            targets[x].append(chunk)

    # Render chunks
    for x in range(chunks):
        row = []
        for y in range(chunks_y):
            print(f"Rendering ({x}, {y})...")

            if torch.cuda.is_available():
                targets[x][y].cuda()

            out = train_painting(
                targets[x][y],
                model,
                epochs=150,
                strokes=6,
                simultaneous=3,
                background=1,
                learning_rate=1e-1,
                verbose=False
            )
            row.append(out)

        stitched = torch.cat(row, dim=2)
        print(stitched.shape)
        outputs.append(stitched)

    stitched = torch.cat(outputs, dim=3)
    print(stitched.shape)
    torchvision.utils.save_image(stitched, "chunk/result.png")



