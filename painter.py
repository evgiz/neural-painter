
import cv2
import sys
import torch
import torchvision
from neural_brush import NeuralPaintStroke
import numpy as np
import os

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


class Painting:

    def __init__(self, path):
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.target = torch.tensor([img / 255.0], dtype=torch.float, device=device)
        self.target = self.target.permute(0, 3, 1, 2)

        self.height = self.target.shape[2]
        self.width = self.target.shape[3]

        self.canvas = torch.ones_like(self.target, dtype=torch.float, device=device)
        self.simulated_canvas = torch.ones_like(self.canvas, dtype=torch.float, device=device)
        self.stroke_modulator = torch.zeros((1, self.height, self.width), dtype=torch.float, device=device)

        self.layers = []
        self.action_history = []

    def blend(self, stroke, color):
        result = self.canvas.clone().detach().to(device)
        
        for s, c in zip(stroke, color):
            color_action = c.view(-1, 3, 1, 1)
            color_action = color_action.repeat(1, 1, self.height, self.width)
            col = s * color_action
            result = col + (1 - s) * result

        return result.to(device)

    def add_layer(self, strokes, iterations, chunks=1, min_size=0.0, max_size=1.0, repeat=1):
        for _ in range(repeat):
            self.layers.append({
                "strokes": strokes,
                "iterations": iterations,
                "chunks": chunks,
                "min_size": min_size,
                "max_size": max_size
            })

    def update(self, canvas, strokes):
        # Update painted canvas
        self.canvas.data = canvas.data
        # Update stroke modulator
        for s in strokes:
            self.stroke_modulator.data += s

    def add_history(self, stroke, pos, scale, color):
        self.action_history.append({
            "stroke": stroke.detach().cpu().numpy(),
            "position": pos.detach().cpu().numpy(),
            "scale": scale.detach().cpu().numpy(),
            "color": color.detach().cpu().numpy()
        })
        np.save("painting/strokes", np.array(self.action_history))

    def get_priority_positions(self, count):
        # (input, k, dim=None, largest=True, sorted=True, *, out=None)
        modulator = self.stroke_modulator.view(-1)
        indices = torch.topk(modulator, count, 0, False, False).indices
        indices = [[[i / self.width], [i % self.width]] for i in indices]
        positions = torch.tensor(indices, dtype=torch.float, device=device)
        positions[:, 0, :] /= self.height
        positions[:, 1, :] /= self.width
        positions = positions * 2 - 1
        return -positions

    def get_position(self, layer, chunk):
        n = int(layer["chunks"])
        x = int(chunk) % n
        y = int(chunk) // n
        return torch.tensor([[float(y) / float(n)], [float(x) / float(n)]], dtype=torch.float, device=device) * 2 - 1

    def get_chunk_size(self, layer):
        n = float(layer["chunks"])
        w = (self.width / n) / self.width
        h = (self.height / n) / self.height
        return torch.tensor([[h], [w]], dtype=torch.float, device=device)


def paint(target):

    # Prepare target painting
    painting = Painting(target)
    painting.add_layer(16, 1, chunks=8, min_size=0.2, max_size=1, repeat=1)

    # Stroke model setuasd
    action_size = 6
    stroke_model = NeuralPaintStroke(action_size).to(device)
    stroke_model.load_state_dict(torch.load("goodmodel/bezier256", map_location=device))
    pos_identity = torch.tensor([[0, 1], [1, 0]], device=device, dtype=torch.float)

    loss_function = torch.nn.MSELoss(reduction='sum')

    print("Loaded model. Starting training...")

    # Training
    for i, layer in enumerate(painting.layers):

        print(f"Drawing layer {i} using {layer['strokes']} strokes")
        print(f"\t{layer['iterations']} iterations, size {layer['min_size']} - {layer['max_size']}")

        for chunk in range(layer["chunks"] * layer["chunks"]):

            # Trainable parameters
            n_strokes = layer["strokes"]
            stroke_params = torch.rand([n_strokes, action_size], requires_grad=True, dtype=torch.float, device=device)
            col_params = torch.rand([n_strokes, 3], requires_grad=True, dtype=torch.float, device=device)
            pos_params = torch.rand([n_strokes, 2, 1], requires_grad=True, dtype=torch.float, device=device)
            scale_params = torch.rand(n_strokes, requires_grad=True, dtype=torch.float, device=device)

            chunk_position = painting.get_position(layer, chunk)
            chunk_size = painting.get_chunk_size(layer)
            chunk_position += chunk_size / 2

            initial_stroke_pos = chunk_position.repeat(n_strokes, 1, 1)

            # Initialize values
            stroke_params.data = stroke_params.data * 6 - 3
            pos_params.data = torch.rand_like(pos_params) * 6 - 3
            scale_params.data = scale_params.data * 6 - 3
            col_params.data = col_params.data * 6 - 3

            # Optimizer
            optimizer = torch.optim.Adam([col_params, pos_params, scale_params, stroke_params], 0.1)

            for iter in range(layer["iterations"]):

                col_params.data = col_params.clamp(0, 1).data

                # Create translation matrix
                thetas = []
                for pos, scale in zip(pos_params, scale_params):
                    s = torch.sigmoid(scale)
                    scl = s * layer["max_size"] + (1 - s) * layer["min_size"]
                    scl /= 10
                    relative_pos = chunk_position + torch.tanh(pos) * chunk_size
                    scaled_identity = pos_identity / scl
                    scaled_position = relative_pos / scl
                    theta = torch.hstack((scaled_identity, scaled_position))
                    thetas.append(theta)

                theta = torch.stack(thetas)

                # Prediction
                p_stroke = stroke_model.forward(torch.sigmoid(stroke_params))
                pos_grid = torch.nn.functional.affine_grid(theta, (n_strokes, 1, painting.height, painting.width), align_corners=True)
                p_canvas = torch.nn.functional.grid_sample(p_stroke, pos_grid)
                p_blend = painting.blend(p_canvas, col_params)

                # Optimize
                optimizer.zero_grad()
                loss = loss_function(painting.target, p_blend)
                loss.backward()
                optimizer.step()

                if iter % 25 == 0:
                    print(f"\t\tLayer {i} chunk {chunk} iter {iter} loss = {loss.item()}")

            painting.update(p_blend, p_canvas)
            painting.add_history(stroke_params, pos_params, scale_params, col_params)

            # stroke_scale -= 0.1
            # stroke_scale = max(0.25, stroke_scale)

            if chunk % 1 == 0:
                torchvision.utils.save_image(painting.canvas, "painting/layer{:05d}-chk{:05d}.png".format(i, chunk))

    torchvision.utils.save_image(painting.canvas, "painting/done.png")


if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("Please provide the path to a target image.")
        exit(-1)

    if not os.path.exists("painting"):
        os.mkdir("painting")
    if not os.path.exists("painting/out"):
        os.mkdir("painting/out")
    if not os.path.exists("painting/out"):
        os.mkdir("painting/out")

    target = sys.argv[1]
    print(f"Painting target image '{target}'...")
    paint(target)