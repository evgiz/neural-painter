
import cv2
import sys
import torch
import torchvision
from neural_painter import NeuralPaintStroke

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


class Painting:

    def __init__(self, path):
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.target = torch.tensor([img / 255.0], dtype=torch.float, device=device)
        self.target = self.target.permute(0, 3, 1, 2)
        self.canvas = torch.ones_like(self.target, dtype=torch.float, device=device) * 0.5
        self.height = self.target.shape[2]
        self.width = self.target.shape[3]

    def blend(self, stroke, color):
        result = self.canvas.clone().detach().to(device)

        for s, c in zip(stroke, color):
            color_action = c.view(-1, 3, 1, 1)
            color_action = color_action.repeat(1, 1, self.height, self.width)
            col = s * color_action
            result = col + (1 - s) * result
        return result.to(device)

    def update(self, canvas):
        self.canvas.data = canvas.data


def paint(target):

    # Hyperparameters
    n_epochs = 2048
    n_stroke_iterations = 250
    n_simultaneous_strokes = 16
    stroke_size = 32
    stroke_scale = 1

    # Prepare painting
    painting = Painting(target)

    # Stroke model setup
    stroke_model = NeuralPaintStroke(5).to(device)
    stroke_model.load_state_dict(torch.load("goodmodel/clean32"))
    stroke_params = torch.zeros([n_simultaneous_strokes, 5], requires_grad=True, dtype=torch.float, device=device)

    # Painting parameters
    col_params = torch.zeros([n_simultaneous_strokes, 3], requires_grad=True, dtype=torch.float, device=device)
    pos_params = torch.zeros([n_simultaneous_strokes, 2, 1], requires_grad=True, dtype=torch.float, device=device)
    scale_params = torch.ones(1, requires_grad=True, dtype=torch.float, device=device)
    pos_identity = torch.tensor([[[0, 1], [1, 0]] for _ in range(n_simultaneous_strokes)], device=device)

    identity = torch.tensor([[[0, 1, 0], [1, 0, 0]] for _ in range(n_simultaneous_strokes)], device=device, dtype=torch.float)

    loss_function = torch.nn.MSELoss()

    print("Loaded model. Starting training...")

    # Training
    for epoch in range(n_epochs):

        # Initialize parameters
        stroke_params.data = torch.rand_like(stroke_params).data * 6 - 3
        pos_params.data = torch.rand_like(pos_params).data * 2 - 1
        scale_params.data = torch.rand_like(scale_params) * 6 - 3
        col_params.data = torch.rand_like(col_params).data * 6 - 3

        # Optimizer
        optimizer = torch.optim.Adam([stroke_params, col_params], 1e-2)

        for iter in range(n_stroke_iterations):

            pos_params.data = pos_params.clip(-1, 1).data
            
            # Create translation matrix
            #scaled_identity = pos_identity / stroke_scale
            #scaled_position = pos_params / stroke_scale
            #theta = torch.hstack((scaled_identity, scaled_position))
            theta = identity.to(device) #theta.view(n_simultaneous_strokes, 2, 3).to(device)

            # Prediction
            p_stroke = stroke_model.forward(torch.sigmoid(stroke_params))
            pos_grid = torch.nn.functional.affine_grid(theta, (n_simultaneous_strokes, 1, painting.height, painting.width), align_corners=True)
            p_canvas = torch.nn.functional.grid_sample(p_stroke, pos_grid)
            p_blend = painting.blend(p_canvas, torch.sigmoid(col_params))

            # Optimize
            loss = loss_function(painting.target, p_blend)
            loss.backward()
            optimizer.step()

            # pos_params.data = pos_params.clip(-1, 1).data

        print("Stroke {:05d} loss: {} (scale {})".format(epoch, loss.item(), stroke_scale))
        painting.update(p_blend)

        stroke_scale -= 0.1
        stroke_scale = max(0.25, stroke_scale)

        if epoch % 1 == 0:
            torchvision.utils.save_image(painting.canvas, "painting/{:05d}.png".format(epoch))

    torchvision.utils.save_image(painting.canvas, "painting/done.png")

if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("Please provide the path to a target image.")
        exit(-1)

    target = sys.argv[1]
    print(f"Painting target image '{target}'...")
    paint(target)