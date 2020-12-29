
import torch
import numpy as np
from torchvision.utils import save_image
import cv2


# =================================== #
#       Brush Simulator Interface     #
# =================================== #

class BrushSimulator:

    # Single pixel, no parameters
    actions = []

    def __init__(self, size):
        self.size = size

    def action_size(self):
        return len(self.actions)

    def _center(self):
        return np.array((self.size / 2, self.size / 2), dtype=np.float)

    def _action_dict(self, actions):
        return {
            k: actions[i] for i, k in enumerate(self.actions)
        }

    def batch(self, n):
        xs = []
        ys = []
        for _ in range(n):
            x, y = self.random()
            xs.append(x)
            ys.append(y)
        return xs, ys

    def random(self):
        act = np.random.rand(len(self.actions))
        return act, self.generate(act)

    def generate(self, actions):
        assert len(actions) == len(self.actions), "Wrong number of actions provided"
        return self._generate(self._action_dict(actions))

    # Should return canvas with shape (channels, size, size)
    def _generate(self, actions):
        canvas = np.zeros((1, self.size, self.size))
        canvas[0, self.size // 2, self.size // 2] = 1
        return canvas


# =================================== #
#          Straight Brush             #
# =================================== #

class BrushStraight(BrushSimulator):

    actions = [
        "length",
        "angle",
        "thickness",
        "falloff"
    ]

    def __init__(self, size):
        super(BrushStraight, self).__init__(size)

    def _generate(self, actions):
        canvas = np.zeros((self.size, self.size))
        angle = actions["angle"] * np.pi * 2
        length = int(actions["length"] * self.size / 2)

        # Thickness between 1% and 75%
        thickness = actions["thickness"]
        thickness = thickness * 0.5 + (1 - thickness) * 0.01
        thickness = int(thickness * self.size)

        # Position
        delta = np.array([np.sin(angle), np.cos(angle)])
        print("Angle: ", angle, "delta: ", delta)

        start = self._center() - delta * length * 0.5
        stop = self._center() + delta * length * 0.5
        start = start.clip(thickness / 2, self.size - thickness / 2)
        stop = stop.clip(thickness / 2, self.size - thickness / 2)

        cv2.line(canvas, tuple(start.astype(np.int)), tuple(stop.astype(np.int)), (1, 1, 1), thickness=thickness)
        return np.array([canvas])


# =================================== #
#            Bezier Brush             #
# =================================== #

class BrushBezier(BrushSimulator):

    actions = [
        "ctrl_x",
        "ctrl_y",
        "stop_x",
        "stop_y",
        "thickness"
    ]

    def __init__(self, size):
        super(BrushBezier, self).__init__(size)

    def _bezier(self, nodes, t):
        return nodes[0] * t ** 2 + nodes[1] * 2 * t * (1 - t) + nodes[2] * (1 - t) ** 2

    def _generate(self, actions):
        canvas = np.zeros((self.size, self.size))

        # Thickness between 5% and 75%
        thickness = actions["thickness"]
        thickness = thickness * 0.05 + (1 - thickness) * 0.25
        thickness = int(thickness * self.size)

        ctrl = np.array([actions["ctrl_x"], actions["ctrl_y"]]) * self.size
        ctrl = ctrl.clip(thickness / 2, self.size - thickness / 2)

        stop = np.array([actions["stop_x"], actions["stop_y"]]) * self.size
        stop = stop.clip(thickness / 2, self.size - thickness / 2)

        nodes = np.array([self._center(), ctrl, stop])
        evals = np.linspace(0, 1, 100)
        points = [self._bezier(nodes, t) for t in evals]

        for i in range(len(points) - 1):
            falloff = 1 * evals[i]
            thick = int(thickness * ((falloff * 0.5) + 0.5))
            if thick > 0:
                cv2.line(canvas, tuple(points[i].astype(np.int)), tuple(points[i+1].astype(np.int)), (1, 1, 1), thickness=thick)

        return np.array([canvas])


# =================================== #
#            Paint Brush              #
# =================================== #

class BrushPaint(BrushSimulator):

    actions = [
        "ctrl_x",
        "ctrl_y",
        "stop_x",
        "stop_y",
        "thickness"
    ]

    def __init__(self, size):
        super(BrushPaint, self).__init__(size)
        self.template = cv2.imread('brushes/paint_template.png', cv2.IMREAD_GRAYSCALE)

    def _bezier(self, nodes, t):
        return nodes[0] * t ** 2 + nodes[1] * 2 * t * (1 - t) + nodes[2] * (1 - t) ** 2

    def _generate(self, actions):
        canvas = np.zeros((self.size, self.size))

        # Thickness between 5% and 75%
        thickness = actions["thickness"]
        thickness = thickness * 0.05 + (1 - thickness) * 0.25
        thickness = int(thickness * self.size)

        ctrl = np.array([actions["ctrl_x"], actions["ctrl_y"]]) * self.size
        ctrl = ctrl.clip(thickness / 2, self.size - thickness / 2)

        stop = np.array([actions["stop_x"], actions["stop_y"]]) * self.size
        stop = stop.clip(thickness / 2, self.size - thickness / 2)

        nodes = np.array([self._center(), ctrl, stop])
        evals = np.linspace(0, 1, 100)
        points = [self._bezier(nodes, t) for t in evals]

        for i in range(len(points)):
            thick = int(thickness)

            points[i] += np.random.rand(2) * thickness / 2

            sx, ex = int(points[i][1] - thick / 2), int(points[i][1] + thick / 2)
            sy, ey = int(points[i][0] - thick / 2), int(points[i][0] + thick / 2)
            w, h = ex-sx, ey-sy

            if sx >= 0 and sy >= 0 and ex < self.size and ey < self.size and w > 0 and h > 0:
                angle = np.random.rand(1)[0] * 360
                M = cv2.getRotationMatrix2D((self.template.shape[0] // 2, self.template.shape[1] // 2), angle, 1)
                rotated = cv2.warpAffine(self.template, M, self.template.shape)

                template = cv2.resize(rotated, (h, w))
                canvas[sy:ey, sx:ex] += template

        return np.array([canvas]) / np.max(canvas)


class Painting:

    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.canvas = np.zeros((width, height, 3)) * 255

    def stroke(self, stroke, samples=250):
        ts = np.linspace(0.0, 1.0, samples)
        for t in ts:
            p = stroke.eval(t)
            x = int(p[0] * self.width)
            y = int(p[1] * self.height)
            r = stroke.pressure * t # + stroke.pressure * 0.1 * (1 - t) # Radius falloff
            r = int(r * max(self.width, self.height) * 0.05)

            cv2.circle(self.canvas, (x, y), max(0, r), stroke.color * 255.0, -1)

    # Returns normalized first channel of canvas
    def norm_canvas(self):
        canvas = self.canvas / 255.0
        canvas = np.moveaxis(canvas, 2, 0)
        return canvas[0]

    def save(self, name):
        cv2.imwrite(f"{name}.png", self.canvas)


class Stroke:

    def __init__(self, start, stop, color=(0, 0, 0), pressure=0.1):
        self.start = start
        self.stop = stop
        self.color = color
        self.pressure = pressure

    @staticmethod
    def random(color=False):
        return Stroke(
            np.random.rand(2),
            np.random.rand(2),
            np.random.rand(3) if color else np.array([1.0, 1.0, 1.0]),
            np.random.rand(1)[0]
        )

    def actions(self):
        return np.concatenate([self.start, self.stop, [self.pressure]])
        # return np.concatenate([self.nodes.reshape(-1), self.pressure])

    def eval(self, t):
        return self.start * t + (1 - t) * self.stop
        # return self.nodes[0] * t ** 2 + self.nodes[1] * 2 * t * (1 - t) + self.nodes[2] * (1 - t) ** 2


if __name__ == "__main__":
    import os

    if not os.path.exists("brushes"):
        os.mkdir("brushes")

    brushes = {
        "interface": BrushSimulator(128),
        "straight": BrushStraight(128),
        "bezier": BrushBezier(128),
        "paint": BrushPaint(128),
    }

    for name in brushes:
        images = []
        for _ in range(32):
            _, example = brushes[name].random()

            images.append(example)
        result = torch.tensor(images, dtype=torch.float)
        save_image(result, "brushes/" + name + ".png")