
import torch
import numpy as np
from torchvision.utils import save_image
import cv2


class Painting:

    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.canvas = np.zeros((width, height, 3)) * 255

    def stroke(self, stroke, samples=250):
        ts = np.linspace(0.0, 1.0, samples)

        # Stroke pressure is translated to between 0.1 and 1.0
        p = stroke.pressure * 1.0 + (1 - stroke.pressure) * 0.1
        pressure = np.interp(ts, [0, 1], p / 10.0)

        for t, r in zip(ts, pressure):
            p = stroke.eval(t)
            x = int(p[0] * self.width)
            y = int(p[1] * self.height)
            r = int(r * max(self.width, self.height))

            cv2.circle(self.canvas, (x, y), max(0, r), stroke.color * 255.0, -1)

    # Returns normalized first channel of canvas
    def norm_canvas(self):
        canvas = self.canvas / 255.0
        canvas = np.moveaxis(canvas, 2, 0)
        return canvas[0]

    def save(self, name):
        cv2.imwrite(f"{name}.png", self.canvas)


class Stroke:

    def __init__(self, nodes, color=(0, 0, 0), pressure=(0.01, 0.02)):
        self.nodes = nodes
        self.color = color
        self.pressure = pressure

    @staticmethod
    def random(color=False):
        return Stroke(
            np.random.rand(3, 2),
            np.random.rand(3) if color else np.array([1.0, 1.0, 1.0]),
            np.random.rand(2)
        )

    def actions(self):
        return np.concatenate([self.nodes.reshape(-1), self.pressure])

    def eval(self, t):
        return self.nodes[0] * t ** 2 + self.nodes[1] * 2 * t * (1 - t) + self.nodes[2] * (1 - t) ** 2