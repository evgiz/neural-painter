
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