
import torch
import bezier
import numpy as np
from torchvision.utils import save_image
import cv2


class Painting:

    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.canvas = np.ones((width, height, 3)) * 255

    def stroke(self, stroke, samples=250):
        ts = np.linspace(0.0, 1.0, samples)
        points = stroke.curve.evaluate_multi(ts)
        points = np.transpose(points)
        pressure = np.interp(ts, [0, 1], stroke.pressure)

        for p, r in zip(points, pressure):
            x = int(p[0] * self.width)
            y = int(p[1] * self.height)
            r = int(r * max(self.width, self.height))

            cv2.circle(self.canvas, (x, y), r, stroke.color, -1)

            # for xx in range(-r // 2, r // 2):
            #     for yy in range(-r // 2, r // 2):
            #         tx = x+xx
            #         ty = y+yy
            #         if (tx-x)**2 + (ty-y)**2 < (r/2)**2:
            #             if self.width > tx > 0 and self.height > ty > 0:
            #                 self.canvas[0][ty][tx] = stroke.color[0]
            #                 self.canvas[1][ty][tx] = stroke.color[1]
            #                 self.canvas[2][ty][tx] = stroke.color[2]

    def save(self, name):
        cv2.imwrite(f"{name}.png", self.canvas)


class Stroke:

    def __init__(self, nodes, color=(0, 0, 0), pressure=(0.01, 0.02)):
        self.nodes = np.asfortranarray(np.transpose(nodes))
        self.curve = bezier.Curve(self.nodes, degree=2)
        self.color = color * 255
        self.pressure = pressure
