
import numpy as np
from paint import Painting, Stroke


class Batch:

    def __init__(self, n=1000, image_size=32):
        self.actions, self.outputs = generate(n, image_size, verbose=False)
        self._next = 0

    def reset(self):
        self._next = 0

    def has_next(self):
        return self._next < len(self.actions)

    def next_batch(self, size=32):
        if not self.has_next():
            return [], []

        x = self.actions[self._next:min(len(self.actions), self._next + size)]
        y = self.outputs[self._next:min(len(self.outputs), self._next + size)]
        self._next += size
        return x, y


def generate(n=1000, size=32, verbose=True):
    actions = []
    outputs = []
    for i in range(n):
        p = Painting(size, size)
        stroke = Stroke.random()
        actions.append(stroke.actions())
        p.stroke(stroke)
        outputs.append([p.norm_canvas()])

        if i % 100 == 0 and verbose:
            print("{:.2f}%".format(i/n*100))
    return actions, outputs


def generate_from_painter(actions, colors):
    p = Painting(32, 32)

    for act, c in zip(actions, colors):
        stroke = Stroke(
            np.array([act[0].item(), act[1].item()]),
            np.array([act[2].item(), act[3].item()]),
            np.ones(3) * c.item(),
            np.array([act[4].item()])
        )
        p.stroke(stroke)

    canvas = p.canvas / 255.0
    canvas = np.moveaxis(canvas, 2, 0)
    return canvas
