
import numpy as np
from paint import Painting, Stroke


class Batch:

    def __init__(self, size=1000):
        self.actions, self.outputs = generate(size, verbose=False)
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


def generate(n=1000, verbose=True):
    actions = []
    outputs = []
    for i in range(n):
        p = Painting(64, 64)
        stroke = Stroke.random()
        p.stroke(stroke)

        actions.append(stroke.actions())
        outputs.append([p.norm_canvas()])

        if i % 100 == 0 and verbose:
            print("{:.2f}%".format(i/n*100))
    return actions, outputs


def generate_from_painter(actions, colors):
    p = Painting(64, 64)

    for act, c in zip(actions, colors):
        pos = np.array([
            [act[0].item(), act[1].item()],
            [act[2].item(), act[3].item()],
            [act[4].item(), act[5].item()]
        ])
        pres = np.array(
            [act[5].item(), act[6].item()]
        )
        stroke = Stroke(pos, np.ones(3) * c.item(), pres)
        p.stroke(stroke)

    canvas = p.canvas / 255.0
    canvas = np.moveaxis(canvas, 2, 0)
    return canvas
