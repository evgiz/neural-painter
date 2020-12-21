
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

# TODO does flipping target color ease learning? Mask of one requires activation only for stroke not surround
def generate(n=1000, verbose=True):
    actions = []
    outputs = []
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
        if i % 100 == 0 and verbose:
            print("{:.2f}%".format(i/n*100))
    return actions, outputs

