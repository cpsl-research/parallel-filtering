import numpy as np


class Landmark:
    def __init__(self, x: np.ndarray):
        self.x = x

    @property
    def position(self):
        return self.x