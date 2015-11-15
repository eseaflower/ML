import numpy as np

class Transform(object):
    def __init__(self, xform=None):
        if xform is None:
            xform = np.eye(3)
        self.transform = xform
    
    def clone(self):
        return Transform(np.copy(self.transform))
    
    def translate(self, x, y):
        self.transform[:2, 2] += np.asarray([x, y])

    def scale(self, xf, yf):
        prepend = np.eye(3)
        prepend[0, 0] = xf
        prepend[1, 1] = yf
        self.transform = np.dot(prepend, self.transform)

    def mirrorY(self):
        self.scale(-1.0, 1.0)

    def apply(self, x, y):
        val = np.asarray([x, y, 1])
        result = np.dot(self.transform, val)
        return result[0], result[1]

    def invert(self):
        self.transform = np.linalg.inv(self.transform)

    def prepend(self, other):
        self.transform = np.dot(other.transform, self.transform)

class Translate(Transform):
    def __init__(self, x, y):
        super().__init__()
        self.translate(x,y)
class Scale(Transform):
    def __init__(self, xf, yf):
        super().__init__()
        self.scale(xf, yf)
