from matplotlib import pyplot as plt


class LinearDecayParameter:
    """parameter that decay linearly"""

    def __init__(self, x_0, y_0, x_1, y_1, *, min_value=None):
        """specify two points(x_0, y_0), (x_1, y_1) to calculate `y=kx+b`"""
        self.x_0 = x_0
        self.x_1 = x_1
        self.min_value = min_value  # used to clip value
        self.k = (y_1 - y_0) / (x_1 - x_0)
        self.b = y_1 - self.k * x_1
        if min_value is not None:
            self.min_value_x = (min_value - self.b) / self.k

    def __call__(self, x):
        value = self.k * x + self.b
        if self.min_value is None:
            return value
        return max(value, self.min_value)

    def plot(self, x_0=None, x_1=None):
        if x_0 is None: x_0 = self.x_0
        if x_1 is None: x_1 = self.x_1

        if x_0 > x_1:
            x_0, x_1 = x_1, x_0  # ensure x_0 is the small one

        x = x_0
        x_values, y_values = [], []
        while x < x_1:
            y = self(x)
            x_values.append(x)
            y_values.append(y)
            x += 0.1
        fig, ax = plt.subplots()
        ax.plot(x_values, y_values)
        plt.show()


if __name__ == '__main__':
    p = LinearDecayParameter(0, 0.3, 500, 0, min_value=0)
    p.plot(0, 1000)
