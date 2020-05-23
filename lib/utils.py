import numpy as np

def SMA(data: np.array, degree: int):
    data = np.append([data[0]] * (degree - 1), data)
    smoothed = np.array([
        np.roll(data, i) for i in range(degree)
    ]).T
    return np.sum(smoothed, axis=1)[degree - 1:] / degree
