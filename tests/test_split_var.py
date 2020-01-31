import pickle
import numpy as np

from split_functions import variance_split

p_x = pickle.load(open("tests/p_x0.pkl", "rb"))
p_y = pickle.load(open("tests/p_y0.pkl", "rb"))

results = [(variance_split(p_x[:, i], p_y), i) for i in range(p_x.shape[1])]

assert (
        min(
            [
                np.var(p_y[:i]) / i + np.var(p_y[i:]) / (p_y.size - i)
                for i in range(p_y.size)
                if i > 0
            ]
        )
        == 1.098721280759848
)
assert results == [((-1, 0), 0), ((-1, 0), 1), ((0.74, 0.05316685548190847), 2), ((-1, 0), 3), ((-1, 0), 4),
                   ((8.725, 0.05316685548190825), 5), ((-1, 0), 6), ((1.3459, 0.11030799831056826), 7), ((-1, 0), 8),
                   ((-1, 0), 9), ((-1, 0), 10), ((-1, 0), 11), ((3.01, 0.05316685548190869), 12)]
