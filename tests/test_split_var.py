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
assert all([i[0][0] == -1 for i in results])
