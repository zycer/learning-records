import matplotlib.pyplot as plt
from scipy import spatial
from operator import itemgetter
import itertools
import numpy as np
import random

line_list = [[[0, 5], [3, 0.8]], [[4.5, 1.2], [9, 4.5]], [[11, 5.6], [4.8, 10.5]]]
# line_list = [[[0, 1], [1, 0]], [[5, 4], [10, 4]]]
# point = [[1, 2.5]]
point = [[0, 0]]
line_ix = tuple(itertools.chain.from_iterable(
    [itertools.repeat(i, x) for i, x in enumerate(list(map(len, line_list)))]
))
line_list = np.concatenate(line_list)
tree = spatial.cKDTree(line_list)

dist, idx = tree.query(point, k=1)

idx = itemgetter(*idx)(line_ix)

print(line_ix)
print(tree.data)

print(idx, dist)

