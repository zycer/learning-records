import matplotlib.pyplot as plt
from scipy import spatial
from operator import itemgetter
import itertools
import numpy as np

lines = [[[0, 5], [3, 0.8]], [[4.5, 1.2], [9, 4.5]], [[11, 5.6], [4.8, 10.5]]]
points = [[0, 0], [7, 5]]

line_ix = tuple(itertools.chain.from_iterable(
    [itertools.repeat(i, x) for i, x in enumerate(list(map(len, lines)))]
))
line_list = np.concatenate(lines)
tree = spatial.cKDTree(line_list)

dist, idx = tree.query(points, k=1)

idx = itemgetter(*idx)(line_ix)

print("树结构~~~~~~~~~~")
print(tree.data)
print()

print("结果：")
print(idx, dist)
x_list = []
y_list = []
for line in lines:
    for num, value in enumerate(zip(line[0], line[1])):
        if num % 2 == 0:
            x_list.append(value)
        else:
            y_list.append(value)

point_x = []
point_y = []
for point in points:
    point_x.append(point[0])
    point_y.append(point[1])


for i in range(len(x_list)):
    plt.plot(x_list[i], y_list[i])

plt.scatter(point_x, point_y)
plt.show()
