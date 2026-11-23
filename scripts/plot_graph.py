import matplotlib.pyplot as plt
import csv
from matplotlib import colors as mcolors
import numpy as np
from itertools import cycle
import math

colors = cycle(["b", "g", "r", "c", "m", "y", "k"])

folder_name = input("Dataset name")
iteration = int(input("Number of iterations (-1 for final -2 for optimal)"))
file_name = "path"
if iteration == -1:
    file_name = file_name
elif iteration == -2:
    file_name = file_name + "_optimal"
else:
    file_name = file_name + str(iteration)
file_name = file_name + ".txt"

with open("outputs/" + folder_name + "/coordinates.csv") as coordinates_file:
    coordinates = [
        (float(x), float(y)) for x, y in csv.reader(coordinates_file, delimiter=",")
    ]

paths = list()

with open("outputs/" + folder_name + "/" + file_name, "r") as path:
    for line in path.readlines():
        temp = [int(x) for x in line.rstrip().split(",")]
        paths.extend(temp)

x = [x[0] for x in coordinates]
y = [y[1] for y in coordinates]

distance = 0.0
for i in range(0, len(paths) - 1):
    source = [x[paths[i]], y[paths[i]]]
    dest = [x[paths[i + 1]], y[paths[i + 1]]]
    distance += math.dist(source, dest)
print(distance)

for i in range(0, 32):
    plt.plot(
        x[i * 20 : i * 20 + 20],
        y[i * 20 : i * 20 + 20],
        color=next(colors),
        marker="X",
        linestyle="None",
    )

plt.annotate("0", (x[0], y[0]))
plt.xlabel("Distance = " + str(distance))
for i in range(0, len(paths) - 1):
    plt.plot([x[paths[i]], x[paths[i + 1]]], [y[paths[i]], y[paths[i + 1]]], "k-")
# plt.xticks(np.arange(0, 41, 10))
# plt.yticks(np.arange(0, 81, 10))
plt.grid()
plt.show()
