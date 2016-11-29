from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

plot = plt.figure()
ax = plot.add_subplot(111, projection='3d')

data = open('data.dat', "r")

line = data.readline()

x = []
y = []
z = []
while line:
    dataPoints = line.split('\t')
    x.append(float(dataPoints[0]))
    y.append(float(dataPoints[1]))
    z.append(float(dataPoints[2].replace('\n', '')))
    line = data.readline()

ax.set_xlim(xmin=0, xmax=max(x))
ax.set_ylim(ymin=0, ymax=max(y))
ax.set_zlim(zmin=0, zmax=max(z))

ax.scatter(x, y, z)
plt.show()
