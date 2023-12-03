import numpy as np
import matplotlib.pyplot as plt

from scipy.spatial import Delaunay, delaunay_plot_2d
from scipy.spatial import Voronoi, voronoi_plot_2d

def _adjust_bounds(ax, points):
    margin = 0.1 * points.ptp(axis=0)
    xy_min = points.min(axis=0) - margin
    xy_max = points.max(axis=0) + margin
    ax.set_xlim(xy_min[0], xy_max[0])
    ax.set_ylim(xy_min[1], xy_max[1])

# rng = np.random.default_rng()
# print('rng=', rng)

# points = rng.random((30, 2))
# points = np.array([[0, 0], [0, 1.1], [1, 0], [1, 1]])
# points = np.array([[0.1, 0.1], [0.1, 1.1], [0.4, 0.8], [0.6, 0.6], [0.8, 0.8], [1.1, 1.1], [1.1, 0.1]])
points = np.array([[0.1, 0.1], [0.1, 1.1], [0.6, 0.6], [0.8, 0.8], [1.1, 1.1], [1.1, 0.1]])
# print('points=', points)

print("points", points)
plt.plot(points[:,0], points[:,1], 'x')

# print("tri.simplices=", tri.simplices)
# plt.plot(tri.simplices[:,0], tri.simplices[:,1], 'o')
# print("tri.simplices=", tri.vertices)

vor = Voronoi(points)

print("points=", points)
print("vor.vertices=", vor.vertices)

# print("vor.ridge_points=", vor.ridge_points)
# plt.plot(vor.ridge_points[:,0], vor.ridge_points[:,1], 'o')

# print("vor.ridge_vertices=", vor.ridge_vertices)
# plt.plot(vor.ridge_points[:,0], vor.ridge_points[:,1], 'o')

# print("vor.vertices=", vor.vertices)
# plt.plot(vor.vertices[:,0], vor.vertices[:,1], 'o')
# plt.triplot()


result = np.concatenate([points, vor.vertices])
print("result=", result)

# _ = delaunay_plot_2d(tri)
# _, voronoi_finite_segments, voronoi_infinite_segments = voronoi_plot_2d(vor)

# print("voronoi_finite_segments[0]=", voronoi_finite_segments[0])
# print("voronoi_infinite_segments[0]=", voronoi_infinite_segments[0])

tri = Delaunay(points)
_ = delaunay_plot_2d(tri)

plt.show()