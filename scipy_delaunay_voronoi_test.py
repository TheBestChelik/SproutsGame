import numpy as np
import matplotlib.pyplot as plt

from scipy.spatial import Delaunay, delaunay_plot_2d
from scipy.spatial import Voronoi, voronoi_plot_2d


points = np.array([[0.4, 0.5], [0.6, 0.4], [0.6, 0.6],[0.4, 0.6], [0.9,0.9]])




vor = Voronoi(points)

_ = voronoi_plot_2d(vor)




print("Voronoi vertices: ", vor.vertices)
print("Voronoi points: ", vor.points)
print("Voronoi regions: ", vor.regions)
print("Voronoi ridge_points: ", vor.ridge_points)
for r_p in vor.ridge_points:
    print("Ridge point line: ", vor.points[r_p[0]], vor.points[r_p[1]])
print("Voronoi vertices: ", vor.vertices)
for r_v in vor.ridge_vertices:
    print("Ridge vertices line: ", vor.vertices[r_v[0]], vor.vertices[r_v[1]])
print("Voronoi regions", vor.regions)
print("Voronoi point region: ", vor.point_region)
print("Voronoi furthest site: ", vor.furthest_site)

finite_segments = []
infinite_segments = []

center = vor.points.mean(axis=0)
ptp_bound = vor.points.ptp(axis=0)

print(center)


for pointidx, simplex in zip(vor.ridge_points, vor.ridge_vertices):
    simplex = np.asarray(simplex)
    #print(pointidx, simplex)
    if np.all(simplex >= 0):
            finite_segments.append(vor.vertices[simplex])
    else:
            i = simplex[simplex >= 0][0]  # finite end Voronoi vertex

            t = vor.points[pointidx[1]] - vor.points[pointidx[0]]  # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal

            midpoint = vor.points[pointidx].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            if (vor.furthest_site):
                direction = -direction
            far_point = vor.vertices[i] + direction * ptp_bound.max()

            infinite_segments.append([vor.vertices[i], far_point])
print(finite_segments)
print(infinite_segments)




# def delaunay_edges(points):
#     # Compute the Delaunay triangulation
#     tri = Delaunay(points)
    
#     # Extract edges from the Delaunay triangulation
#     edges = set()

#     for simplex in tri.simplices:
#         # Generate all pairs of vertices for each simplex (triangle)
#         for i in range(3):
#             for j in range(i + 1, 3):
#                 edge = tuple(sorted([simplex[i], simplex[j]]))
#                 edges.add(edge)
    
#     # Convert the set of edges to a list
#     edges = list(edges)
    
#     return edges

# edges = delaunay_edges(points)
# print(edges)

# _ = delaunay_plot_2d(tri)


plt.show()
