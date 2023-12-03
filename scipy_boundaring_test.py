import numpy as np
import matplotlib.pyplot as plt
from test_cases_bouundary_test import *



def get_boundaries(points, handedness):#rule always turn left or right
    def angle_between_vectors(vec1, vec2):
        cosine_theta = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        angle_rad = np.arccos(np.clip(cosine_theta, -1.0, 1.0))
        return angle_rad
    boundaries = []
    boundary_sets = []
    for p in points:
        current_point = p
        current_direction = np.array([0,1])#looking up
        if handedness == "R":
            current_direction = np.array([0,-1])#looking down
        boundary = []
        prev_point = None
        while True:
            boundary.append(current_point)
            if current_point.connections == []:
                break
            max_angle = -float('inf')
            most_handed_neighbor = None
            
            if len(current_point.connections) == 2 and prev_point is not None:
                for n in current_point.connections:
                    if n != prev_point:
                        most_handed_neighbor = n
                        break
            else:
                for n in current_point.connections:
                    neighbor_position = np.array([n.x, n.y])
                    current_position = np.array([current_point.x, current_point.y])
                    neighbor_vector = neighbor_position - current_position
                    cross_prod = np.cross(current_direction, neighbor_vector)
                    angle = angle_between_vectors(current_direction, neighbor_vector)
                    if cross_prod <= 0:
                        angle *= -1
                
                    # if handedness == "L":
                    #     if cross_prod <= 0:
                    #         angle *= -1
                    # else:
                    #     if cross_prod >=0:
                    #         angle *= -1
                    
                    if angle > max_angle:
                        max_angle = angle
                        most_handed_neighbor = n

            if most_handed_neighbor == p:
                break
            
            prev_point = current_point

            neighbor_position = np.array([most_handed_neighbor.x, most_handed_neighbor.y])
            current_direction = neighbor_position - np.array([current_point.x, current_point.y])

            current_point = most_handed_neighbor

        b_set = set(boundary)


        if b_set not in boundary_sets:
            for b in boundary_sets:
                Subset = True
                for p in b_set:
                    if p not in b:
                        Subset = False
                        break
                if Subset:
                    break
            else:
                boundary_sets.append(b_set)
                boundaries.append(boundary)

    
    return boundaries
            
        
def merge_boundaries(boundaries_L, boundaries_R):
    res = boundaries_L
    b_L_sets = []
    for b_L in boundaries_L:
        b_L_sets.append(set(b_L))
    for b_R in boundaries_R:
        if set(b_R) not in b_L_sets:
            res.append(b_R)
    return res
        
import time
# Example usage:
t = time.time()

points = Test3()
boundaries_L = get_boundaries(points, 'L')
boundaries_R = get_boundaries(points, 'R')
merged = merge_boundaries(boundaries_L, boundaries_R)

print(f"This operation took {time.time()-t} seconds, found {len(merged)} regions in {len(points)} points")
# print("Left hand boundaryies:")
# for boundary in boundaries_L:
#     for p in boundary:
#         print(p.name, end = "")
#     print()
# print("Right hand boundaryies:")
# for boundary in boundaries_R:
#     for p in boundary:
#         print(p.name, end = "")
#     print()
# print("Merged boundaries")
# for boundary in merged:
#     for p in boundary:
#         print(p.name, end = "")
#     print()



x_coords = [point.x for point in points]
y_coords = [point.y for point in points]
plt.scatter(x_coords, y_coords, color='b', marker='o')


# Plot the connections
for p in points:
    for connection in p.connections:
        plt.plot([p.x, connection.x], [p.y, connection.y], 'k-')

plt.show()

