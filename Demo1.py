import time
import numpy as np
import matplotlib.pyplot as plt
from Voronoi import Voronoi
import matplotlib.tri as tri
from collections import deque


class Node:
    def __init__(self, x, y) -> None:
        self.x = x
        self.y = y
        self.connections = []
        self.stepable_connection = []

    def len(self):
        return np.linalg.norm([self.x, self.y])

    def __add__(self, other):
        return Node(self.x + other.x, self.y + other.y)

    def __sub__(self, other):
        return Node(self.x - other.x, self.y - other.y)

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return Node(self.x * other, self.y * other)
        elif isinstance(other, Node):
            return Node(self.x * other.x, self.y * other.y)
        else:
            raise ValueError("Unsupported operand type for multiplication")

    def __truediv__(self, scalar):
        return Node(self.x / scalar, self.y / scalar)

    def __str__(self):
        return f"({self.x}, {self.y})"

    def move(self, node):
        self.x += node.x
        self.y += node.y

    def __eq__(self, other):
        if isinstance(other, Node):
            return self.x == other.x and self.y == other.y
        return False

    def connect(self, node):
        if node not in self.connections:
            self.connections.append(node)
        if self not in node.connections:
            node.connect(self)

    def disconnect(self, node):
        if node in self.connections:
            self.connections.remove(node)
        if self in node.connections:
            node.disconnect(self)

    def add_stepable_connection(self, node):
        if node not in self.stepable_connection:
            self.stepable_connection.append(node)
        if self not in node.stepable_connection:
            node.add_stepable_connection(self)

    def reset_stepable_connections(self):
        self.stepable_connection = []


class Spot(Node):
    def __init__(self, x, y) -> None:
        super().__init__(x, y)
        self.liberties = 3


MAX_MAGNITUDE = 0.01


def apply_force(node, force):

    force_magnitude = force.len()

    if force_magnitude > MAX_MAGNITUDE:
        balanced_force = (force / force_magnitude) * MAX_MAGNITUDE
        node.move(balanced_force)
    else:
        node.move(force)


def project_node_onto_line_segment(start_node, end_node, node_v):
    # Calculate the direction vector of the line segment
    line_direction = Node(end_node.x - start_node.x, end_node.y - start_node.y)

    # Calculate the vector from start_node to node_v
    to_node_v = Node(node_v.x - start_node.x, node_v.y - start_node.y)

    # Calculate the dot product of to_node_v and line_direction
    dot_product = (to_node_v.x * line_direction.x) + \
        (to_node_v.y * line_direction.y)

    # Calculate the projection of node_v onto the line segment
    projection = Node(
        start_node.x + (dot_product * line_direction.x),
        start_node.y + (dot_product * line_direction.y)
    )

    return projection


def F_attraction(u, v, attract_const, l_opt):
    dist = (u - v).len()
    f = (u - v) * (dist / attract_const)
    if dist <= l_opt:
        f = Node(0, 0)

    return f


def F_repulsion_edge(v, edge, l_opt, edge_repuls_const):
    v_edge = project_node_onto_line_segment(edge[0], edge[1], v)
    if v_edge is None:
        return Node(0, 0)
    dist = (v_edge - v).len()
    if dist >= l_opt:
        return Node(0, 0)
    if dist == 0:
        print("edge distance is 0")
        print(edge[0], edge[1], v)
        return Node(0, 0)
    # f = ( (v - v_edge) / dist ) * ( (edge_repuls_const - dist) ** 2 )
    f = ((v - v_edge) / dist) * ((edge_repuls_const - dist) ** 2)
    # f = (v - v_edge) * ( edge_repuls_const / dist )
    return f


def F_repulsion(u, v, l_opt, repuls_const):
    dist = (u-v).len()
    if dist >= l_opt:
        return Node(0, 0)
    if dist == 0:
        u.move(Node(0.01, 0.01))
        v.move(Node(-0.01, -0.01))
        dist = (u-v).len()
        print("distance is 0")
        print(u, v)
    f = (v - u) * ((repuls_const / dist) ** 2)
    return f


def project_node_onto_line_segment(start_node, end_node, node_v):
    line_direction = np.array(
        [end_node.x - start_node.x, end_node.y - start_node.y])
    to_node_v = np.array([node_v.x - start_node.x, node_v.y - start_node.y])

    # Calculate the dot product
    dot_product = np.dot(to_node_v, line_direction)

    # Calculate the length of the line segment
    line_length_squared = np.dot(line_direction, line_direction)

    # Calculate the projection scalar
    projection_scalar = dot_product / line_length_squared

    # Check if the projection is outside the line segment
    if projection_scalar < 0 or projection_scalar > 1:
        return None

    # Calculate the projection vector
    projection = np.array([start_node.x, start_node.y]) + \
        projection_scalar * line_direction

    return Node(projection[0], projection[1])


def BorderForse(v, edge, l_opt,  bord_repuls_const):
    # if v.x >= 1:
    #     v.x = 0.99
    # if v.x <= 0:
    #     v.x = 0.01
    # if v.y >= 1:
    #     v.y = 0.99
    # if v.y <= 0:
    #     v.y = 0.01

    v_edge = project_node_onto_line_segment(edge[0], edge[1], v)
    dist = (v_edge - v).len()
    if dist == 0:
        return (Node(0.5, 0.5) - v) * bord_repuls_const
    if dist >= l_opt:
        return Node(0, 0)
    else:
        # return ((v - v_edge) / dist ) * bord_repuls_const
        return ((v - v_edge) / dist) * ((bord_repuls_const - dist) ** 2)
    # return f


def Redraw(nodes, spots,  long_edges, Borders, l_opt, width_small_edge):

    attract_const = l_opt / 4
    repuls_const = l_opt
    edge_repuls_const = width_small_edge * 10
    small_edges = []
    for long_edge in long_edges:
        small_edges += long_edge
    t = time.time()
    connect_nodes(small_edges)
    print("Connent nodes time", (time.time() - t))
    all_nodes = nodes + spots
    print("Number of points", len(all_nodes))

    t = time.time()
    for i in range(30):
        # new_nodes = []\

        ForsesToApply = []

        for i in range(len(all_nodes)):
            Forse_Sum = Node(0, 0)
            for j in range(len(all_nodes)):
                if j != i:
                    u, v = all_nodes[j], all_nodes[i]
                    # if (u, v) in small_edges or (v,u) in small_edges:
                    if u in v.connections:
                        F_atr = F_attraction(u, v, attract_const, l_opt)
                        Forse_Sum = Forse_Sum + F_atr
                    else:
                        F_repuls = F_repulsion(u, v, l_opt, repuls_const)
                        Forse_Sum = Forse_Sum + F_repuls

            for edge in small_edges:
                if all_nodes[i] in edge:
                    continue
                # F_repuls_edge = F_repulsion_edge(all_nodes[i], edge,l_opt, edge_repuls_const)
                F_repuls_edge = F_repulsion_edge(
                    all_nodes[i], edge, l_opt, repuls_const)

                Forse_Sum = Forse_Sum + F_repuls_edge

            for b_edge in Borders:
                # F_border = BorderForse(all_nodes[i], b_edge, l_opt, edge_repuls_const * 10)
                F_border = BorderForse(
                    all_nodes[i], b_edge, l_opt, repuls_const * 10)
                Forse_Sum = Forse_Sum + F_border

            ForsesToApply.append((all_nodes[i], Forse_Sum))
            # new_nodes.append(v_new)

        for v, f in ForsesToApply:
            apply_force(v, f)
    print("Average iteration time", (time.time() - t) / 30)


def segment_line(A, B, nodes, l_opt):
    length = (A - B).len()

    if length <= 4 * l_opt / 3:
        return [(A, B)]

    fraction = 0.5

    new_x = A.x + (B.x - A.x) * fraction
    new_y = A.y + (B.y - A.y) * fraction

    divided_node = Node(new_x, new_y)
    nodes.append(divided_node)

    left_half = segment_line(A, divided_node, nodes,  l_opt)
    right_half = segment_line(divided_node, B, nodes,  l_opt)

    return left_half + right_half


def merge_long_edge(long_edge, nodes, l_opt):

    merged_edges = []
    i = 0

    while i < len(long_edge):
        node1, node2 = long_edge[i]

        # Calculate the distance between node1 and node2
        distance = (node2 - node1).len()

        if distance < (2/3) * l_opt:
            # Merge node1 and node2 into a single tuple
            if i + 1 < len(long_edge):
                # Get the second node of the next tuple
                node3 = long_edge[i + 1][1]
                nodes.remove(node2)
                merged_edges.append((node1, node3))
                i += 1  # Skip the next tuple since it's merged
            else:
                prev_node, _ = merged_edges.pop()
                nodes.remove(_)
                merged_edges.append((prev_node, node2))
        else:
            # Distance is greater than or equal to (2/3) * l_opt, keep the original tuple
            merged_edges.append((node1, node2))

        i += 1

    return merged_edges


def connect_nodes(small_edges):
    for small_edge in small_edges:
        small_edge[0].connect(small_edge[1])


def display(nodes, spots, vor, long_edges, border_points, border_edges, fig_num, title, route=[]):
    plt.figure(fig_num)
    plt.title(title)

    drawn = []
    for p in nodes + spots + vor + border_points:
        for sc in p.stepable_connection:
            if [sc, p] not in drawn and [p, sc] not in drawn:
                x_values = [p.x, sc.x]
                y_values = [p.y, sc.y]
                plt.plot(x_values, y_values, marker='o',
                         linestyle='dashed', color="grey", markersize=1, zorder=0)
                drawn.append([sc, p])

    # for border in border_edges:
    #     x_values = [border[0].x, border[1].x]
    #     y_values = [border[0].y, border[1].y]
    #     plt.plot(x_values, y_values, marker='o', linestyle='-', markersize=1, zorder = 0)

    for long_edge in long_edges:
        for edge in long_edge:
            x_values = [edge[0].x, edge[1].x]
            y_values = [edge[0].y, edge[1].y]
            plt.plot(x_values, y_values, marker='o', linestyle='-',
                     color="k", zorder=1, linewidth=1.5)

    for i in range(1, len(route)):
        x_values = [route[i-1].x, route[i].x]
        y_values = [route[i-1].y, route[i].y]
        plt.plot(x_values, y_values, marker='o',
                 linestyle='-', color="blue", zorder=1)

    x_coords = [v.x for v in vor]
    y_coords = [v.y for v in vor]
    plt.scatter(x_coords, y_coords, color='g', marker='o', zorder=1, s=15)

    x_coords = [v.x for v in border_points]
    y_coords = [v.y for v in border_points]
    plt.scatter(x_coords, y_coords, color='y', marker='o', zorder=1, s=15)

    x_coords = [spot.x for spot in spots]
    y_coords = [spot.y for spot in spots]
    plt.scatter(x_coords, y_coords, color='r', marker='o', zorder=2, s=15)


def insert_spot(long_edge, nodes, spots):
    if len(long_edge) != 1:
        middle_point = (len(long_edge) - 1) // 2
        short_edge = long_edge[middle_point]
        node = short_edge[1]
        if node in spots:
            print("Node in spots error")
        spot = Spot(node.x, node.y)
        new_short_edge = (short_edge[0], spot)
        long_edge[middle_point] = new_short_edge
        for con in node.connections:
            spot.connect(con)
            node.disconnect(con)
        spots.append(spot)
        nodes.remove(node)
    else:
        x = (long_edge[0].x + long_edge[1].x) / 2
        y = (long_edge[0].y + long_edge[1].y) / 2
        spot = Spot(x, y)
        spot.connect(long_edge[0])
        spot.connect(long_edge[1])
        spots.append(spot)
        short_edge1 = (long_edge[0], spot)
        short_edge2 = (spot, long_edge[1])
        long_edge[0] = short_edge1
        long_edge.append(short_edge2)


def get_vor_ridges(vor_points, vor_vertices, ridge_points, ridge_vertices):
    finite_segments = []
    infinite_segments = []

    center = vor_points.mean(axis=0)
    ptp_bound = vor_vertices.ptp(axis=0)

    for pointidx, simplex in zip(ridge_points, ridge_vertices):
        simplex = np.asarray(simplex)
        # print(pointidx, simplex)
        if np.all(simplex >= 0):
            finite_segments.append(vor_vertices[simplex])
        else:
            i = simplex[simplex >= 0][0]  # finite end Voronoi vertex

            t = vor_points[pointidx[1]] - vor_points[pointidx[0]]  # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal

            midpoint = vor_points[pointidx].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor_vertices[i] + direction * ptp_bound.max()

            infinite_segments.append([vor_vertices[i], far_point])

    return finite_segments, infinite_segments
    print("Finite:", finite_segments)
    print("Infinite:", infinite_segments)


def adjust_vor_borders(finite, infinite, Borders):
    def find_intersection(line1_start, line1_end, line2_start, line2_end):
        det = np.linalg.det(
            np.array([line1_end - line1_start, line2_start - line2_end]))

        if det == 0:
            # Lines are parallel, no intersection point
            return None

        t = np.linalg.det(
            np.array([line2_start - line1_start, line2_start - line2_end])) / det
        u = np.linalg.det(
            np.array([line1_end - line1_start, line2_start - line1_start])) / det

        if 0 <= t <= 1 and 0 <= u <= 1:
            intersection_point = line1_start + t * (line1_end - line1_start)
            return intersection_point
        else:
            # Intersection point is outside the line segments
            return None
    intersection_points = []
    for ridge in finite + infinite:
        for border in Borders:
            intersect_point = find_intersection(np.array([border[0].x, border[0].y]), np.array([
                                                border[1].x, border[1].y]), ridge[0], ridge[1])
            if intersect_point is not None:
                intersection_points.append(
                    Node(intersect_point[0], intersect_point[1]))
                break

    return intersection_points


def RebalanceVoronoi(voronoi_nodes, long_edges, spots, Borders):
    repuls_const = l_opt
    small_edges = []
    for long_edge in long_edges:
        small_edges += long_edge
    connect_nodes(small_edges)
    all_nodes = voronoi_nodes

    for i in range(30):
        # new_nodes = []\

        ForsesToApply = []

        for i in range(len(all_nodes)):
            Forse_Sum = Node(0, 0)
            for j in range(len(all_nodes)):
                if j != i:
                    u, v = all_nodes[j], all_nodes[i]
                    F_repuls = F_repulsion(u, v, l_opt, repuls_const)
                    Forse_Sum = Forse_Sum + F_repuls

            for s in spots:
                F_repuls = F_repulsion(s, all_nodes[i], l_opt, repuls_const)
                Forse_Sum = Forse_Sum + F_repuls

            for edge in small_edges:
                # if all_nodes[i] in edge:
                #     continue
                # F_repuls_edge = F_repulsion_edge(all_nodes[i], edge,l_opt, edge_repuls_const)
                F_repuls_edge = F_repulsion_edge(
                    all_nodes[i], edge, l_opt, repuls_const)

                Forse_Sum = Forse_Sum + F_repuls_edge

            for b_edge in Borders:
                # F_border = BorderForse(all_nodes[i], b_edge, l_opt, edge_repuls_const * 10)
                F_border = BorderForse(
                    all_nodes[i], b_edge, l_opt, repuls_const * 10)
                Forse_Sum = Forse_Sum + F_border

            ForsesToApply.append((all_nodes[i], Forse_Sum))
            # new_nodes.append(v_new)

        for v, f in ForsesToApply:
            apply_force(v, f)


def merge_close_points(vor_nodes, l_opt):
    merged_points = []
    merged = False

    for point1 in vor_nodes:
        for point2 in merged_points:
            if (point1 - point2).len() < l_opt / 5:
                # Merge points if they are closer than l_opt/2
                merged_point = Node((point1.x + point2.x) / 2,
                                    (point1.y + point2.y) / 2)
                merged_points.remove(point2)  # Remove the old point
                merged_points.append(merged_point)  # Add the merged point
                merged = True
                break

        if not merged:
            # If the point was not merged, add it to the merged list
            merged_points.append(point1)
        merged = False

    return merged_points


def do_lines_intersect(P1, P2, P3, P4):
    # Calculate direction vectors
    dir1 = P2 - P1
    dir2 = P4 - P3

    # Calculate determinant
    det = dir1.x * (-dir2.y) - (-dir2.x) * dir1.y

    if det == 0:
        # Lines are parallel and may or may not intersect
        if (P1.x == P3.x and P1.y == P3.y) or (P2.x == P4.x and P2.y == P4.y):
            return True
        elif (P1.x == P4.x and P1.y == P4.y) or (P2.x == P3.x and P2.y == P3.y):
            return True
        return False

    # Calculate t1 and t2
    t1 = ((P3.x - P1.x) * (-dir2.y) - (-dir2.x) * (P3.y - P1.y)) / det
    t2 = (dir1.x * (P3.y - P1.y) - (P3.x - P1.x) * dir1.y) / det

    # Check if t1 and t2 are within [0, 1]
    if 0 < t1 < 1 and 0 < t2 < 1:
        # Lines intersect within the given segments
        return True
    else:
        # Lines do not intersect within the given segments
        return False


def triangulate(nodes, spots, border_points, long_edges, vor_nodes):
    all_nodes = vor_nodes + border_points + spots
    x = [point.x for point in all_nodes]
    y = [point.y for point in all_nodes]
    triang = tri.Triangulation(x, y)
    for p1, p2 in triang.edges:
        if all_nodes[p1] in all_nodes[p2].stepable_connection:
            continue
        for long_edge in long_edges:
            for edge in long_edge:
                if do_lines_intersect(all_nodes[p1], all_nodes[p2], edge[0], edge[1]):
                    break
            else:
                continue
            break
        else:

            all_nodes[p1].add_stepable_connection(all_nodes[p2])
    # print(triang.edges)
    # plt.triplot(triang, linewidth=1, color='grey', linestyle="dashed")


def remove_bad_voronoi(vor_nodes, long_edges, l_opt):
    for long_edge in long_edges:
        for edge in long_edge:
            copy = vor_nodes.copy()
            for v in copy:
                projected = project_node_onto_line_segment(edge[0], edge[1], v)
                if projected is None:
                    continue
                if (projected - v).len() < l_opt / 3:
                    # print("Point", v, "Edge" , edge[0], edge[1], "projected", projected, "distance", (projected - v).len())
                    # print((projected - v).len(), l_opt/2)
                    vor_nodes.remove(v)


def build_route(start_node, end_node):
    visited = set()
    queue = deque()
    parent_map = {}  # To store parent nodes for constructing the path

    queue.append(start_node)
    visited.add(id(start_node))

    while queue:
        current_node = queue.popleft()
        if current_node == end_node:
            # Reconstruct the path from end_node to start_node
            path = []
            while current_node:
                path.append(current_node)
                current_node = parent_map.get(id(current_node))
            return list(reversed(path))

        for neighbor in current_node.stepable_connection:
            # Check if the neighbor is a Node (not a Spot)
            if neighbor == end_node or (type(neighbor) is Node and id(neighbor) not in visited):
                queue.append(neighbor)
                visited.add(id(neighbor))
                parent_map[id(neighbor)] = current_node

    # If no path is found
    return None


if __name__ == "__main__":
    A = Spot(0.01, 0.01)
    B = Node(0.5, 0.01)
    C = Node(0.01, 0.5)
    D = Spot(0.5, 0.5)

    D2 = Spot(0.51, 0.51)
    E = Spot(0.01, 0.99)
    F = Node(0.5, 0.99)

    G = Node(0.3, 0.95)
    H = Node(0.45, 0.95)
    I = Node(0.45, 0.55)
    J = Spot(0.3, 0.55)

    K = Spot(0.55, 0.01)
    L = Node(0.55, 0.99)
    M = Node(0.99, 0.99)
    N = Node(0.99, 0.01)

    s1 = Node(0, 0)
    s2 = Node(0, 1)
    s3 = Node(1, 1)
    s4 = Node(1, 0)
    Borders = [(s1, s2), (s2, s3), (s3, s4), (s4, s1)]
    border_points = [s1, s2, s3, s4]

    # small_edges = [(A, B), (B,C), (C,D), (D2, F), (E,F), (G,H),(H,I), (I,J), (J,G), (K,L), (L,M), (M,N), (N,K)]
    long_edges = [[(A, B), (B, C), (C, D)],
                  [(D2, F), (F, E)],
                  [(G, H), (H, I), (I, J), (J, G)],
                  [(K, L), (L, M), (M, N), (N, K)]]

    vor_nodes = []
    l_opt = 0.1

    nodes = [B, C, F, G, H, I, L, M, N]
    # nodes = [B, C]
    spots = [A, D, D2, E, J, K, Spot(
        0.35, 0.75), Spot(0.4, 0.75), Spot(0.4, 0.8)]
    # spots = [A, D]
    # display(nodes, spots,vor_nodes, long_edges,Borders, 1, "Initial")
    t = time.time()
    for i in range(len(long_edges)):
        new_long_edge = []
        for small_edge in long_edges[i]:
            edges = segment_line(small_edge[0], small_edge[1], nodes,  l_opt)
            new_long_edge += edges

        long_edges[i] = new_long_edge
    print("Seqmentation time: ", (time.time() - t))

    # display(nodes, spots, vor_nodes, long_edges,border_points, Borders, 2, "Divided")

    t = time.time()
    Redraw(nodes, spots, long_edges, Borders, l_opt, 0.002)
    print("Redrawing time: ", (time.time() - t))

    # display(nodes, spots, vor_nodes, long_edges, border_points, Borders, 3, "Rebalanced")

    t = time.time()
    for i in range(len(long_edges)):
        long_edges[i] = merge_long_edge(long_edges[i], nodes, l_opt)
    print("Merge time", (time.time() - t))

    # display(nodes, spots,vor_nodes,  long_edges, Borders, 4, "merged")

    for long_edge in long_edges:
        insert_spot(long_edge, nodes, spots)

    # display(nodes, spots,vor_nodes,  long_edges, Borders, 5, "inserted")
    arr = []
    for p in nodes + spots:
        arr.append([p.x, p.y])
    np_arr = np.array(arr)
    vor = Voronoi(np_arr)

    for v in vor.vertices:
        vor_nodes.append(Node(v[0], v[1]))

    # display(nodes, spots, vor_nodes, long_edges, Borders, 6, "voronoi vertices")

    # Redraw(vor_nodes, [], [],Borders, l_opt, 0.02)
    finite, infinite = get_vor_ridges(
        np_arr, vor.vertices, vor.ridge_points, vor.ridge_vertices)

    # display(nodes, spots, vor_nodes, long_edges, Borders, 7, "voronoi ridges", finite, infinite)

    additional_vor_points = adjust_vor_borders(finite, infinite, Borders)

    vor_nodes += additional_vor_points
    vor_nodes_copy = vor_nodes.copy()
    for v in vor_nodes_copy:
        if v.x < s1.x or v.y < s1.y or v.x > s3.x or v.y > s3.y:
            vor_nodes.remove(v)
    display(nodes, spots, vor_nodes, long_edges, border_points,
            Borders, 8, "vor vertices + additional")

    # RebalanceVoronoi(vor_nodes, long_edges, spots, Borders)

    # display(nodes, spots, vor_nodes, long_edges, border_points, Borders, 9, "vor vertices rebalanced")

    # vor_nodes = merge_close_points(vor_nodes, l_opt)

    # display(nodes, spots, vor_nodes, long_edges,border_points, Borders, 9, "Merged voronoi  points")
    # remove_bad_voronoi(vor_nodes, long_edges, l_opt)
    # display(nodes, spots, vor_nodes, long_edges,border_points, Borders, 10, "removed bad points")

    triangulate(nodes, spots, border_points, long_edges, vor_nodes)

    display(nodes, spots, vor_nodes, long_edges,
            border_points,  Borders, 11, "triangulated")
    route = build_route(A, J)
    if route is not None:
        display(nodes, spots, vor_nodes, long_edges, border_points,
                Borders, 12, "route AJ", route=route)

    # i = 0
    # for p1 in range(len(spots)):
    #     for p2 in range(p1 + 1, len(spots)):

    #         route = build_route(spots[p1], spots[p2])
    #         if route is not None:
    #             i+=1
    #             display(nodes, spots, vor_nodes, long_edges, border_points,  Borders, 11+i, f"{i} path", route=route)

    plt.show()
