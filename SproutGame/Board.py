import matplotlib.pyplot as plt
import numpy as np
import time
from typing import List, Set, Tuple

from SproutGame.primitives import Vertex, Vector, Spot, Path
from SproutGame.resources.constants import Color, LineStyle, CANVAS_SIZE, VERTEX_SIZE
from SproutGame.modules.forces import calculate_resultant_forces, apply_force_to_vertex
from SproutGame.modules.geometry import Voronoi, Delaunay
from SproutGame.resources.GUI_Constants import GAME_OVER_TITLE



class Board:
    def __init__(self,
                 time_based_game,
                 vertices,
                 players,
                 temporary_vertices=set(),
                 border_points=[
                     Vertex(0.05, 0.05, Color.GREEN.value),
                     Vertex(0.05, 0.95, Color.GREEN.value),
                     Vertex(0.95, 0.95, Color.GREEN.value),
                     Vertex(0.95, 0.05, Color.GREEN.value)],
                 optimum_length=0.001) -> None:

        self.__vertices = vertices  # spots + path_vertices

        # all stepable vertices (voronoi + border)
        self.__temporary_vertices = temporary_vertices

        self.__pathes = {}

        self.__temporary_edges = set()  # triengulation, stepable edges

        self.border_points = border_points

        self.__optimum_length = optimum_length

        self.current_path = []
        self.players = players
        if time_based_game:
            self.player_timers = [60 for p in players]
        self.current_player = 0

        self.game_overed = False

    @property
    def vertices(self) -> Set:
        return self.__vertices

    @property
    def temporary_edges(self) -> Set:
        return self.__temporary_edges

    @property
    def optimum_length(self) -> float:
        return self.__optimum_length

    @property
    def temporary_verties(self) -> set:
        return self.__temporary_vertices

    @property
    def pathes(self):
        return self.__pathes

    @property
    def current_player_color(self):
        return self.players[self.current_player][1]

    def update_current_player_timer(self):
        self.player_timers[self.current_player] -= 0.1

    @property
    def current_player_timer(self):
        return self.player_timers[self.current_player]

    @property
    def current_player_name(self):
        return self.players[self.current_player][0]

    def end_of_time_based_game(self):
        self.game_overed = True

    def step(self, label=None):
        path1, path2 = None, None
        if len(self.current_path) != 0:
            path1, path2 = self.insert_path(
                self.current_path, self.current_player_color)
        self.current_player += 1
        self.current_player = self.current_player % len(self.players)
        self.current_path = []
        self.__temporary_edges = set()
        self.__temporary_vertices = set()
        self.segment_paths(path1, path2)
        self.rebalance()
        self.adjust_to_new_boundaries(self.vertices, 0.1)
        self.merge_edges()
        self.create_temp_vertices()
        self.adjust_to_new_boundaries(self.vertices, 0.1)
        self.add_boundary_points()
        self.triangulate()
        end_of_game = self.check_end_of_game()
        if end_of_game:
            if label:
                label.config(text=GAME_OVER_TITLE)
            self.game_overed = True
            return
        self.__optimum_length = 0.03
        if len(self.vertices) < 140:
            self.__optimum_length = 0.1 - 0.0005 * len(self.vertices)
        if label:
            label.config(text=f"{self.current_player_name} can walk")

    def check_end_of_game(self):
        def build_graph(edges):
            graph = {}
            for start, end in edges:
                graph.setdefault(start, []).append(end)
                graph.setdefault(end, []).append(start)  # Add the reverse edge
            return graph

        def has_path(graph, start, end, black_list, visited=None):
            if visited is None:
                visited = set()
            visited.add(start)
            if start == end:
                return True
            for neighbor in graph.get(start, []):
                # Check if the neighbor is an uppercase letter
                if neighbor == end:
                    return True
                if (neighbor not in black_list) and neighbor not in visited:
                    # if neighbor not in visited:
                    if has_path(graph, neighbor, end, black_list, visited):
                        return True
            return False

        def has_cycle(graph, vertex, black_list):
            for neighbor1 in graph.get(vertex, []):
                if neighbor1 in black_list:
                    continue
                for neighbor2 in graph.get(vertex, []):
                    if neighbor2 in black_list:
                        continue
                    if neighbor1 == neighbor2:
                        continue
                    new_black_list = black_list.union({vertex})
                    if has_path(graph, neighbor1, neighbor2, new_black_list):
                        return True
            return False

        graph = build_graph(self.__temporary_edges)
        black_list = set()
        for path in self.pathes.values():
            for edge in path.edges:
                black_list.add(edge[0])
                black_list.add(edge[1])

        for v1 in self.vertices:
            if type(v1) is not Spot:
                continue
            if v1.liberties < 1:
                continue
            for v2 in self.vertices:
                if type(v2) is not Spot:
                    continue
                if v2.liberties < 1:
                    continue
                if v1 == v2:
                    continue
                if has_path(graph, v1, v2, black_list):
                    return False
            if v1.liberties > 1 and has_cycle(graph, v1, black_list):
                return False

        return True

    # returning (redraw_frame, end_of_path)
    def spot_in_path(self, spot) -> Tuple[bool, bool]:

        if len(self.current_path) == 0:
            if spot.liberties < 1:
                return (False, False)

            spot.minus_liberty()
            self.current_path.append(spot)

            return (True, False)

        if len(self.current_path) == 1:
            if spot in self.current_path:
                return (self.check_move_can_cancel(spot), False)

        prev_vertex = self.current_path[-1]
        if (prev_vertex, spot) not in self.__temporary_edges and (spot, prev_vertex) not in self.__temporary_edges:
            return False, False
        
        prev_prev_vertex = self.current_path[-2]
        if prev_prev_vertex == spot:
            return False, False        

        if spot.liberties < 1:
            return (False, False)

        spot.minus_liberty()
        self.current_path.append(spot)
        return (True, True)

    # return redraw_frame
    def update_current_path(self, vertex) -> bool:
        if len(self.current_path) == 0:
            return False
        if vertex in self.current_path:
            # cancel move
            return self.check_move_can_cancel(vertex)
        prev_vertex = self.current_path[-1]
        if (prev_vertex, vertex) not in self.__temporary_edges and (vertex, prev_vertex) not in self.__temporary_edges:
            return False

        if vertex not in self.__temporary_vertices:
            return False

        self.current_path.append(vertex)
        return True

    # return canceled
    def check_move_can_cancel(self, vertex) -> bool:
        if len(self.current_path) == 0:
            return False
        last_vertex = self.current_path[-1]
        if last_vertex != vertex:
            return False
        if type(last_vertex) is Spot:
            last_vertex.add_liberty()
        self.current_path.pop()
        return True

    def find_closest_vertex(self, pos: Vector) -> Vertex:
        minDist = float('inf')
        closest_veretx = None
        for p in list(self.vertices) + list(self.__temporary_vertices):
            dist = (pos - Vector(p.x, p.y)).magnitude
            if dist < minDist:
                closest_veretx = p
                minDist = dist
        return closest_veretx

    def get_visible_verties(self):
        visible_vertices = set()
        for edge in self.__temporary_edges:
            if edge[0] in self.current_path and edge[1] not in self.current_path:
                visible_vertices.add(edge[1])
            elif edge[0] not in self.current_path and edge[1] in self.current_path:
                visible_vertices.add(edge[0])
        return visible_vertices

    def insert_spot(self, path: List[Vertex]) -> Tuple[List[Tuple[Vertex, Vertex]], List[Tuple[Vertex, Vertex]]]:
        if len(path) != 2:
            middle_point = (len(path) - 1) // 2

            vertex = path[middle_point]
            new_spot = Spot(vertex.x, vertex.y, Color.RED.value, liberties=1)
            path[middle_point] = new_spot

            self.vertices.add(new_spot)
            self.vertices.remove(vertex)

            first_half = path[:middle_point]
            first_half.append(new_spot)
            second_half = path[middle_point:]

            return first_half, second_half
        else:
            x = (path[0].x + path[1].x) / 2
            y = (path[0].y + path[1].y) / 2
            new_spot = Spot(x, y, Color.RED.value, liberties=1)
            self.__vertices.add(new_spot)
            first_half = [path[0], new_spot]
            second_half = [new_spot, path[1]]
            return first_half, second_half

    def array_to_edges(self, array: List[Vertex]) -> List[Tuple[Vertex, Vertex]]:
        edges = []
        for i in range(len(array) - 1):
            edges.append((array[i], array[i+1]))

        return edges

    def insert_path(self, path: List[Vertex], color) -> Tuple[Path, Path]:
        A = path[0]
        B = path[-1]
        if A not in self.vertices or B not in self.vertices:
            raise RuntimeError("Spots that form path are not in vertices set")

        for v in path:
            if type(v) is not Spot:
                if v not in self.temporary_verties:
                    raise RuntimeError(
                        "Edge vertex not in temporary vertices set")

                v.color = color
                self.__temporary_vertices.remove(v)
                self.__vertices.add(v)

        first_arr, second_arr = self.insert_spot(path)
        first = self.array_to_edges(first_arr)
        second = self.array_to_edges(second_arr)
        path1 = Path(color, first)
        path2 = Path(color, second)
        self.__pathes[(A, first_arr[1])] = path1
        self.__pathes[(second_arr[0], B)] = path2
        return (path1, path2)

    def segment_paths(self, newPath1, newPath2):
        for path in self.__pathes.values():
            l_opt = self.optimum_length
            path.segment_edges(self.vertices, l_opt)

    def adjust_to_new_boundaries(self, vertex, boundary: float):
        x_min = 0
        x_max = 1
        y_min = 0
        y_max = 1
        if len(vertex) > 1:
            x_values = [v.x for v in vertex]
            y_values = [v.y for v in vertex]
            x_min = min(x_values)
            x_max = max(x_values)
            y_min = min(y_values)
            y_max = max(y_values)

        for v in vertex:
            x = (v.x - x_min) / (x_max - x_min)
            y = (v.y - y_min) / (y_max - y_min)
            x = x * (1 - 2*boundary) + boundary
            y = y * (1 - 2*boundary) + boundary
            vector = Vector(x, y) - Vector(v.x, v.y)
            v.translate(vector)

    def rebalance(self):
        boundaries = {(self.border_points[0], self.border_points[1]),
                      (self.border_points[1], self.border_points[2]),
                      (self.border_points[2], self.border_points[3]),
                      (self.border_points[3], self.border_points[0])}

        edges = []
        for path in self.pathes.values():
            edges += path.edges
        all_ver = self.vertices.union(self.__temporary_vertices)
        t = time.time()
        no_forces_applied = True
        for _ in range(30):

            resultant_forces = calculate_resultant_forces(
                all_ver, set(edges), boundaries, self.optimum_length)

            for vertex, force in resultant_forces.items():
                if force.magnitude > 0.0001:
                    no_forces_applied = False
                apply_force_to_vertex(vertex, force)
            if no_forces_applied:
                break

    def __get_vor_ridges(self, vor_points, vor_vertices, ridge_points, ridge_vertices):
        finite_segments = []
        infinite_segments = []

        center = vor_points.mean(axis=0)
        ptp_bound = vor_vertices.ptp(axis=0)

        for pointidx, simplex in zip(ridge_points, ridge_vertices):
            simplex = np.asarray(simplex)

            if np.all(simplex >= 0):
                finite_segments.append(vor_vertices[simplex])
            else:
                i = simplex[simplex >= 0][0]  # finite end Voronoi vertex

                t = vor_points[pointidx[1]] - \
                    vor_points[pointidx[0]]  # tangent
                t /= np.linalg.norm(t)
                n = np.array([-t[1], t[0]])  # normal

                midpoint = vor_points[pointidx].mean(axis=0)
                direction = np.sign(np.dot(midpoint - center, n)) * n
                far_point = vor_vertices[i] + direction * ptp_bound.max()

                infinite_segments.append([vor_vertices[i], far_point])

        return finite_segments, infinite_segments

    def __adjust_vor_borders(self, finite, infinite, Borders, color):
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
                intersection_point = line1_start + \
                    t * (line1_end - line1_start)
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
                        Vertex(intersect_point[0], intersect_point[1], color))
                    break

        return intersection_points

    def add_boundary_points(self):
        for p in self.border_points:
            self.__temporary_vertices.add(Vertex(p.x, p.y, p.color))

    def merge_points(self, vertices, dist_threshold):
        merged_points = []
        merged_indices = set()

        for i in range(len(vertices)):
            if i not in merged_indices:
                merged_point = vertices[i]
                for j in range(i + 1, len(vertices)):
                    if j not in merged_indices:
                        resultant = Vector(
                            vertices[i].x, vertices[i].y) - Vector(vertices[j].x, vertices[j].y)
                        if resultant.magnitude < dist_threshold:
                            merged_point = Vertex(
                                (merged_point.x + vertices[j].x) / 2,
                                (merged_point.y + vertices[j].y) / 2,
                                vertices[j].color
                            )
                            merged_indices.add(j)
                merged_points.append(merged_point)
        return merged_points

    def create_temp_vertices(self):
        vertices = []
        temp_vertices = []
        for p in self.border_points:
            vertices.append([p.x, p.y])
        for p in self.vertices:
            vertices.append([p.x, p.y])
        voronoi = Voronoi(np.array(vertices))

        for v in voronoi.vertices:
            temp_vertices.append(Vertex(v[0], v[1], Color.GREEN.value))
        temp_vertices += self.__create_additional_verties(
            np.array(vertices), voronoi, color=Color.GREEN.value)

        min_dist = VERTEX_SIZE / CANVAS_SIZE
        merged_voronoi = self.merge_points(temp_vertices, min_dist)

        for v in merged_voronoi:
            if v.x < self.border_points[0].x or v.y < self.border_points[0].y or v.x > self.border_points[2].x or v.y > self.border_points[2].y:
                continue
            self.__temporary_vertices.add(v)

    def __create_additional_verties(self, np_arr, voronoi, color):
        finite, infinite = self.__get_vor_ridges(
            np_arr, voronoi.vertices, voronoi.ridge_points, voronoi.ridge_vertices)

        border_edges = [(self.border_points[0], self.border_points[1]),
                        (self.border_points[1], self.border_points[2]),
                        (self.border_points[2], self.border_points[3]),
                        (self.border_points[3], self.border_points[0])]

        additional_vor_points = self.__adjust_vor_borders(
            finite, infinite, border_edges, color)

        return additional_vor_points

    def __add_edges_to_temp_edges(self, del_edges, self_edges):
        def do_lines_intersect(P1, P2, P3, P4):
            dir1 = Vector(P2.x, P2.y) - Vector(P1.x, P1.y)
            dir2 = Vector(P4.x, P4.y) - Vector(P3.x, P3.y)

            det = dir1.x * (-dir2.y) - (-dir2.x) * dir1.y

            if det == 0:
                return False

            t1 = ((P3.x - P1.x) * (-dir2.y) - (-dir2.x) * (P3.y - P1.y)) / det
            t2 = (dir1.x * (P3.y - P1.y) - (P3.x - P1.x) * dir1.y) / det

            if 0 < t1 < 1 and 0 < t2 < 1:
                # Lines intersect within the given segments
                return True
            else:
                # Lines do not intersect within the given segments
                return False

        for (A, B) in del_edges:
            if (A, B) in self_edges or (B, A) in self_edges:
                continue
            for (E1, E2) in self_edges:
                if A in (E1, E2) or B in (E1, E2):
                    continue
                if do_lines_intersect(A, B, E1, E2):
                    break
            else:
                self.__temporary_edges.add((A, B))
                continue

    def triangulate(self):
        all_vertices = list(self.__vertices) + list(self.__temporary_vertices)
        delaunay = Delaunay(all_vertices)
        del_edges = delaunay.edges
        edges = []
        for path in self.pathes.values():
            edges += path.edges

        self.__add_edges_to_temp_edges(del_edges, list(edges))

    def merge_edges(self) -> None:
        for path in self.pathes.values():
            path.merge_edges(self.vertices, self.optimum_length)

