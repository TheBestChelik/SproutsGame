import numpy as np

from SproutGame.resources.constants import Color, LineStyle
from typing import Set, Tuple, List


class Vector:
    def __init__(self, x: float, y: float) -> None:
        self.__x = x
        self.__y = y

    @property
    def x(self) -> float:
        return self.__x

    @property
    def y(self) -> float:
        return self.__y

    @property
    def magnitude(self) -> float:
        return np.linalg.norm([self.x, self.y])

    def __add__(self, other):
        return Vector(self.x + other.x, self.y + other.y)

    def __iadd__(self, other):
        self.__x += other.x
        self.__y += other.y
        return self

    def __sub__(self, other):
        return Vector(self.x - other.x, self.y - other.y)

    def __mul__(self, other):
        if isinstance(other, float):
            return Vector(self.x * other, self.y * other)

        if isinstance(other, Vector):
            return Vector(self.x * other.x, self.y * other.y)

        raise ValueError("Unsupported operand type for multiplication")

    def __truediv__(self, scalar: float):
        return Vector(self.x / scalar, self.y / scalar)

    def dot(self, other) -> float:
        return np.dot(np.array([self.x, self.y]), np.array([other.x, other.y]))

    def __str__(self) -> str:
        return f"{self.x, self.y}"


class Vertex:
    def __init__(self, x: float, y: float, color) -> None:
        self.__x = x
        self.__y = y
        self.__color = color

    @property
    def x(self) -> float:
        return self.__x

    @property
    def y(self) -> float:
        return self.__y

    @property
    def color(self):
        return self.__color

    @color.setter
    def color(self, color) -> None:
        self.__color = color

    # Rewrite types if python version is 3.11

    def translate(self, vector: Vector) -> None:
        self.__x += vector.x
        self.__y += vector.y

    def __str__(self) -> str:
        return f"({self.__x}; {self.__y})"


class Spot(Vertex):
    def __init__(self, x: float, y: float, color, liberties: int) -> None:
        super().__init__(x, y, color)
        self.__liberties = liberties

    @property
    def liberties(self) -> int:
        return self.__liberties

    def minus_liberty(self) -> None:
        self.__liberties -= 1
        if self.__liberties == 0:
            self.color = Color.GREY.value

    def add_liberty(self) -> None:
        self.__liberties += 1
        if self.__liberties != 0:
            self.color = Color.RED.value


class Path:
    def __init__(self, color, edges: List[Tuple[Vertex, Vertex]]) -> None:
        self.__color = color
        self.__edges = edges

    @property
    def color(self):
        return self.__color

    @property
    def edges(self) -> Set[Tuple[Vertex, Vertex]]:
        return self.__edges

    @edges.setter
    def edges(self, edges: Set[Tuple[Vertex, Vertex]]) -> None:
        self.__edges = edges

    def __segment_edge(self, A: Vertex, B: Vertex, vertices: Set[Vertex], optimum_length: float) -> List[Tuple[Vertex, Vertex]]:
        length = (Vector(A.x, A.y) - Vector(B.x, B.y)).magnitude

        if length <= 4 * optimum_length / 3:
            return [(A, B)]

        fraction = 0.5

        new_x = A.x + (B.x - A.x) * fraction
        new_y = A.y + (B.y - A.y) * fraction

        divided_node = Vertex(new_x, new_y, self.color)
        vertices.add(divided_node)

        left_half = self.__segment_edge(
            A, divided_node, vertices, optimum_length)
        right_half = self.__segment_edge(
            divided_node, B, vertices, optimum_length)

        return left_half + right_half

    def segment_edges(self, vertices, optimum_length):
        indices_to_remove = []
        edges_to_add = []
        edges_set = set(self.__edges)

        for (A, B) in edges_set:
            new_edge_list = self.__segment_edge(A, B, vertices, optimum_length)
            indices_to_remove.append((A, B))
            edges_to_add.append(new_edge_list)

        # Insert new edges at the same indices where old edges were removed
        for (new_edge_list, (A, B)) in zip(edges_to_add, indices_to_remove):
            index = self.__edges.index((A, B))
            self.__edges.remove((A, B))
            for new_edge in new_edge_list:
                self.__edges.insert(index, new_edge)
                index += 1

    def merge_edges(self, vertices: Set[Vertex], optimum_length: float):
        if len(self.__edges) == 1:
            return
        # print("Merging")
        merged_edges = []
        i = 0
        # for p in self.__edges:
        #     print(p[0], p[1])

        while i < len(self.__edges):
            node1, node2 = self.__edges[i]

            # Calculate the distance between node1 and node2
            distance = (Vector(node2.x, node2.y) -
                        Vector(node1.x, node1.y)).magnitude

            if distance < (2/3) * optimum_length:
                # Merge node1 and node2 into a single tuple
                if i + 1 < len(self.__edges):
                    # Get the second node of the next tuple
                    node3 = self.__edges[i + 1][1]
                    vertices.remove(node2)
                    merged_edges.append((node1, node3))
                    i += 1  # Skip the next tuple since it's merged
                else:
                    if len(merged_edges) != 0:
                        prev_node, _ = merged_edges.pop()
                        vertices.remove(_)
                        merged_edges.append((prev_node, node2))
                    else:
                        merged_edges.append((node1, node2))
            else:
                # Distance is greater than or equal to (2/3) * l_opt, keep the original tuple
                merged_edges.append((node1, node2))

            i += 1

        self.__edges = merged_edges
