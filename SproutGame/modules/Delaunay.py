from typing import List, Tuple
import matplotlib.tri as tri
from SproutGame.primitives import Vertex
from SproutGame.modules.Delaunator import Delaunator


class Delaunay:
    def __init__(self, vertices: List[Vertex]) -> None:
        self.vertices = vertices
        # x = []
        # y = []
        # for v in vertices:
        #     x.append(v.x)
        #     y.append(v.y)
        # self.vertices = vertices
        # self.triang = tri.Triangulation(x, y)

    @property
    def edges(self) -> List[Tuple[Vertex, Vertex]]:
        start = 0
        end = 3
        count = 0
        triangles = Delaunator(self.vertices).triangles
        triangles_list = list(range(len(triangles)//3))
        for idx in triangles_list:
            triangles_list[count] = tuple(triangles[start:end])
            start += 3
            end += 3
            count += 1
        all_edges_list = []
        for i in range(len(triangles_list)):
            for idx, a in enumerate(triangles_list[i]):
                for b in triangles_list[i][idx + 1:]:
                    all_edges_list.append(tuple({a, b}))

        edges_list = list(set(all_edges_list))
        # v1, v2 are indexes in vertices list
        edges = []
        for v1, v2 in edges_list:
            edges.append((self.vertices[v1], self.vertices[v2]))
        return edges
