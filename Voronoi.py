import matplotlib.pyplot as plt
import matplotlib.tri as tri
import numpy as np
import itertools


class Voronoi:
    
    def __init__(self, points) -> None:
        lineList = []
        mPointsList = []
        perpendicularLineList = []
        self.vertexList = []

        x = points[:,0]
        y = points[:,1]

        triang = tri.Triangulation(x, y)
        triList = triang.get_masked_triangles()
        for i in range(len(triList)):
            lineList.append([])
            for idx, a in enumerate(triList[i]):
                for b in triList[i][idx + 1:]:
                    lineList[i].append((a, b))

        count = 0
        for triangle in lineList:
            perpendicularLineList.append([])
            mPointsList.append([])

            for line in triangle:
                x1 = x[line[0]]
                y1 = y[line[0]]
                x2 = x[line[1]]
                y2 = y[line[1]]

                mX = (x1 + x2) / 2
                mY = (y1 + y2) / 2

                if ((y2 - y1) != 0):
                    k = -((x2 - x1) / (y2 - y1))
                    b = mY - (k * mX)
                    perpendicularLineList[count].append((k, b))

                mPointsList[count].append((mX, mY))
            count += 1

        for triangle in perpendicularLineList:
            k1 = triangle[0][0]
            k2 = triangle[1][0]
            b1 = triangle[0][1]
            b2 = triangle[1][1]

            X = (b1 - b2) / (k2 - k1)
            Y = (k1 * X) + b1

            self.vertexList.append((X, Y))

        connectorsList = []
        counter = 0
        for triangle in mPointsList:
            connectorsList.append([])
            for lineIdx in range(len(triangle)):
                connectorsList[counter].append(
                    ((triangle[lineIdx][0], triangle[lineIdx][1]), (self.vertexList[counter][0], self.vertexList[counter][1])))
            counter += 1

        connectorListCompare = list(itertools.chain(*connectorsList))

        self.ridge_vertices_list = []
        self.ridge_points_notnp = []

        for connectorCompare1 in connectorListCompare:
            if list(itertools.chain(*list(itertools.chain(*connectorsList)))).count(connectorCompare1[0]) == 1:
                self.ridge_vertices_list.append([-1, self.vertexList.index(connectorCompare1[1])])
                self.ridge_points_notnp.append(list(
                    list(itertools.chain(*lineList))[list(itertools.chain(*mPointsList)).index(connectorCompare1[0])]))
            else:
                for connectorCompare2 in connectorListCompare:
                    if connectorCompare1[0] == connectorCompare2[0] and connectorCompare1[1] != connectorCompare2[1]:
                        self.ridge_vertices_list.append(
                            [self.vertexList.index(connectorCompare2[1]), self.vertexList.index(connectorCompare1[1])])
                        self.ridge_points_notnp.append(list(list(itertools.chain(*lineList))[
                                                           list(itertools.chain(*mPointsList)).index(
                                                               connectorCompare1[0])]))
                        connectorListCompare.remove(connectorCompare2)

    
    @property
    def vertices(self):
        vor_vertices = np.array(self.vertexList)
        return vor_vertices
    
    @property
    def ridge_points(self):
        ridge_points = np.array(self.ridge_points_notnp)
        return ridge_points
    
    @property
    def ridge_vertices(self):
        ridge_vertices = self.ridge_vertices_list
        return ridge_vertices
        