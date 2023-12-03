import numpy as np
import matplotlib.pyplot as plt
import random


l_opt = 0.05 * 3
width_small_edge = 0.02

attract_const = l_opt / 2
repuls_const = l_opt 
edge_repuls_const = width_small_edge * 10

class Point:
    def __init__(self, x, y, origin="", color = "") -> None:
        self.x = x
        self.y = y
        self.origin = origin
        self.connection = []
        self.connection_colors = []
        self.color = color
        self.lifes = 3
        self.InDeadRegion = False

    def len(self):
        return np.linalg.norm([self.x,self.y])

    def __add__(self, other):
        return Point(self.x + other.x, self.y + other.y, self.origin)
    
    def __sub__(self, other):
        return Point(self.x - other.x, self.y - other.y, self.origin)
    
    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return Point(self.x * other, self.y * other, self.origin)
        elif isinstance(other, Point):
            return Point(self.x * other.x, self.y * other.y, self.origin)
        else:
            raise ValueError("Unsupported operand type for multiplication")

    def __truediv__(self, scalar):
        return Point(self.x / scalar, self.y / scalar, self.origin)
    
    def __str__(self):
        return f"({self.x}, {self.y})"
    
    def move(self, point):
        self.x += point.x
        self.y += point.y

    def connect(self, point, color):
        #self.lifes -= 1
        if point not in self.connection:
            self.connection_colors.append(color)
            self.connection.append(point)
        if self not in point.connection:
            point.connect(self, color)


# force-directed algorithm

def ApplyForse(point, forse):
    Move_max = 0.01
    forse_len = forse.len()
    #res = Point(0.5,0.5, point.origin)
    if forse_len > Move_max:
        balanced_forse = ( forse / forse_len ) * Move_max
        point.move(balanced_forse)
    else:
        point.move(forse)
    # if point.x > 0.85:
    #     point.x = 0.85
    # if point.x < 0.15:
    #     point.x = 0.15
    # if point.y < 0.15:
    #     point.y = 0.15
    # if point.y > 0.85:
    #     point.y = 0.85




def project_point_onto_line_segment(start_point, end_point, point_v):
    # Calculate the direction vector of the line segment
    line_direction = Point(end_point.x - start_point.x, end_point.y - start_point.y)

    # Calculate the vector from start_point to point_v
    to_point_v = Point(point_v.x - start_point.x, point_v.y - start_point.y)

    # Calculate the dot product of to_point_v and line_direction
    dot_product = (to_point_v.x * line_direction.x) + (to_point_v.y * line_direction.y)

    # Calculate the projection of point_v onto the line segment
    projection = Point(
        start_point.x + (dot_product * line_direction.x),
        start_point.y + (dot_product * line_direction.y)
    )

    return projection

def F_attraction(u,v,attract_const, l_opt):
    dist = (u - v).len()
    f = (u - v) * (dist / attract_const)
    if dist <= l_opt:
        f = Point(0,0)

    return f

def F_repulsion_edge(v, edge, edge_repuls_const):
    v_edge = project_point_onto_line_segment(edge[0], edge[1], v)
    dist = (v_edge - v).len()
    if dist >= edge_repuls_const:
        return Point(0,0)
    if dist == 0:
        print("edge distance is 0")
        print(edge, v)
    #f = ( (v - v_edge) / dist ) * ( (edge_repuls_const - dist) ** 2 ) 
    f = ( (v - v_edge) / dist ) * ( (edge_repuls_const - dist) ** 2 ) 
    #f = (v - v_edge) * ( edge_repuls_const / dist )
    return f

def F_repulsion(u, v, repuls_const):
    dist = (u-v).len()
    if dist >= repuls_const:
        return Point(0,0)
    if dist == 0:
        u.move(Point(0.01,0.01))
        v.move(Point(-0.01,-0.01))
        dist = (u-v).len()
        print("distance is 0")
        print(u, v)
    f = ( v - u ) * ( ( repuls_const / dist ) ** 2 )
    return f

def BorderForse(v, edge, bord_repuls_const):
    if v.x >= 0.85:
        v.x = 0.84
    if v.x <= 0.15:
        v.x = 0.16
    if v.y >= 0.85:
        v.y = 0.84
    if v.y <= 0.15:
        v.y = 0.16

    v_edge = project_point_onto_line_segment(edge[0], edge[1], v)
    dist = (v_edge - v).len()
    if dist >= bord_repuls_const:
        return Point(0,0)
    else:
        return (v - v_edge) * (bord_repuls_const / dist) 
    # f = (v - v_edge) * ( ( (bord_repuls_const - dist) ** 2 ) / dist )
    # return f

def Redraw(Game_Points, Border_Points, Voronoi_Points, Line_Points, small_edges, border_edges):
    points = Game_Points + Border_Points + Voronoi_Points + Line_Points
    
    #game_verties = len(Game_Points)

    # print(game_verties)



    # l_opt = 0.045 - 0.0008 * game_verties
    # width_small_edge = 0.023 - 0.0004 * game_verties
    # if game_verties < 40:
    #     l_opt = 0.013
    #     width_small_edge = 0.007


    for i in range(10):
        #new_points = []\
        ForsesToApply = []
        
        for i in range(len(points)):
            Forse_Sum = Point(0,0)
            for j in range(len(points)):
                if j != i:
                    
                    u, v = points[j], points[i]
                    if v in u.connection:
                    #if (u.origin == "L" or v.origin == "L") and v in u.connection:
                        F_atr = F_attraction(u,v, attract_const,l_opt)
                        Forse_Sum = Forse_Sum + F_atr
                        if u.origin == "G" and v.origin == "G":
                            F_repuls = F_repulsion(u, v, repuls_const)
                            Forse_Sum = Forse_Sum + F_repuls
                    else:
                        new_repuls_const = repuls_const
                        if u.origin == "V" and v.origin != "V":
                            new_repuls_const = repuls_const / 2
                        if u.origin == "G" and v.origin == "L":
                            new_repuls_const = repuls_const * 2    
                        F_repuls = F_repulsion(u, v, new_repuls_const)
                        Forse_Sum = Forse_Sum + F_repuls
            
            for edge in small_edges:
                if points[i] in edge:
                    continue
                F_repuls_edge = F_repulsion_edge(points[i], edge, edge_repuls_const)
                Forse_Sum = Forse_Sum + F_repuls_edge

                
            for b_edge in border_edges:
                if points[i].origin == "B":
                    continue
                F_border = BorderForse(points[i], b_edge, edge_repuls_const)
                Forse_Sum = Forse_Sum + F_border

            ForsesToApply.append((points[i], Forse_Sum))
            #new_points.append(v_new)

        for v, f in ForsesToApply:
            ApplyForse(v, f)

if __name__ == "__main__":

    points = []
    Game_Points = []
    Voronoi_Points = [Point(0.301,0.4, "V")]
    Border_Points = []
    Line_Points = [Point(0.3,0.3, "L"), Point(0.3,0.5, "L"), Point(0.5, 0.8, "L")]
    small_edges = [(Line_Points[0],Line_Points[1]), (Line_Points[1], Line_Points[2])]
    
    # for i in range(10):
    #     Game_Points.append(Point(random.uniform(0.15, 0.85), random.uniform(0.15, 0.85), "G"))

    # for i in range(10):
    #     Border_Points.append(Point(random.uniform(0.15, 0.85), random.uniform(0.15, 0.85), "B"))
    
    # for i in range(10):
    #     Voronoi_Points.append(Point(random.uniform(0.15, 0.85), random.uniform(0.15, 0.85), "V"))
    
    # for i in range(10):
    #     Line_Points.append(Point(random.uniform(0.15, 0.85), random.uniform(0.15, 0.85), "L"))

    points = Game_Points + Border_Points + Voronoi_Points + Line_Points
    plt.figure(1)
    plt.title("Random")
    x_coords = [point.x for point in points]
    y_coords = [point.y for point in points]
    plt.scatter(x_coords, y_coords, color='b', marker='o')
    for edge in small_edges:
        x_values = [edge[0].x, edge[1].x]
        y_values = [edge[0].y, edge[1].y]
        plt.plot(x_values, y_values, marker='o', linestyle='-', markersize=5)

    Redraw(Game_Points, Border_Points, Voronoi_Points , Line_Points, small_edges)




    plt.figure(2)
    plt.title("Redrawed")
    x_coords = [point.x for point in points]
    y_coords = [point.y for point in points]
    plt.scatter(x_coords, y_coords, color='b', marker='o')
    for edge in small_edges:
        x_values = [edge[0].x, edge[1].x]
        y_values = [edge[0].y, edge[1].y]
        plt.plot(x_values, y_values, marker='o', linestyle='-', markersize=5)

    plt.show()

    
