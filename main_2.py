from tkinter import *
from tkinter import messagebox
from tkinter import ttk
import numpy as np

from scipy.spatial import Delaunay
from scipy.spatial import Voronoi

import scipy_redrawing_test as rdr

        

class Game:
    def __init__(self, Voronoi_lines = True, Triengulate_lines = True, Triengulate_dead_points = False) -> None:
        self.points = []
        self.LinePoints = []
        #self.lines = []
        self.stepable_edges = set()

        self.step_points = []

        self.vor_size = 10
        self.game_pnt_size = 15
        self.num_game_points = 0

        self.root = Tk()
        frm = ttk.Frame(self.root)
        frm.grid()
        self.canvas_width = 640
        self.canvas_height = 640
        self.canvas = Canvas(frm, 
                width=self.canvas_width,
                height=self.canvas_height,
                bg='white')
        self.canvas.bind("<Button-1>", self.MouseClick)
        self.canvas.bind("<Button-3>", self.cancelMove)
        self.canvas.pack()

        self.drawing = False
        self.current_Line = []
        self.current_connecting_lines = []

        self.firstPlayerTurn = True
        self.moveColors = ['blue', 'yellow']


        self.voronoi_for_lines = Voronoi_lines
        self.triengulation_for_lines = Triengulate_lines
        self.Triengulation_for_dead_points = Triengulate_dead_points

    def place_point(self, x, y):
        self.num_game_points += 1
        point = rdr.Point(x, y,"G")
        self.points.append(point)
        return point
    
    def delaunay_edges(self, points):
        tri = Delaunay(points)
        edges = set()

        for simplex in tri.simplices:
            for i in range(3):
                for j in range(i + 1, 3):
                    edge = tuple(sorted([simplex[i], simplex[j]]))
                    edges.add(edge)
        
        edges = list(edges)
        
        return edges

    def do_lines_intersect(self, P1, P2, P3, P4):
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

    def redraw(self):
        self.canvas.delete('all')
        border_points = [rdr.Point(0.10,0.10,"B"),
                         rdr.Point(0.10,0.90, "B"),
                         rdr.Point(0.90,0.90,"B"),
                         rdr.Point(0.90,0.10,"B")]
        


        small_edges = []
        for lp in self.points + self.LinePoints:
            for p in lp.connection:
                if (lp, p) not in small_edges and (p, lp) not in small_edges:
                    small_edges.append((lp, p))

        # for edge in small_edges:
        #     print(edge[0], edge[1])


        

        # for p in self.LinePoints + self.points:
        #     print(p,":", end = " " )
        #     for c in p.connection:
        #         print(c, end = " ")
        #     print()
        border_edges = [(border_points[0], border_points[1]),
                        (border_points[1], border_points[2]),
                        (border_points[2], border_points[3]),
                        (border_points[3], border_points[0])]
        rdr.Redraw(self.points, border_points, self.LinePoints, small_edges, border_edges)
        # print("Redrawing")
        # for p in self.LinePoints:
        #     print(p)

    


        Voronoi_points = []
        if self.num_game_points > 3:
            pnts = []
            for p in self.points + self.LinePoints:
                pnts.append([p.x, p.y])
            # if self.voronoi_for_lines:
            #     for p in self.points + border_points + self.LinePoints:
            #         pnts.append([p.x, p.y])
            # else:
            #     for p in self.points + border_points:
            #         pnts.append([p.x, p.y])
            np_points = np.array(pnts)
            vor = Voronoi(np_points).vertices
            for p in vor:
                # for pt in self.LinePoints:
                #     if p.x == pt.x and p.y == pt.y:
                #         break
                # else:
                
                Voronoi_points.append(rdr.Point(p[0], p[1],"V"))

        pnts = []

        if self.triengulation_for_lines:
            for p in self.points + border_points + Voronoi_points + self.LinePoints:
                pnts.append([p.x, p.y])
        else:
            for p in self.points + border_points + Voronoi_points:
                pnts.append([p.x, p.y])
        np_points = np.array(pnts)


        edges = self.delaunay_edges(np_points)



        
        
        self.step_points = Voronoi_points + border_points



        drawn = set()
        for p in self.points + self.LinePoints:
            for i, cp in enumerate(p.connection):
                
                if (cp, p) in drawn or (p, cp) in drawn:
                    continue
                drawn.add((p, cp))
                x1, y1 = self.canvas_width * p.x, self.canvas_height * p.y
                x2, y2 = self.canvas_width * cp.x , self.canvas_height * cp.y
                color = p.connection_colors[i]
                self.canvas.create_line(x1, y1, x2, y2, fill=color ,width=2)

        for e in edges:
            p1 = e[0]
            p2 = e[1]
            P1, P2  = None, None
            all_points = self.points + border_points + Voronoi_points
            if self.triengulation_for_lines:
                all_points = all_points + self.LinePoints
            
            for p in all_points:
                if P1 is None and p.x == np_points[p1][0] and p.y == np_points[p1][1]:
                    P1 = p
                if P2 is None and p.x == np_points[p2][0] and p.y == np_points[p2][1]:
                    P2 = p
            if not self.Triengulation_for_dead_points and ((P1.origin == "G" and P1.lifes <= 0) or (P2.origin == "G" and P2.lifes <= 0)):
                continue

            for (P3, P4) in drawn:
                if self.do_lines_intersect(P1, P2, P3, P4):
                    break
            else:
                if P1.lifes > 0 and P2.lifes > 0:
                    self.stepable_edges.add((P1, P2))
                x1, y1 = self.canvas_width * P1.x, self.canvas_height *P1.y
                x2, y2 = self.canvas_width * P2.x, self.canvas_height *P2.y
                
                self.canvas.create_line(x1, y1, x2, y2, fill="gray", dash=(16,8) ,width=2)
        
        for p in Voronoi_points:
            x = self.canvas_width * p.x
            y = self.canvas_height * p.y
            x0 = x - self.vor_size / 2
            x1 = x + self.vor_size / 2
            y0 = y - self.vor_size / 2
            y1 = y + self.vor_size / 2

            self.canvas.create_oval(x0, y0, x1, y1, fill="green")
        
        for p in border_points:
            x = self.canvas_width * p.x
            y = self.canvas_height * p.y
            x0 = x - self.vor_size / 2
            x1 = x + self.vor_size / 2
            y0 = y - self.vor_size / 2
            y1 = y + self.vor_size / 2

            self.canvas.create_oval(x0, y0, x1, y1, fill="green")
        
        for p in self.LinePoints:
            x = self.canvas_width * p.x
            y = self.canvas_height * p.y
            x0 = x - self.vor_size / 2
            x1 = x + self.vor_size / 2
            y0 = y - self.vor_size / 2
            y1 = y + self.vor_size / 2
            color = p.color
            if p.color == '':
                print("Error in line point color")
                color = 'pink'
            self.canvas.create_oval(x0, y0, x1, y1, fill=color)
            
            
        for p in self.points:
            x = self.canvas_width * p.x
            y = self.canvas_height * p.y
            x0 = x - self.game_pnt_size / 2
            x1 = x + self.game_pnt_size / 2
            y0 = y - self.game_pnt_size / 2
            y1 = y + self.game_pnt_size / 2
            if p.lifes <= 0:
                self.canvas.create_oval(x0, y0, x1, y1, fill="grey")
            else:    
                self.canvas.create_oval(x0, y0, x1, y1, fill="red")


        # ##check for the end of game
        # Path_avaliable = False
        # for p in self.points:
        #     if p.InDeadRegion:
        #         continue
        #     if p.lifes == 2:
        #         Path_avaliable = True
        #         break
        #     if p.lifes <= 0:
        #         continue
        #     for pt in self.points:
        #         if pt.InDeadRegion:
        #             continue
        #         if pt == p:
        #             continue
        #         if pt.lifes <= 0:
        #             continue
        #         if self.is_route_exists(p, pt, self.stepable_edges):
        #             Path_avaliable = True
        #             break
        #     if Path_avaliable:
        #         break
        #     else:
        #         p.InDeadRegion = True
        # if not Path_avaliable:
        #     print("Game over")
        #     if self.firstPlayerTurn:
        #         print("Player 2 wins!")
        #         messagebox.showinfo("Winner", "Player 2 won this game!")
        #     else:
        #         print("Player 1 wins!") 
        #         messagebox.showinfo("Winner", "Player 1 won this game!")
            

    def is_route_exists(self, start_point, end_point, step_labeled_edges):
        visited = set()

        def dfs(current_point):
            if current_point == end_point:
                return True
            visited.add(current_point)

            for edge in step_labeled_edges:
                p1, p2 = edge
                if current_point == p1 and p2 not in visited:
                    if dfs(p2):
                        return True
                elif current_point == p2 and p1 not in visited:
                    if dfs(p1):
                        return True

            return False
        return dfs(start_point)
    
    
    def UpdateCurrentLine(self, point):
        if point.origin == "L":
            print("You can not intersect lines")
            return False
        if point.lifes <= 0:
            print("Point is now dead")
            return False

        if len(self.current_Line) != 0:
            prev_point = self.current_Line[-1]
            if (point, prev_point) not in self.stepable_edges and (prev_point, point) not in self.stepable_edges:
                print("You can step only within dotted lines")
                return False
        point.lifes -= 1

        x = self.canvas_width * point.x
        y = self.canvas_height * point.y
        x0,x1,y0,y1 = 0,0,0,0
        if point.origin == "G":
            x0 = x - self.game_pnt_size / 2
            x1 = x + self.game_pnt_size / 2
            y0 = y - self.game_pnt_size / 2
            y1 = y + self.game_pnt_size / 2
        else:
            x0 = x - self.vor_size / 2
            x1 = x + self.vor_size / 2
            y0 = y - self.vor_size / 2
            y1 = y + self.vor_size / 2
        
        color = self.moveColors[1]
        if self.firstPlayerTurn:
            color = self.moveColors[0]
        self.canvas.create_oval(x0, y0, x1, y1, fill=color)

        if len(self.current_Line)!=0:   
            x2 = self.canvas_width * self.current_Line[-1].x
            y2 = self.canvas_height * self.current_Line[-1].y
            line = self.canvas.create_line(x,y,x2,y2,fill = color, width=2)
            self.current_connecting_lines.append(line)
        self.current_Line.append(point)
        return True



    def divide_line_segment(self, A, B, l_opt, color):
    # Calculate the total length of the line segment AB
        length_AB = (A - B).len()

        # Calculate the number of segments needed
        num_segments = int(length_AB / l_opt)

        # Initialize a list to store the divided points
        divided_points = []

        # Calculate the coordinates of the points dividing the line segment
        for i in range(1, num_segments):
            ratio = i / num_segments
            x = A.x + ratio * (B.x - A.x)
            y = A.y + ratio * (B.y - A.y)
            divided_points.append(rdr.Point(x, y, origin="L", color=color))

        return divided_points

    def Step(self):
        color = self.moveColors[1]
        if self.firstPlayerTurn:
            color = self.moveColors[0]

        
        if len(self.current_Line) != 2:
            middle_line = len(self.current_Line) // 2
            self.current_Line[middle_line].origin = "G"
            self.current_Line[middle_line].lifes = 1
            self.num_game_points += 1
            self.points.append(self.current_Line[middle_line])
        else:
            x = (self.current_Line[0].x + self.current_Line[1].x) / 2
            y = (self.current_Line[0].y + self.current_Line[1].y) / 2
            point = self.place_point(x,y)
            point.lifes = 1
            self.current_Line.insert(1, point)
        
        for i in range(len(self.current_Line)):
            
            if self.current_Line[i] not in self.points:
                self.current_Line[i].origin = "L"
                self.current_Line[i].color = color
                self.LinePoints.append(self.current_Line[i])

            if i < len(self.current_Line) - 1:

                insert_points = self.divide_line_segment( self.current_Line[i], self.current_Line[i+1], rdr.l_opt, color)
                if len(insert_points) > 0:
                    self.current_Line[i].connect(insert_points[0], color)
                    for i_p in range(len(insert_points)):
                        self.LinePoints.append(insert_points[i_p])
                        if i_p < len(insert_points) - 1:
                            insert_points[i_p].connect(insert_points[i_p + 1], color)
                    insert_points[-1].connect(self.current_Line[i+1], color)

                else:        
                    self.current_Line[i].connect(self.current_Line[i+1], color)

        self.current_Line = []
        self.current_connecting_lines = []
        self.drawing = False
        self.stepable_edges = set()
        self.redraw()
        self.firstPlayerTurn = not self.firstPlayerTurn
        

    def MouseClick(self, event):
        pos = rdr.Point(event.x/self.canvas_width , event. y/self.canvas_height)
        minDist = float('inf')
        closest_point = None
        for p in self.points + self.step_points + self.LinePoints:
            dist = (pos - p).len()
            if dist<minDist:
                closest_point = p
                minDist = dist
        
        if not self.drawing and closest_point.origin != "G": 
            print("Line has to start from game point")
            return
        
        if closest_point.origin == "G" and self.drawing == True:
            if self.UpdateCurrentLine(closest_point):
                self.drawing = False
                self.Step()
            else:
                print("Move not finished")
        else:            
            if self.UpdateCurrentLine(closest_point):
                self.drawing = True
    
    def cancelMove(self, event):
        if len(self.current_Line) == 0:
            print("Current line is empty, nothing to cancel")
            return
        pos = rdr.Point(event.x/self.canvas_width , event. y/self.canvas_height)
        minDist = float('inf')
        closest_point = None
        for p in self.points + self.step_points:
            dist = (pos - p).len()
            if dist<minDist:
                closest_point = p
                minDist = dist
        if self.current_Line[-1] != closest_point:
            print("You can cancel only the last point")
            return
        self.current_Line.pop()
        closest_point.lifes += 1
        x = self.canvas_width * closest_point.x
        y = self.canvas_height * closest_point.y
        if closest_point.origin == "G":
            x0 = x - self.game_pnt_size / 2
            x1 = x + self.game_pnt_size / 2
            y0 = y - self.game_pnt_size / 2
            y1 = y + self.game_pnt_size / 2
            self.canvas.create_oval(x0, y0, x1, y1, fill="red")
        else:
            x0 = x - self.vor_size / 2
            x1 = x + self.vor_size / 2
            y0 = y - self.vor_size / 2
            y1 = y + self.vor_size / 2
            self.canvas.create_oval(x0, y0, x1, y1, fill="green")
        
        if (len(self.current_connecting_lines) != 0):
            line = self.current_connecting_lines.pop()
            self.canvas.delete(line)
        if len(self.current_Line) == 0:
            self.drawing = False
        print("Move canceled")

    
        
    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    game = Game(Voronoi_lines=True, Triengulate_lines=True, Triengulate_dead_points=True)
    game.place_point(0.3, 0.3)
    # game.place_point(0.6, 0.3)
    # game.place_point(0.6, 0.6)
    # game.place_point(0.3, 0.6)
    game.redraw()
    game.run()