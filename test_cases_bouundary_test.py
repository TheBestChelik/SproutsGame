class Point:
    def __init__(self, x, y, name) -> None:
        self.x = x
        self.y = y
        self.name = name
        self.connections = []
    def addConnection(self, point):
        if point not in self.connections:
            self.connections.append(point)
        if self not in point.connections:
            point.addConnection(self)
    def getConnections_str(self):
        res = ""
        for c in self.connections:
            res +=" " + c.name
        return res
    
def Test1():
    A = Point(1,1,"A")
    B = Point(1,3, "B")
    C = Point(3,3, "C")
    D  = Point(3,1, "D")
    E = Point(2,2, "E")


    F = Point(0,4, "F")
    G = Point(0,0, "G")
    H = Point(4,0, "H")
    I = Point(4,4, "I")

    A.addConnection(B)
    B.addConnection(C)
    C.addConnection(D)
    D.addConnection(A)

    F.addConnection(G)
    F.addConnection(B)
    G.addConnection(H)
    H.addConnection(I)
    I.addConnection(C)

    return [A,B,C,D,E,F,G,H,I]

def Test2():
    A = Point(0,1, "A")
    B = Point(1,1,"B")
    C = Point(2,1,"C")
    D = Point(3,2, "D")
    E = Point(3,1,"E")
    F = Point(3,0,"F")
    G = Point(4,1,"G")
    H = Point(5,1,"H")

    I = Point(7,1,"I")
    J = Point(8,1,"J")
    K = Point(9,1,"K")

    L = Point(10,1,"L")

    A.addConnection(B)
    B.addConnection(C)
    C.addConnection(D)
    C.addConnection(F)
    D.addConnection(E)
    D.addConnection(H)
    E.addConnection(G)
    E.addConnection(F)
    G.addConnection(H)
    H.addConnection(F)

    I.addConnection(J)
    J.addConnection(K)

    return [A,B,C,D,E,F,G,H,I,J,K,L]

def Test3():
    A = Point(0,1, "A")
    B = Point(1,1,"B")
    C = Point(2,1,"C")
    D = Point(3,2, "D")
    E = Point(3,1,"E")
    F = Point(3,0,"F")
    G = Point(4,1,"G")
    H = Point(5,1,"H")

    I = Point(7,1,"I")
    J = Point(8,1,"J")
    K = Point(9,1,"K")

    L = Point(10,1,"L")

    M = Point(11,1, "M")
    N = Point(11,2,"M")
    O = Point(12,2,"O")
    P = Point(12,1,"P")
    Q = Point(13,1,"Q")
    R = Point(14,1,"R")
    S = Point(14,2,"S")
    T = Point(15,2,"T")
    U = Point(15,1, "U")
    V = Point(16,1,"V")

    W = Point(7,0, "W")
    X = Point(16,0,"X")
    Y = Point(10,0.5,"Y")
    Z = Point(12,0.5,"Z")

    A.addConnection(B)
    B.addConnection(C)
    C.addConnection(D)
    C.addConnection(F)
    D.addConnection(E)
    D.addConnection(H)
    E.addConnection(G)
    E.addConnection(F)
    G.addConnection(H)
    H.addConnection(F)
    H.addConnection(I)

    I.addConnection(J)
    J.addConnection(K)
    K.addConnection(L)
    L.addConnection(M)

    M.addConnection(N)
    M.addConnection(P)
    N.addConnection(O)
    O.addConnection(P)
    P.addConnection(Q)
    Q.addConnection(R)
    R.addConnection(S)
    R.addConnection(U)
    S.addConnection(T)
    T.addConnection(U)
    U.addConnection(V)

    W.addConnection(I)
    W.addConnection(X)
    X.addConnection(V)
    V.addConnection(Z)
    Z.addConnection(Y)
    Y.addConnection(J)

    L.addConnection(N)
    K.addConnection(Z)
    Y.addConnection(W)

    return [A,B,C,D,E,F,G,H,I,J,K,L,M,N,O,P,R,S,T,U,V,W,X,Y,Z]