import numpy as np
import measurement

class Line:

    def __init__(self, a,b,c):
        """ standard form a*x + b*y = c"""
        self.a = a
        self.b = b
        self.c = c

    def get_x(self,y):
        x = (-self.c - self.b*y) / float(self.a)
        return x

    def get_y(self,x):
        y = (-self.c - self.a*x) / float(self.b)
        return x

    def __str__(self):
        return str(self.a) +'x + ' + str(self.b) + 'y = ' + str(self.c)
    
def intersections(lines):
    intersections = []

    for i in range(len(lines)):
        for j in range(i+1, len(lines)):
            point = intersect(lines[i],lines[j])
            if point:
                intersections.append(point)
    return intersections

def intersect(l1, l2):
    detA = (l1.a*l2.b - l1.b*l2.a)
    if not detA == 0:
        x = (l1.c*l2.b - l1.b*l2.c) /  detA
        y = (l1.a*l2.c - l1.c*l2.a) /  detA
        return x,y
    return None

def line_from_points(p1,p2):
    x1,y1,z1 = p1
    x2,y2,z2 = p2
    a = y1-y2
    b = x2-x1
    c = (x1-x2)*y1 + (y2-y1)*x1
    return Line(a,b,-c)

class ImageSourceData:
    
    def __init__(self, data, N, r, room_dimensions):
        self.data = data
        self.N = N
        self.r = r
        self.L = room_dimensions
        self.images = None
        self.sources = None
        self.midpoints = None
        self.normals = None
        self.wallpoints = None
        self.vertices = None
        
    def find_walls(self, threshold, bestN):
        wall_points = self._calculate_wall_points(self.N, threshold=threshold, bestN=bestN)
        todel = []
        for i,wset in enumerate(wall_points):
            if np.any(np.array(wset)[:,2] > (self.L[2]-0.5)):
                todel.append(int(i))
            elif np.any(np.array(wset)[:,2] < 0.5):
                todel.append(int(i))
        self.wallpoints = np.delete(np.array(wall_points), np.array(todel, dtype='int'), axis=0)

        self.vertices = self._calculate_vertices2D(self.wallpoints)
        return self.wallpoints, self.vertices
        
    def _calculate_wall_points(self, N, threshold=0.05, bestN=5):
        """
            
        N            number of sources
        threshold    data filter
        bestN        Only use bestN results
        
        """
        X = self.data
        self.sources = []
        self.midpoints = []
        self.normals = []
        self.images = []
        keys = [k for k in sorted(X) if k < threshold][:bestN]
        for k in keys:
            data, distance_data = X[k]
            for j,e in enumerate(distance_data):
                source = measurement.locate_source(self.r, e[0,:])
                #i = e.measurement.index
                #source = data[:,i]
                est_images = (data[:,(N+j*6):(N+(j+1)*6)]).T
                mid_points = (est_images + source) / 2.0
                normal = source - est_images
                unit_normal = normal / np.linalg.norm(normal, axis=1).reshape(len(normal),1)
                self.images.append(est_images)
                self.normals.append(unit_normal)
                self.midpoints.append(mid_points)
                self.sources.append(source)

        sets = [[p] for p in self.midpoints[0]]
        for i,normal in enumerate(self.normals[0]):
            for k,normal_set in enumerate(self.normals[1:]):
                for j,other_normal in enumerate(normal_set):
                    if 0.9 < normal.dot(other_normal) < 1.1:
                        sets[i].append(self.midpoints[k+1][j])
        return sets

    def _calculate_vertices2D(self, wall_points):
        lines = []
        res_A = []
        res_B = []
        f_A = []
        f_B = []
        for i in range(len(wall_points)):
            data = np.vstack(wall_points[i])
            if len(data) > 2:
                x = data[:,0]
                y = data[:,1]
                fit_A, residuals_A, rank, singular_values, rcond = np.polyfit(x, y, 1, full=True)
                fit_B, residuals_B, rank, singular_values, rcond = np.polyfit(y, x, 1, full=True)
                f_A.append(fit_A)
                f_B.append(fit_B)
                res_A.append(residuals_A)
                res_B.append(residuals_B)
            elif len(data) == 2:
                lines.append(line_from_points(data[0], data[1]))


        if len(res_A) > 0:

            f_A = np.array(f_A)
            f_B = np.array(f_B)
            res_A = np.array(res_A)
            res_B = np.array(res_B)

            favorA = res_A < res_B

            for f in f_A[(favorA).ravel()]:
                #y = mx+c
                m,c = f
                lines.append(Line(-m, 1, c))

            for f in f_B[(~favorA).ravel()]:
                # x = ny+d
                n,d = f
                lines.append(Line(1, -n, d))
        
        if len(lines) > 1:
            intersects = np.array([i for i in intersections(lines) if np.all(np.abs(np.array(i)) < 100)])
        else: intersects = None

        return intersects
