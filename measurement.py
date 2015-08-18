import numpy as np
import multiprocessing
import itertools
from scipy import interpolate

def squared_distance_matrix(X, augmented=False):
    """ Calculate the squared distance matrix for pointset X

        X               M by n pointset, with M number of points and n the dimension
        augmented       if True, the matrix will be augmented by a row and column of
                        zeros to make room for extra entries

        returns D, M by M array or (M+1) by (M+1) matrix

    """
    XX = np.dot(X,X.T)
    D = np.outer(np.diag(XX), np.ones(len(X)))-2*XX+np.outer(np.ones(len(X)),np.diag(XX))
    if augmented == True:
        n = len(D)
        zeros_v = np.zeros((n,1))
        zeros_h = np.zeros((1,n+1))
        D = np.bmat('D zeros_v; zeros_h')
    return D

def interpolate_spline(y, N):
    l = len(y)
    x = np.linspace(0, l, l)
    spline = interpolate.InterpolatedUnivariateSpline(x,y)
    xnew = np.linspace(0, l, N*l)
    ynew = spline(xnew)
    return ynew
    
def local_max(x, threshold=1e-5):
    """
        Get all local maxima of x by selecting all points which are 
        higher than its left and right neighbour
    """
    maxima = np.r_[True, x[1:] > x[:-1]] & np.r_[x[:-1] > x[1:] , True]
    # select all local maxima above the threshold
    maxima_f = maxima & np.r_[x > threshold , True][:-1]
    peak_indices =  np.where(maxima_f==True)[0]
    return np.array(peak_indices)
 
def direct_sound(x):
    all_peak_indices = local_max(x,threshold=1e-5)
    i = np.argsort(x[all_peak_indices])
    direct_sound_index = all_peak_indices[i][-1]
    peak_indices = all_peak_indices[np.where(all_peak_indices < direct_sound_index)[0]]
    values = x[peak_indices]
    # direct sound side lobe range
    rng = direct_sound_index - max(peak_indices[np.where(values < (x[direct_sound_index] * 0.02))[0]])
    return direct_sound_index, rng, all_peak_indices    

def locate_source(p,d):
    """ Locate source x using multilateration

        p    sensors as a Mxn arraylike with M, number of sensors and n the dimension
        d    distances from x to all sensors in d

        returns x
    """
    # M = sensors, n = dimensions
    M, n = p.shape
    p = np.matrix( p ).T

    # pick closest receiver
    c = np.argmin(d)
    #sensors delta time relative to sensor c
    d = d - min(d)

    indices = list(range(M))
    del indices[c]

    A = np.zeros([M-2,n])
    b = np.zeros([M-2,1])

    i = indices[0]
    for row,j in enumerate(indices[1:]):
        A[row,:] = 2*( (d[j])*(p[:,i]-p[:,c]).T - (d[i])*(p[:,j]-p[:,c]).T )
        b[row,0] = (d[i])*((d[j])**2-p[:,j].T*p[:,j]) + \
        ((d[i])-(d[j]))*p[:,c].T*p[:,c] + \
        (d[j])*(p[:,i].T*p[:,i]-(d[i])**2)


    x = np.asarray( np.linalg.lstsq(A,b)[0] )[:,0]
    return x

class MeasurementData:
    
    def __init__(self, data, receivers, sources, room_dimensions, c=343, fs=96000):
        """
        
        Container class for measurement data and parameters
        
        """
        self.data = data
        self.fs = fs
        self.c = c
        self.r = receivers
        self.s = sources
        self.L = room_dimensions
        self.upsampling_rate = 1
           
    def crop(self, N):
        self.data = self.data[:,:N]
        
    def interpolate(self, N):
        self.upsampling_rate *= N
        self.data = np.array([interpolate_spline(x,N) for x in self.data])
      
    def interpolate_parallel(self, N):
        with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:        
            self.new_data = np.array(pool.starmap(interpolate_spline, zip(self.data, itertools.repeat(N))))
            
    def find_echoes(self, n=7, correct_offset=True, interpolate=None, crop=None):

        if isinstance(crop, (int, float)):
            self.crop(crop)

        if isinstance(interpolate, (int, float)):
            self.interpolate(interpolate)

        echoes = np.zeros([len(self.data), n])
        for j,rir in enumerate(self.data):
            direct_sound_index, side_lobe_rng, all_peak_indices = direct_sound(rir)
            N = direct_sound_index + side_lobe_rng
            peak_indices = all_peak_indices[np.where(all_peak_indices > N)[0]]
            i = np.argsort(rir[peak_indices])
            sorted_peak_indices = peak_indices[i][::-1]
            echoes[j,:] = np.r_[direct_sound_index, sorted_peak_indices[:(n-1)]]
        
        echoes = echoes * self.c / (self.upsampling_rate * self.fs)
        if correct_offset == True:
            offset = self._calculate_offset()
            echoes += offset[:,np.newaxis]
        return echoes
            
    def _calculate_offset(self):
        direct_sound_distances = np.argmax(self.data, axis=1) * self.c / (self.upsampling_rate * self.fs)
        source_est = np.array(locate_source(self.r, direct_sound_distances))
        true_dist = np.linalg.norm(self.r - source_est, axis=1)
        offset = true_dist - direct_sound_distances
        return np.array(offset).T
    
        
