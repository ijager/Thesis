"""
    This module is concerned with storing and manipulating room impulse 
    measurement data.
"""
import numpy as np
import multiprocessing as mp
import itertools
from scipy import interpolate


def squared_distance_matrix(X, augmented=False):
    """ Calculate the squared distance matrix for pointset X

        X               M by n pointset, with M number of points and n 
        the dimension
        augmented       If True, the matrix will be augmented by a row 
                        and column of zeros to make room for extra 
                        entries.

        returns D, M by M array or (M+1) by (M+1) matrix

    """
    XX = np.dot(X,X.T)
    D = np.outer(np.diag(XX), np.ones(len(X)))-2*XX+np.outer(np.ones(len(X)),
            np.diag(XX))
    if augmented == True:
        n = len(D)
        zeros_v = np.zeros((n,1))
        zeros_h = np.zeros((1,n+1))
        D = np.bmat('D zeros_v; zeros_h')
    return D

def interpolate_spline(y, N):
    """ Interpolates signal y by a factor N using a 1 dimensional 
        spline."""
    l = len(y)
    x = np.linspace(0, l, l)
    spline = interpolate.InterpolatedUnivariateSpline(x,y)
    xnew = np.linspace(0, l, N*l)
    ynew = spline(xnew)
    return ynew
    
def local_max(x, threshold=1e-5):
    """ Get all local maxima of x by selecting all points which are 
        higher than its left and right neighbour

        x           1-dimensional arraylike signal
        threshold   select all maxima > threshold

        returns array containing the indices of all 
                local maxima > threshold
    """
    maxima = np.r_[True, x[1:] > x[:-1]] & np.r_[x[:-1] > x[1:] , True]
    # select all local maxima above the threshold
    maxima_f = maxima & np.r_[x > threshold , True][:-1]
    peak_indices =  np.where(maxima_f==True)[0]
    return np.array(peak_indices)
 
def direct_sound(rir):
    """ Finds the direct sound (first peak) in a room impulse response 
        (rir).

        rir     1d impulse response signal

        returns 
        direct_sound_index  index of highest value in rir
        rng                 radius around direct sound which contains 
                            its highest side lobes
        all_peak_indices    array containing all indices of local 
                            maxima in the signal
    """
    all_peak_indices = local_max(rir,threshold=1e-5)
    i = np.argsort(rir[all_peak_indices])
    direct_sound_index = all_peak_indices[i][-1]
    peak_indices = all_peak_indices[np.where(all_peak_indices < 
        direct_sound_index)[0]]
    values = rir[peak_indices]
    # direct sound side lobe range
    rng = direct_sound_index - max(peak_indices[np.where(values < 
        (rir[direct_sound_index] * 0.02))[0]])
    return direct_sound_index, rng, all_peak_indices    

def locate_source(p,d):
    """ Locate source x using multilateration. Compute x using pointset 
        p and distances d. The distances do not need be absolute as 
        multilateration works using time-differences of arrival (TDOA). 

        p   sensors as a Mxn arraylike with M, number of sensors and 
            n the dimension
        d   distances from x to all sensors in d

        returns x, n-dimensional position of the source 
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
        A[row,:] = 2*( (d[j])*(p[:,i]-p[:,c]).T - \
                (d[i])*(p[:,j]-p[:,c]).T )
        b[row,0] = (d[i])*((d[j])**2-p[:,j].T*p[:,j]) + \
        ((d[i])-(d[j]))*p[:,c].T*p[:,c] + \
        (d[j])*(p[:,i].T*p[:,i]-(d[i])**2)


    x = np.asarray( np.linalg.lstsq(A,b)[0] )[:,0]
    return x

class MeasurementData:
    """
        Container class for measurement data and parameters. 

        This class expects data one source recorded by M (all) 
        microphones, the data should be provided in the following 
        shapes:

        data        room impulse response data, arraylike, this should 
                    be shaped as (M, rirlength) with N the number of 
                    receivers and rirlength the length of the room 
                    impulse response.
        receivers   M by 3 arraylike, containing the 3 dimensional 
                    positions of M receivers. (required)
        sources     N by 3 arraylike, containing the 3 dimensional 
                    positions of N sources. (optional)
        room_dimensional    (w,l,h) width, length and heigth of the 
                            shoebox shaped room. Used for verification
        c           speed of sound, default = 343 m/s
        fs          sample frequency, defailt = 96000 Hz
    """
    
    def __init__(self, data, receivers, sources, room_dimensions, c=343, 
            fs=96000):
        self.data = data
        self.fs = fs
        self.c = c
        self.r = receivers
        self.s = sources
        self.L = room_dimensions
        self.upsampling_rate = 1
           
    def crop(self, N):
        """ Crop data to N samples, applied to all rirs."""
        self.data = self.data[:,:N]
        
    def interpolate(self, N):
        """ Interpolates all rirs with a upsampling rate of N. """
        self.upsampling_rate *= N
        self.data = np.array([interpolate_spline(x,N) for x in self.data])
      
    def interpolate_parallel(self, N):
        """ Interpolates all rirs in parallel with a upsampling rate 
            of N. 
        """
        with mp.Pool(processes=mp.cpu_count()) as pool:        
            self.new_data = np.array(pool.starmap(interpolate_spline, 
                zip(self.data, itertools.repeat(N))))
            
    def find_echoes(self, n=7, correct_offset=True, interpolate=None, 
            crop=None):
        """ Finds n highest echoes in each room impulse response

            n   number of echoes (peaks) to select

            correct_offset  indicates whether to correct the offset by 
                            computing the real source location. This 
                            ensures that all echoes correspond to 
                            absolute distances.
            interpolate     upsampling rate if interpolation is 
                            desired.
            crop            crop data to this value before 
                            interpolation.

            returns         M by n arraylike containing the echoes as 
                            distances in meter.  
        """

        if isinstance(crop, (int, float)):
            self.crop(crop)

        if isinstance(interpolate, (int, float)):
            self.interpolate(interpolate)

        echoes = np.zeros([len(self.data), n])
        for j,rir in enumerate(self.data):
            direct_sound_index, side_lobe_rng, all_peak_indices = direct_sound(
                    rir)
            N = direct_sound_index + side_lobe_rng
            peak_indices = all_peak_indices[
                    np.where(all_peak_indices > N)[0]]
            i = np.argsort(rir[peak_indices])
            sorted_peak_indices = peak_indices[i][::-1]
            echoes[j,:] = np.r_[direct_sound_index, 
                    sorted_peak_indices[:(n-1)]]
        
        echoes = echoes * self.c / (self.upsampling_rate * self.fs)
        if correct_offset == True:
            offset = self._calculate_offset()
            echoes += offset[:,np.newaxis]
        return echoes
            
    def _calculate_offset(self):
        direct_sound_distances = np.argmax(self.data, 
                axis=1) * self.c / (self.upsampling_rate * self.fs)
        source_est = np.array(locate_source(self.r, direct_sound_distances))
        true_dist = np.linalg.norm(self.r - source_est, axis=1)
        offset = true_dist - direct_sound_distances
        return np.array(offset).T
    
        
