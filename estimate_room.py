#!/usr/bin/env python
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio

import measurement
import echo
import distance
import image
import argparse

parser = argparse.ArgumentParser(description='Estimate the shape of a room from room impulse response data')
parser.add_argument('dataset', help='Dataset containing the room impulse response data measured in a room using 2 or more sources and 5 microphones')
parser.add_argument('-N', help='Number of sources', default=4)
parser.add_argument('-et', help='Echo Threshold', default=0.005)
parser.add_argument('-rt', help='Result Threshold', default=0.05)
parser.add_argument('--verbose', '-v', help='Print information during the estimation process', action='store_true')
args = parser.parse_args()

dataset = sio.loadmat(args.dataset)

fs =float(dataset['fs'])
M = int(dataset['M'])
#N = int(dataset['N'])
h = float(dataset['h'])
l = float(dataset['l'])
w = float(dataset['w'])
r = dataset['receivers']
s = dataset['sources']
data = dataset['data'].T
c = float(dataset['c'])

N = int(args.N)
et = float(args.et)
rt = float(args.rt)

maxsize = np.sqrt(w**2+l**2+h**2) #m
max_delay = maxsize / float(c) 
maxlength = int(2 * max_delay * fs)

measurements  = [measurement.MeasurementData(data=np.hstack(source_data).T, 
                                 receivers=r, 
                                 sources=s[i], 
                                 room_dimensions=(w,l,h), 
                                 c=c, 
                                 fs=fs) 
                 for i,source_data in enumerate(data)]

echo_data = [echo.EchoData(m.find_echoes(crop=maxlength, interpolate=10)) for m in measurements]

D = measurement.squared_distance_matrix(r, augmented=True)
S, E = zip(*[e.find_labels(D,threshold=et, parallel=True, verbose=args.verbose) for e in echo_data[:N]])
E = [e for e in E if len(e) > 0]
S = np.vstack(S)

distancedata = distance.DistanceData(S,E)
results = distancedata.find_images(r)

if len(results) > 0:
    imagedata = image.ImageSourceData(results, N, r, (w,l,h))
    wall_points,vertices = imagedata.find_walls(threshold=rt, bestN=10)
    im = np.vstack(imagedata.images)
    plt.scatter(im[:,0], im[:,1])
    wp = np.vstack(wall_points)
    plt.scatter(wp[:,0], wp[:,1], color=(1,0,0,1))
    plt.scatter(vertices[:,0], vertices[:,1], color=(0,1,0,1))
    plt.show()
else:
    print('No results to show')
