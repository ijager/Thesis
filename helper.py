#!/usr/bin/env python
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import time
import sys

import measurement
import echo
import distance
import image
import argparse
import glob
import collections
import pickle

def find_matching(X,Y):
    """ find indices so that Y[indices] is congruent with X

        returns indices

    """
    index = []
    for x in X:
        m = 9999999
        temp_i = 0
        for i,y in enumerate(Y):
            d = np.linalg.norm(x-y)
            if d < m:
                m = d
                temp_i = i
        index.append(temp_i)
    return index

def test(datasetname, N, et, rt, ur):
    dataset = sio.loadmat(datasetname)

    fs =float(dataset['fs'])
    h = float(dataset['h'])
    l = float(dataset['l'])
    w = float(dataset['w'])
    r = dataset['receivers']
    s = dataset['sources']
    data = dataset['data'].T
    c = float(dataset['c'])

    room = np.array([[0,0],[0,l],[w,l],[w,0]])

    maxsize = np.sqrt(w**2+l**2+h**2) #m
    max_delay = maxsize / float(c) 
    maxlength = int(2 * max_delay * fs)
    t0 = time.time()
    measurements  = [measurement.MeasurementData(data=np.hstack(source_data).T, 
                                     receivers=r, 
                                     sources=s[i], 
                                     room_dimensions=(w,l,h), 
                                     c=c, 
                                     fs=fs) 
                     for i,source_data in enumerate(data)]

    echo_data = [echo.EchoData(m.find_echoes(crop=maxlength, interpolate=ur)) for m in measurements]

    D = measurement.squared_distance_matrix(r, augmented=True)
    S, E = zip(*[e.find_labels(D,threshold=et, parallel=True, verbose=args.verbose) for e in echo_data[:N]])
    E = [e for e in E if len(e) > 0]
    S = np.vstack(S)

    distancedata = distance.DistanceData(S,E)
    results = distancedata.find_images(r)

    t1 = time.time()
    if len(results) > 0:
        imagedata = image.ImageSourceData(results, N, r, (w,l,h))
        wall_points,vertices = imagedata.find_walls(threshold=rt, bestN=10)
        if len(vertices) == 4:
            i = find_matching(room, vertices)
            error = np.sqrt(np.mean((vertices[i] - room)**2))
            return (w*l*h, error, fs, t1-t0)
    return (w*l*h, -1, fs, t1-t0)

parser = argparse.ArgumentParser(description='Estimate the shape of a room from room impulse response data')
parser.add_argument('dataset', help='Dataset containing the room impulse response data measured in a room using 2 or more sources and 5 microphones')
parser.add_argument('-N', help='Number of sources', default=4)
parser.add_argument('-et', help='Echo Threshold', default=0.005)
parser.add_argument('-rt', help='Result Threshold', default=0.05)
parser.add_argument('-ur', help='Upsampling rate', default=10)
parser.add_argument('-o', help='Output file', default=None)
parser.add_argument('--verbose', '-v', help='Print information during the estimation process', action='store_true')
args = parser.parse_args()


N = int(args.N)
et = float(args.et)
rt = float(args.rt)
upsampling_rate = int(args.ur)

output = collections.defaultdict(list)

datasets = glob.glob(args.dataset)
for dataset in datasets:
    print(dataset)
    try:
        volume, error, fs, t = test(dataset, N, et, rt, upsampling_rate)
        if error > 0:
            print(error)
            output[volume].append((error, t))
    except:
        tb = sys.exc_info()[2]
        print('error:', tb)

if not args.o is None:
    fname = args.o
else:
    fname = 'results_dictionary'

with open(fname, 'wb') as f:
    pickle.dump(output, f)
