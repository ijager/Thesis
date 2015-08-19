import numpy as np
import multiprocessing as mp
import itertools
import networkx as nx
import sys

def find_sorting_indices(x, Y):
    sort_i = np.zeros(len(x), dtype='int')
    for i,el in enumerate(x):
            sort_i[i] = np.where(el == Y)[0][0]
    return sort_i

def calc_rank(D, echo_set, t):
    D[-1,0:-1] = np.array(echo_set).reshape(1,len(echo_set))**2
    D[0:-1,-1] = np.array(echo_set).reshape(len(echo_set),1)**2
    rank = np.linalg.matrix_rank(D, t)
    return (rank, echo_set)

class EchoData:
    
    def __init__(self, data):
        self.data = data
        
    def find_labels(self, D, threshold=0.045, verbose=False, parallel=False):
        E = []
        S = []
        n = 6
        if verbose==True:
            print('Finding echo_set candidates per measurement ...')
        S.append(self.data[:,0])
        t = threshold
        Ei = np.array([])
        while len(Ei) < 1:
            Ci = self._get_candidates(D, t=t, parallel=parallel)
            if verbose == True:
                print('prefilter threshold:', t)
            t *= 2
            if len(Ci) > 100 or t > 8*threshold:
                break
            Ei = self._get_unique_sets(Ci, n=n)
        if 0 < len(Ei) < 20:
            u = np.unique(Ci[:,0])
            if len(u) >= n:
                for ei in Ei:
                    sort_i = find_sorting_indices(u, ei)
                    ei_sorted = ei[sort_i]
                    E.append(np.r_[self.data[:,0][np.newaxis,:], ei_sorted])
        sys.stdout.flush()
        if verbose==True:
            print('Number of unique sets of',n,'echo_sets:',len(E))
        return S,E


    def _get_unique_sets(self, C, n=6):
        G = nx.Graph()
        edge_list = []
        for i in range(len(C)):
            G.add_node(i)
            for j in range(i+1, len(C)):
                if np.any(C[i] - C[j] == 0):
                    edge_list.append((i,j))
        G.add_edges_from(edge_list)
        H = nx.complement(G)
        cliques = nx.find_cliques(H)
        return [C[l] for l in cliques if len(l) == n]

    
    def _get_candidates(self, D, t=0.0002, parallel=False):
        """
        Filters out all non-feasible combinations of echoes based on 
        the rank test of the augmented Euclidean Distance Matrix D. If 
        matrix D after augmenting with echo-data still passes the rank 
        test, then the current set of echoes is saved.

        D       Euclidean distance matrix of size (6,6)
        t       rank test threshold

        returns list of candidate echo sets
        """
        
        if parallel==True:
            echo_sets = [echo_set for echo_set 
                    in itertools.product(*self.data[:,1:])]

            with mp.Pool(processes=mp.cpu_count()) as pool:
                output = pool.starmap(calc_rank, zip(itertools.repeat(D), 
                    echo_sets, itertools.repeat(t)))

            ranks, echo_sets = zip(*output)
            candidates = [echo_set for rank, echo_set in output if rank < 6]            
            #i = np.where(np.array(ranks) < 6)[0]
            #candidates = np.array(echo_sets)[i]

        else:
            candidates = []
            for echo_set in itertools.product(*self.data[:,1:]):
                D[-1,0:-1] = np.array(echo_set).reshape(1,len(echo_set))**2
                D[0:-1,-1] = np.array(echo_set).reshape(len(echo_set),1)**2
                rank = np.linalg.matrix_rank(D, t)
                if rank < 6:
                    candidates.append(echo_set)
        return np.array(candidates)
