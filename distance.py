import numpy as np
import itertools

def compute_sources_and_receivers(distance_data, dim):
    # number of sources and receivers
    M,N = distance_data.shape

    # construct D matrix
    D = distance_data**2

    # reconstruct S and R matrix up to a transformation
    U,si,V_h = np.linalg.svd(D)
    R_hat = np.mat(U[:,:dim].T)
    S_hat = np.mat(np.eye(dim)*si[:dim]) * np.mat(V_h[:dim,:])

    hr = np.ones((1,N)) * np.linalg.pinv(S_hat)
    I = np.eye(4)
    zeros = np.zeros((4,1))
    Hr = np.bmat('hr; zeros I')
    R_hatHr = (R_hat.T * np.linalg.inv(Hr)).H

    hs = np.linalg.pinv(R_hatHr).H * np.ones((M,1))
    zeros = np.zeros((1,4))
    Hs = np.bmat('I; zeros')
    Hs = np.linalg.inv(np.bmat('Hs hs'))

    S_prime = Hs*Hr*S_hat

    A = np.array(S_prime[4,:])
    XYZ = np.array(S_prime[1:4,:])
    X = np.array(S_prime[1,:])
    Y = np.array(S_prime[2,:])
    Z = np.array(S_prime[3,:])

    qq = np.vstack( (np.ones((1,N)), 2*XYZ, XYZ**2, 2*X*Y, 2*X*Z, 2*Y*Z) ).T
    q = np.linalg.pinv(qq).dot(A.T)
    Q = np.vstack( (np.hstack( (np.squeeze(q[:4].T), -0.5) ), np.hstack([q[1], q[4], q[7], q[8], 0]), np.hstack([q[2], q[7], q[5], q[9], 0]), np.hstack([q[3],q[8],q[9],q[6],0]), np.array([-0.5,0,0,0,0]) ) )

    if np.all(np.linalg.eigvals(Q[1:4,1:4]) > 0):
        C = np.linalg.cholesky(Q[1:4,1:4]).T
    else:
        C = np.eye(3)

    Hq = np.vstack((  np.array([1,0,0,0,0]),
                      np.hstack( (np.zeros((3,1)), C, np.zeros((3,1)))),
                      np.hstack( (-q[0], -2*np.squeeze(q[1:4].T), 1))
                    ))

    H = np.mat(Hq) * Hs * Hr
    Se = (H*S_hat)[1:4,:]
    Re = 0.5 * (np.linalg.inv(H).H*R_hat)[1:4,:]

    return Re, Se


def rigid_transform_3D(A, B):
    assert len(A) == len(B)

    N = A.shape[0]; # total points

    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)

    # centre the points
    AA = A - np.tile(centroid_A, (N, 1))
    BB = B - np.tile(centroid_B, (N, 1))

    H = np.transpose(AA) * BB

    U, S, Vt = np.linalg.svd(H)

    R = Vt.T * U.T

    # special reflection case
    if np.linalg.det(R) < 0:
        R = Vt.T * U.T

    t = -R*centroid_A.T + centroid_B.T


    return R, t

def pair_iterator(list_of_lists):
    copy_list_of_lists = list(list_of_lists)
    for set1 in reversed(list_of_lists):
        copy_list_of_lists.pop()
        for set2 in copy_list_of_lists:
            for pair in itertools.product(set1,set2):
                yield pair

class DistanceData:
    
    def __init__(self, S, E):
        self.S = S
        self.E = E

    def find_images(self, r):
        results = {}
        for (E0, E1) in pair_iterator(self.E):
            data_hat = np.hstack( (np.array(self.S).T,(E0[1:,:]).T ,(E1[1:,:]).T) )
            error_r, s_est = self._compute_coordinates(data_hat, r)
            results[error_r] = (s_est, (E0, E1))
        return results

    def _compute_coordinates(self, data, r):
        """
        Test whether the input data can be used for localization of sources.

        data    distance data between N sources and M known receivers
        r       coordinates of M receivers

        returns (valid,r_est)

            valid   if True, r_est and s_est are correct estimations
            r_est   estimation of receiver locations
        """
        M, N = data.shape
        Re, Se = compute_sources_and_receivers(data, 5)
        # find transformation between R and Re
        R_r,t_r = rigid_transform_3D(np.mat(Re.T), np.mat(r))
        # find transformation between R and Re
        #Apply transformation on Re and Se to obtain estimated locations of sources
        r_est = np.array(R_r*np.mat(Re) + np.tile(t_r, (1,M)))
        s_est = np.array((-1*R_r)*np.mat(Se) + np.tile(t_r, (1,N)))

        error_r = np.linalg.norm(r_est.T - r, ord='fro') / np.sqrt(M)

        return error_r, s_est
