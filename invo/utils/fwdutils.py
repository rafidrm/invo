import numpy as np
from scipy.spatial import ConvexHull



def fit_convex_hull(points):
    """ Creates a feasible set by taking a convex hull of the points given. Returns P = { x : Ax >= b }

    Args:
        points (list): Set of numpy points.

    Returns:
        A (numpy): constraint matrix
        b (numpy): constraint vector
    """
    hull = ConvexHull(points)
    m,n = hull.equations.shape
    A = -1 * hull.equations[:,0:n-1]
    b = hull.equations[:,n-1]
    return np.mat(A), np.mat(b).T


