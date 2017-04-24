import cvxpy as cvx
import numpy as np
import pudb
from scipy.spatial import ConvexHull

from ..utils.invoutils import checkFeasibility



class pNorm():
    """ Formulate an Absolute Duality Gap method of generalized linear inverse optimization. 
    """

    def __init__(self, **kwargs):
        """
        kwargs:
            forward (str): Method to construct a feasible set. Set it to hull
            verbose (bool): Sets displays.  Default False. 
        """
        # initialize kwargs
        self._forwardModel = False
        self.solved = False
        self.error = np.nan
        self.dual = np.nan
        self.p = np.nan
        self._kwargs = self._initialize_kwargs(kwargs)
        
        # initialize forward model
        if self._kwargs['forward'] != None:
            self.ForwardModel(self._kwargs['forward'], self._kwargs)

    def ForwardModel(self, forward, **kwargs):
        """ Creates a forward model by user specified parameters. 
        
        Args:
            forward (str): Method to construct a feasible set. Can take values: 'hull', 'poly'.
            points (list): Set of numpy points. Required for hull method.
            A (matrix): numpy array. Required for hull method.
            b (matrix): numpy array. Required for hull method.
        """
        if forward == 'hull':
            self._forward_set = self._fitConvexHull(kwargs['points'])
        elif forward == 'poly':
            self._forward_set = self._fitPolyhedralConstraints(kwargs['A'], kwargs['b'])
        else:
            print ('Could not fit forward model!')
            sys.exit()
        self._forwardModel = True

    def _fitConvexHull(self, points):
        """ Create a feasible set by taking a convex hull of the points given.

        Args:
            points (list): Set of numpy points.
        """
        hull = ConvexHull(points)
        m,n = hull.equations.shape
        self.A = -1 * hull.equations[:,0:n-1]
        self.A = np.mat(self.A)
        self.b = hull.equations[:,n-1]
        self.b = np.mat(self.b).T
        return True

    def _fitPolyhedralConstraints(self, A, b):
        """ User can input A, b such that P = { x | Ax >= b }.

        Args:
            A (matrix): numpy array.
            b (matrix): numpy array.
        """
        self.A = np.mat(A)
        self.b = np.mat(b)
        return True

    def solve(self, points):
        """ 
        """
        points = [ np.mat(point).T for point in points ]
        assert self._forwardModel, 'No forward model given.'
        self.error = self._solveFeasibleProjection(points)
        return True

    def _solveFeasibleProjection(self, points):
        m,n = self.A.shape
        nPoints = len(points)
        bestResult = np.inf

        for i in range(m):
            ai = self.A[i]
            bi = self.b[i]
            
            epsilons = [ cvx.Variable(n) for pt in points ]
            objFunc = []
            cons = []
            for x in range(nPoints):
                objFunc.append( cvx.norm(epsilons[x], self.p) )
                cons.append( self.A * (points[x] - epsilons[x]) >= self.b )
                cons.append( ai * (points[x] - epsilons[x]) == bi )
            obj = cvx.Minimize(sum(objFunc))
            prob = cvx.Problem(obj, cons)
            result = prob.solve()

            if result < bestResult:
                bestResult = result
                self.dual = np.zeros(m)
                self.dual[i] = 1.0 / np.linalg.norm(ai, np.inf)
                self.c = ai / np.linalg.norm(ai, np.inf)
        self.solved = True
        self.error = bestResult
        return result 
        
    def _initialize_kwargs(self, kwargs):
        if 'forward' not in kwargs:
            kwargs['forward'] = None
        if 'verbose' not in kwargs:
            kwargs['verbose'] = False
        if 'tol' not in kwargs:
            self.tol = 8 
        else:
            assert isinstance(kwargs['tol'], int), 'tolerance needs to be an integer.'
            self.tol = kwargs['tol']
        
        # class specific kwargs
        if 'p' not in kwargs:
            self.p = 2
        else:
            assert isinstance(kwargs['p'], int) or kwargs['p'] is 'inf', 'p needs to be an integer'
            self.p = kwargs['p']
        
        return kwargs

