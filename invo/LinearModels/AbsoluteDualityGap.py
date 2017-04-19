import cvxpy as cvx
import numpy as np
import pudb
from scipy.spatial import ConvexHull

from ..utils.fwdutils import checkFeasibility



class AbsoluteDualityGap():
    """ Formulate an Absolute Duality Gap method of generalized linear inverse optimization. 
    """

    def __init__(self, **kwargs):
        """
        kwargs:
            forward (str): Method to construct a feasible set. Set it to hull
            verbose (bool): Sets displays.  Default False. 
        """
        # initialize kwargs
        self._kwargs = self._initialize_kwargs(kwargs)
        self._forwardModel = False
        self.solved = False
        self.error = np.nan
        self.dual = np.nan
        
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
        if self._forwardModel == False:
            print('No forward model set.')
            sys.exit()
        feasible = checkFeasibility(points, self.A, self.b, self.tol)
        if feasible:
            self.error = self._solveHyperplaneProjection(points)
        else:
            self.error = self._solveBruteForce(points)
        return True

    def _solveHyperplaneProjection(self, points):
        m,n = self.A.shape
        errors = np.zeros(m)
        for i in range(m):
            ai = self.A[i] / np.linalg.norm(self.A[i], np.inf)
            bi = self.b[i] / np.linalg.norm(self.A[i], np.inf)
            errors[i] = np.sum([ ai * pt - bi for pt in points ])
        minInd = np.argmin(errors)
        self.c = self.A[minInd] / np.linalg.norm(self.A[minInd], np.inf)
        self.error = errors[minInd]
        self.dual = np.zeros(m)
        self.dual[minInd] = 1 / np.linalg.norm(self.A[minInd], np.inf)
        self.solved = True
        return errors[minInd] 

    def _solveBruteForce(self, points):
        m,n = self.A.shape
        nPoints = len(points)
        nFormulations = 2 ** n - 1
        bestResult = np.inf

        for formulation in range(nFormulations):
            binFormulation = format(formulation, '0{}b'.format(n))
            cSign = [ int(i) for i in binFormulation ]
            cSign = np.mat(cSign)
            cSign[cSign == 0] = -1

            y = cvx.Variable(m)
            z = cvx.Variable(nPoints)
            c = cvx.Variable(n)
            obj = cvx.Minimize(sum(z))
            
            cons = []
            cons.append( y >= 0 )
            cons.append( self.A.T * y == c )
            cons.append( cSign * c == 1 )
            for i in range(n):
                if cSign[0,i] == 1:
                    cons.append( c[i] >= 0 )
                else:
                    cons.append( c[i] <= 0 )
            for i in range(nPoints):
                chi = self.A * points[:,i] - self.b
                cons.append( z[i] >= y.T * chi )
                cons.append( z[i] >= -1 * y.T * chi )
            prob = cvx.Problem(obj, cons)
            result = prob.solve()

            if result < bestResult:
                bestResult = result
                self.c = c.value / np.linalg.norm(c.value, np.inf)
                self.dual = y.value / np.linalg.norm(c.value, np.inf)
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
            self.tol = kwargs['tol']
        
        return kwargs

