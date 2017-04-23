import cvxpy as cvx
import numpy as np
import pudb
from scipy.spatial import ConvexHull

from ..utils.fwdutils import checkFeasibility



class RelativeDualityGap():
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
        self.A = np.nan
        self.b = np.nan
        
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
        
        # for relative duality gap, need to check if b != 0
        check_bEqualsZero = ( np.round(self.b - 0, self.tol) == 0 ).all()
        if check_bEqualsZero:
            print ('b vector equals 0. Relative duality gap is inappropriate.')
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
        """ Solves a linear program.

            min     sum z_q
            st      z_q >= e_q - 1
                    z_q >= 1 - e_q
                    A'y = c
                    c'x_q = e_q
                    b'y = 1
                    y >= 0
        """
        points = [ np.mat(point).T for point in points ]
        if self._forwardModel == False:
            print('No forward model set.')
            sys.exit()
        #feasible = checkFeasibility(points, self.A, self.b, self.tol)
        self.error = self._solveRelativeDGLP(points)
        return True

    def _solveRelativeDGLP(self, points):
        """ Solves a linear program.

            min     sum z_q
            st      z_q >= c'x_q - 1
                    z_q >= 1 - c'x_q 
                    A'y = c
                    b'y = 1
                    y >= 0
        """
        m,n = self.A.shape
        nPoints = len(points)

        y = cvx.Variable(m)
        z = cvx.Variable(nPoints)
        c = cvx.Variable(n)

        #pu.db
        obj = cvx.Minimize(sum(z))

        cons = []
        cons.append( y >= 0 )
        cons.append( self.A.T * y == c )
        if ( self.b <= 0 ).all():
            cons.append ( y.T * self.b == -1 )
        else:
            cons.append( y.T * self.b == 1 )
        for i, point in enumerate(points):
            cons.append( z[i] >= c.T * point - 1 )
            cons.append( z[i] >= 1 - c.T * point )
        
        prob = cvx.Problem(obj, cons)
        result = prob.solve()

        self.c = c.value / np.linalg.norm(c.value, np.inf)
        self.c = self.c.T.tolist()[0]
        self.dual = y.value / np.linalg.norm(c.value, np.inf)
        self.dual = self.dual.T.tolist()[0]
        self.solved = True
        self.error = result
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

