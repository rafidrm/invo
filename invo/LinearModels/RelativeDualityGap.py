import cvxpy as cvx
import numpy as np
#import pudb

from ..utils.invoutils import checkFeasibility



class RelativeDualityGap():
    """ Formulate an Absolute Duality Gap method of generalized linear inverse 
    optimization. 
    
    Args:
        tol (int): Sets number of significant digits. Default is 8. 
        verbose (bool): Sets displays.  Default False. 
    """

    def __init__(self, **kwargs):
        """
        """
        # initialize kwargs
        self._fop = False
        self._solved = False
        self._verbose = False
        self.tol = 8
        self._kwargs = self._initialize_kwargs(kwargs)

    def FOP(self, A, b):
        """ Creates a forward optimization problem, by defining the feasible 
        set P = { x : Ax >= b }.

        Args:
            A (matrix): numpy array.
            b (matrix): numpy array.
        """
        self.A = np.mat(A)
        self.b = np.mat(b)
        self._fop = True

    def solve(self, points):
        """ Solves the inverse optimization problem, by reformulating it to the
        following LP.

        min     sum z_q
        st      z_q >= e_q - 1
        z_q >= 1 - e_q
        A'y = c
        c'x_q = e_q
        b'y = 1
        y >= 0
        """
        points = [ np.mat(point).T for point in points ]
        assert self._fop, 'No forward model given.'
        self.error = self._solveRelativeDGLP(points)
        return self.error 

    def _solveRelativeDGLP(self, points):
        """ Solves a linear program.

            min sum z_q
            st  [ z_q >= c'x_q - 1
            z_q >= 1 - c'x_q 
            A'y = c
            b'y = 1
            y >= 0 ]
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
        self._solved = True
        return result

    def _initialize_kwargs(self, kwargs):
        if 'verbose' in kwargs:
            assert isinstance(kwargs['verbose'], bool), 'verbose needs to be True or False.'
            self._verbose = kwargs['verbose']
        if 'tol' in kwargs:
            assert isinstance(kwargs['tol'], int), 'tolernace needs to be an integer.'
            self.tol = kwargs['tol']

        return kwargs

