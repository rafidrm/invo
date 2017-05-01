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

        # first solve the positive variant
        y1 = cvx.Variable(m)
        z1 = cvx.Variable(nPoints)
        c1 = cvx.Variable(n)
        obj1 = cvx.Minimize(sum(z1))

        cons1 = []
        cons1.append( y1 >= 0 )
        cons1.append( self.A.T * y1 == c1 )
        cons1.append( y1.T * self.b == 1 )
        for i, point in enumerate(points):
            cons1.append( z1[i] >= c1.T * point - 1 )
            cons1.append( z1[i] >= 1 - c1.T * point )
        
        prob1 = cvx.Problem(obj1, cons1)
        result1 = prob1.solve()

        # then solve the negative variant
        y2 = cvx.Variable(m)
        z2 = cvx.Variable(nPoints)
        c2 = cvx.Variable(n)
        obj2 = cvx.Minimize(sum(z2))

        cons2 = []
        cons2.append( y2 >= 0 )
        cons2.append( self.A.T * y2 == c2 )
        cons2.append( y2.T * self.b == -1 )
        for i, point in enumerate(points):
            cons2.append( z2[i] >= c2.T * point + 1 )
            cons2.append( z2[i] >= -1 - c2.T * point )
        
        prob2 = cvx.Problem(obj2, cons2)
        result2 = prob2.solve()

        if result1 < result2:
            self.c = c1.value / np.linalg.norm(c1.value, 1)
            self.dual = y1.value / np.linalg.norm(c1.value, 1)
        else:
            self.c = c2.value / np.linalg.norm(c2.value, 1)
            self.dual = y2.value / np.linalg.norm(c2.value, 1)

        self.c = self.c.T.tolist()[0]
        self.dual = self.dual.T.tolist()[0]
        self._solved = True
        return min(result1, result2)

    def _initialize_kwargs(self, kwargs):
        if 'verbose' in kwargs:
            assert isinstance(kwargs['verbose'], bool), 'verbose needs to be True or False.'
            self._verbose = kwargs['verbose']
        if 'tol' in kwargs:
            assert isinstance(kwargs['tol'], int), 'tolernace needs to be an integer.'
            self.tol = kwargs['tol']

        return kwargs

