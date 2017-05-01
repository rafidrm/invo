import cvxpy as cvx
import numpy as np
#import pudb

from ..utils.invoutils import checkFeasibility



class AbsoluteDualityGap():
    """ Formulate an Absolute Duality Gap method of GMIO. 
        
    Args:
        tol (int): Sets number of significant digits. Default is 8.
        verbose (bool): Sets displays.  Default False. 
    """

    def __init__(self, **kwargs):
        self._fop = False
        self._verbose = False
        self._solved = False
        self.tol = 8
        self._kwargs = self._initialize_kwargs(kwargs)
        
    def FOP(self, A, b):
        """ Creates a forward optimization probelm, by defining the feasible 
        set P = { x : Ax >= b }.

        Args:
            A (matrix): numpy array.
            b (matrix): numpy array.
        """
        self.A = np.mat(A)
        self.b = np.mat(b)
        self._fop = True

    def solve(self, points):
        """ Solves the inverse optimization problem. First check if all of the 
        points are feasible, in which case, can solve m linear programs. 
        Otherwise, solve 2^n linear programs with the enumeration method.
        """
        points = [ np.mat(point).T for point in points ]
        assert self._fop, 'No forward model given.'
        feasible = checkFeasibility(points, self.A, self.b, self.tol)
        if feasible:
            self.error = self._solveHyperplaneProjection(points)
        else:
            self.error = self._solveBruteForce(points)
        return self.error 

    def _solveHyperplaneProjection(self, points):
        m,n = self.A.shape
        errors = np.zeros(m)
        for i in range(m):
            ai = self.A[i] / np.linalg.norm(self.A[i], np.inf)
            bi = self.b[i] / np.linalg.norm(self.A[i], np.inf)
            errors[i] = np.sum([ ai * pt - bi for pt in points ])
        minInd = np.argmin(errors)
        self.c = self.A[minInd] / np.linalg.norm(self.A[minInd], np.inf)
        self.c = self.c.tolist()[0]
        self.error = errors[minInd]
        self.dual = np.zeros(m)
        self.dual[minInd] = 1 / np.linalg.norm(self.A[minInd], np.inf)
        self._solved = True
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
                chi = self.A * points[i] - self.b
                cons.append( z[i] >= y.T * chi )
                cons.append( z[i] >= -1 * y.T * chi )
            prob = cvx.Problem(obj, cons)
            result = prob.solve()

            if result < bestResult:
                bestResult = result
                self.c = c.value / np.linalg.norm(c.value, 1)
                self.dual = y.value / np.linalg.norm(c.value, 1)
        self._solved = True
        self.error = bestResult
        self.dual = self.dual.T.tolist()[0] # reconvert to just a list
        self.c = self.c.T.tolist()[0]
        return result 
        
    def _initialize_kwargs(self, kwargs):
        if 'verbose' in kwargs:
            assert isinstance(kwargs['verbose'], bool), 'verbose needs to be True or False.'
            self._verbose = kwargs['verbose']
        if 'tol' in kwargs:
            assert isinstance(kwargs['tol'], int), 'tolerance needs to be an integer.'
            self.tol = kwargs['tol']
        
        return kwargs

