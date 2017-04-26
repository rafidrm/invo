import cvxpy as cvx
import numpy as np
#import pudb




class pNorm():
    """ Formulate an Absolute Duality Gap method of GMIO.

    Args:
        tol (int): Sets number of significant digits. Default is 8.
        p (int): Sets p for lp norm. Can be integer or 'inf'. Default is 2.
        verbose (bool): Sets displays.  Default is False. 
    """

    def __init__(self, **kwargs):
        self._fop = False
        self._verbose = False
        self._solved = False
        self.p = 2
        self.tol = 8
        self._kwargs = self._initialize_kwargs(kwargs)

    def FOP(self, A, b):
        """ Creates a forward optimization problem, by defining the feasible 
        set P = { x | Ax >= b }.

        Args:
            A (matrix): numpy array.
            b (matrix): numpy array.
        """
        self.A = np.mat(A)
        self.b = np.mat(b)
        self._fop = True

    def solve(self, points):
        """ Solves the inverse optimization problem, by calculating the 
        feasible projection.
        """
        points = [ np.mat(point).T for point in points ]
        assert self._fop, 'No forward model given.'
        self.error = self._solveFeasibleProjection(points)
        return self.error 

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
        self._solved = True
        #self.dual = self.dual.T.tolist()[0]
        self.c = self.c.tolist()[0]
        self.error = bestResult
        return result 
        
    def _initialize_kwargs(self, kwargs):
        # common kwargs
        if 'verbose' in kwargs:
            assert isinstance(kwargs['verbose'], bool), 'verbose needs to be True or False.'
            self._verbose = kwargs['verbose']
        if 'tol' in kwargs:
            assert isinstance(kwargs['tol'], int), 'tolerance needs to be an integer.'
            self.tol = kwargs['tol']
        
        # class specific kwargs
        if 'p' in kwargs:
            assert isinstance(kwargs['p'], int) or kwargs['p'] is 'inf', 'p needs to be an integer'
            self.p = kwargs['p']
        
        return kwargs

