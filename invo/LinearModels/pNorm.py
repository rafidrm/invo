""" Decision Space Inverse Optimization with a p-norm.

These models measure error in the space of decision variables, rather than 
objective values. In particular, these models aim to identify a cost vector that
induces optimal decisions for the forward problem that are of minimum aggregate 
distance to the corresponding observed decisions. Any norm can be used in the 
decision space, but the key distinction is that the imputed optimal decisions 
must be feasible for the forward problem. A decision space inverse optimization
problem is formulated as

.. math::

    \min_{\mathbf{c, y}, \\boldsymbol{ \epsilon_1,} \dots, \\boldsymbol{ \epsilon_Q }} \quad  & \sum_{q=1}^Q \| \\boldsymbol{\epsilon_q} \|
    
    \\text{s.t.}\quad\quad  & \mathbf{A'y = c}
    
    & \mathbf{c'\hat{x}_q = b'y} + \\boldsymbol{\epsilon_q}, \quad \\forall q 
    
    & \mathbf{A ( \hat{x}_q -} \\boldsymbol{ \epsilon_q } \mathbf{ ) \geq b}
    
    & \| \mathbf{c} \|_1 = 1 
    
    & \mathbf{y \geq 0}
"""
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
        """ Create a forward optimization problem.
        
        Args:
            A (matrix): numpy matrix of shape :math:`m \\times n`.
            b (matrix): numpy matrix of shape :math:`m \\times 1`.

        Currently, the forward problem is constructed by the user supplying a
        constraint matrix `A` and vector `b`. The forward problem is

        .. math::

            \min_{\mathbf{x}} \quad&\mathbf{c'x}

            \\text{s.t} \quad&\mathbf{A x \geq b}
        """
        self.A = np.mat(A)
        self.b = np.mat(b)
        self._fop = True

    def solve(self, points):
        """ Solves the inverse optimization problem. 
        
        Args:
            points (list): list of numpy arrays, denoting the (optimal) observed points.

        Returns:
            error (float): the optimal value of the inverse optimization problem.
        
        To solve a decision space inverse optimization problem, we solve the 
        following convex problem for every constraint

        .. math::

            \min_{\\boldsymbol{ \epsilon_1,} \dots, \\boldsymbol{ \epsilon_Q }} \quad  & \sum_{q=1}^Q \| \\boldsymbol{\epsilon_q} \|_p
            
            \\text{s.t.} \quad\quad  & \mathbf{ a_i'(x_q - }\\boldsymbol{\epsilon_q}\mathbf{)} = b_i
            
            & \mathbf{ A ( x_q - }\\boldsymbol{\epsilon_q}\mathbf{) \geq b}
        """
        points = [ np.mat(point).T for point in points ]
        assert self._fop, 'No forward model given.'
        self.error = self._solveFeasibleProjection(points)
        return self.error 

    def _solveFeasibleProjection(self, points):
        m,n = self.A.shape
        bestResult = np.inf

        for i in range(m):
            ai = self.A[i]
            bi = self.b[i]
            result = self._project_to_hyperplane(points, ai, bi)

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
    
    def _project_to_hyperplane(self, points, ai, bi):
        """ Helper function that solves feasible projection to a hyperplane
        """
        m,n = self.A.shape
        nPoints = len(points)
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
        return result


    def optimal_points(self, points):
        """ Get the projected optimal points.

        Args:
            points (list): list of numpy arrays, denoting the observed points.

        Returns:
            truePoints (list): list of numpy arrays denoting the imputed optimal points.
        
        Once an inverse optimization problem is solved and the forward model is
        completed, you can take a collection of observed data points and get
        the 'imputed' optimal points.
        """
        if self._solved:
            m,n = self.A.shape
            ind = np.argmax(self.dual)
            ai = self.A[ind]
            bi = self.b[ind]
            epsilons = [ cvx.Variable(n) for pt in points ]
            objFunc = []
            cons = []
            for x,point in enumerate(points):
                objFunc.append( cvx.norm(epsilons[x], self.p) )
                cons.append( self.A * (points[x] - epsilons[x]) >= self.b )
                cons.append( ai * (points[x] - epsilons[x]) == bi )
            obj = cvx.Minimize(sum(objFunc))
            prob = cvx.Problem(obj, cons)
            result = prob.solve()
            xStars = [ (np.mat(point).T - epsilon.value) for point, epsilon in zip(points, epsilons) ]
        else:
            self.solve(points)
            xStars = self.optimal_points(points)
        return xStars

    def rho(self, points):
        """ Solves the goodness of fit.
        """
        assert self._solved, 'you need to solve first.'
        
        m,n = self.A.shape
        projections = self.optimal_points(points)
        _pts = [ np.mat(pt).T for pt in points ]
        numer = [ np.linalg.norm(pj - pt, self.p) for pj, pt in zip(projections, _pts) ]
        numer = sum(numer)
        denom = 0
        for i in range(m):
            ai = self.A[i]
            bi = self.b[i]
            result = self._project_to_hyperplane(points, ai, bi)
            denom += result
        rho = 1 - numer/denom
        return rho
    
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

