""" Absolute Duality Gap Inverse Optimization

The absolute duality gap method for inverse optimization minimizes the aggregate
duality gap between the primal and dual objective values for each observed
decision. The problem is formulated as follows

.. math::
    
    \min_{\mathbf{c, y},\epsilon_1, \dots, \epsilon_Q} \quad  & \sum_{q=1}^Q | \epsilon_q |

         \\text{s.t.}\quad\quad  & \mathbf{A'y = c} 
         
         & \mathbf{c'\hat{x}_q = b'y} + \epsilon_q, \quad \\forall q
         
         & \| \mathbf{c} \|_1 = 1
         
         & \mathbf{y \geq 0}
"""
import cvxpy as cvx
import numpy as np
#import pudb

from ..utils.invoutils import checkFeasibility



class AbsoluteDualityGap():
    """ Formulate an Absolute Duality Gap method of GMIO. 
        
    Args:
        tol (int): Sets number of significant digits. Default is 8.
        verbose (bool): Sets displays.  Default False. 
    
    Example:
        Suppose that the variables ``A`` and ``b`` are numpy matrices and ``points`` is
        a list of numpy arrays::

           model = AbsoluteDualityGap()
           model.FOP(A, b)
           model.solve(points)
           print (model.c)
    """

    def __init__(self, **kwargs):
        self._fop = False
        self._verbose = False
        self._solved = False
        self.tol = 8
        self.solver = cvx.ECOS_BB
        self._kwargs = self._initialize_kwargs(kwargs)
        
    def FOP(self, A, b):
        """ Create a forward optimization problem.
        
        Args:
            A (matrix): numpy matrix of shape :math:`m \\times n`.
            b (matrix): numpy matrix of shape :math:`m \\times 1`.

        Currently, the forward problem is constructed by the user supplying a
        constraint matrix ``A`` and vector ``b``. The forward problem is

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
        
        First check if all of the points are feasible, in which case we can
        just project the points to each of the hyperplanes. Let :math:`\\bar{x}`
        denote the centroid of the points. Then, we just solve

        .. math::

            \min_{i \in \mathcal{M}} \left\{ \\frac{\mathbf{a_i'\\bar{x} - }b_i }{\| \mathbf{a_i} \|_1} \\right\}

        Let :math:`i^*` denote the optimal index. The optimal cost and dual 
        variables are

        .. math::

            \mathbf{c^*} &= \mathbf{\\frac{a_{i^*}}{\|a_{i^*}\|}}

            \mathbf{y^*} &= \mathbf{\\frac{e_{i^*}}{\|a_{i^*}\|}}
        
        If not all of the points are feasible, then we need to solve an
        exponential number of optimization problems. Let :math:`\mathcal{C}^+, \mathcal{C}^- \subseteq \{ 1, \dots, n \}`
        be a partition of the index set of length ``n``. For each possible
        partition, we solve the following problem

        .. math::

            \min_{\mathbf{c, y}, \epsilon_1,\dots,\epsilon_Q} \quad  & \sum_{q=1}^Q | \epsilon_q |

            \\text{s.t.} \quad  & \mathbf{A'y = c}
            
            & \mathbf{c'\hat{x}_q = b'y} + \epsilon_q, \quad \\forall q 
            
            & \sum_{i \in \mathcal{C}^+} c_i + \sum_{i \in \mathcal{C}^-} c_i = 1 
            
            & c_i \geq 0, \quad i \in \mathcal{C}^+
            
            & c_i \leq 0, \quad i \in \mathcal{C}^-
            
            & \mathbf{y \geq 0}
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
        nFormulations = 2 ** n 
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
            result = prob.solve(solver=self.solver)

            if result < bestResult:
                bestResult = result
                self.c = c.value / np.linalg.norm(c.value, 1)
                self.dual = y.value / np.linalg.norm(c.value, 1)
        self._solved = True
        self.error = bestResult
        self.dual = self.dual.T.tolist()[0] # reconvert to just a list
        self.c = self.c.T.tolist()[0]
        return self.error 

    def rho(self, points):
        """ Solves the goodness of fit.
        """
        assert self._solved, 'you need to solve first.'

        m,n = self.A.shape
        numer = [ np.abs(np.dot(self.c, point) - np.dot(self.dual, self.b)) for point in points ]
        numer = sum(numer)
        denom = 0 
        for i in range(m):
            denomTerm = [ np.abs(np.dot(self.A[i], point) - self.b[i]) for point in points ]
            denom += sum(denomTerm)
        rho = 1 - numer/denom
        return rho[0,0]
        
    def _initialize_kwargs(self, kwargs):
        if 'verbose' in kwargs:
            assert isinstance(kwargs['verbose'], bool), 'verbose needs to be True or False.'
            self._verbose = kwargs['verbose']
        if 'tol' in kwargs:
            assert isinstance(kwargs['tol'], int), 'tolerance needs to be an integer.'
            self.tol = kwargs['tol']
        if 'solver' in kwargs:
            if kwargs['solver'] in cvx.installed_solvers():
                self.solver = getattr(cvx, kwargs['solver'])                
            else:
                print ('you do not have this solver.')
        
        return kwargs

