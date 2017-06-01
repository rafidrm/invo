""" Relative Duality Gap Inverse Optimization

The relative duality gap uses the ratio of the primal objective value over the 
dual objective value of the forward problem  as the measure of sub-optimality.
This inverse optimization problem is formulated as 

.. math::
    
    \min_{\mathbf{c, y},\epsilon_1, \dots, \epsilon_Q} \quad  & \sum_{q=1}^Q | \epsilon_q - 1 |

    \\text{s.t.}\quad\quad  & \mathbf{A'y = c} 
         
    & \mathbf{c'\hat{x}_q = b'y} \epsilon_q
         
    & \| \mathbf{c} \|_1 = 1
         
    & \mathbf{y \geq 0}
"""
import cvxpy as cvx
import numpy as np
import pudb

from ..utils.invoutils import checkFeasibility



class RelativeDualityGap():
    """ Formulate an Absolute Duality Gap method of generalized linear inverse 
    optimization. 
    
    Args:
        tol (int): Sets number of significant digits. Default is 8. 
        verbose (bool): Sets displays.  Default False. 
    
    Example:
        Suppose that the variables ``A`` and ``b`` are numpy matrices and ``points`` is
        a list of numpy arrays::

           model = RelativeDualityGap()
           model.FOP(A, b)
           model.solve(points)
           print (model.c)
    """

    def __init__(self, **kwargs):
        """
        """
        # initialize kwargs
        self._fop = False
        self._solved = False
        self._verbose = False
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
        
        To solve a relative duality gap problem, we solve the three following
        optimization problems.

        .. math::

            \min_{\mathbf{c, y},\epsilon_1, \dots, \epsilon_Q} \quad  & \sum_{q=1}^Q | \epsilon_q - 1 |

            \\text{s.t.}\quad\quad  & \mathbf{A'y = c} 
                 
            & \mathbf{c'\hat{x}_q = } \epsilon_q
                 
            & \mathbf{b'y} = 1
                 
            & \mathbf{y \geq 0}

        .. math::

            \min_{\mathbf{c, y},\epsilon_1, \dots, \epsilon_Q} \quad  & \sum_{q=1}^Q | \epsilon_q - 1 |

            \\text{s.t.}\quad\quad  & \mathbf{A'y = c} 
                 
            & \mathbf{c'\hat{x}_q = } -\epsilon_q
                 
            & \mathbf{b'y} = -1
                 
            & \mathbf{y \geq 0}

        .. math::

            \min_{\mathbf{c, y},\epsilon_1, \dots, \epsilon_Q} \quad  & 0 

            \\text{s.t.}\quad\quad  & \mathbf{A'y = c} 
                 
            & \mathbf{c'\hat{x}_q =} 0 
                 
            & \mathbf{b'y} = 0

            & \mathbf{y'1} = 0
                 
            & \mathbf{y \geq 0}
        
        The optimal value of the relative duality gap problem is equal to the 
        optimal value of the minimum of these problems. Let :math:`\mathbf{\hat{c}, \hat{y}}`
        denote the optimal solution of that corresponding problem. Then, the 
        optimal solution of the relative duality gap problem is

        .. math::

            \mathbf{c^*} &= \mathbf{\\frac{\hat{c}}{\|\hat{c}\|_1}}

            \mathbf{y^*} &= \mathbf{\\frac{\hat{y}}{\|\hat{c}\|_1}}
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
        result1 = prob1.solve(solver=self.solver)

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
        result2 = prob2.solve(solver=self.solver)

        # then solve the zero variant
        y3 = cvx.Variable(m)
        c3 = cvx.Variable(n)
        obj3 = cvx.Minimize(0)

        cons3 = []
        cons3.append( y3 >= 0 )
        cons3.append( self.A.T * y3 == c3 )
        cons3.append( y3.T * self.b == 0 )
        allOnes = np.mat(np.ones(15)).T
        cons3.append( y3.T * allOnes == 1 )
        for point in points:
            cons3.append( c3.T * point == 0 )
        
        prob3 = cvx.Problem(obj3, cons3)
        result3 = prob3.solve(solver=self.solver)

        optimalReform = np.argmin([ result1, result2, result3 ])
        if optimalReform == 0:
            self.c = c1.value / np.linalg.norm(c1.value, 1)
            self.dual = y1.value / np.linalg.norm(y1.value, 1)
        elif optimalReform == 1:
            self.c = c2.value / np.linalg.norm(c2.value, 1)
            self.dual = y2.value / np.linalg.norm(y2.value, 1)
        elif optimalReform == 2:
            self.c = c3.value / np.linalg.norm(c3.value, 1)
            self.dual = y3.value / np.linalg.norm(y3.value, 1)

        self.c = self.c.T.tolist()[0]
        self.dual = self.dual.T.tolist()[0]
        self._solved = True
        return np.min([result1, result2, result3])

    def rho(self, points):
        """ Solves the goodness of fit.
        """
        assert self._solved, 'you need to solve first.'

        m,n = self.A.shape
        #numer = [ np.abs(np.dot(self.c, point) - np.dot(self.dual, self.b)) / np.abs(np.dot(self.dual, self.b)) for point in points ]
        numer = [ np.abs(np.dot(self.c, point) - 1) for point in points ]
        numer = sum(numer)
        denom = 0
        for i in range(m):
            #denomTerm = [ np.abs(np.dot(self.A[i], point) - self.b[i]) / np.abs(self.b[i]) for point in points ]
            denomTerm = [ np.abs(np.dot(self.A[i], point) - 1) for point in points ]
            denom += sum(denomTerm)
        rho = 1 - numer/denom
        return rho[0,0]

    def _initialize_kwargs(self, kwargs):
        if 'verbose' in kwargs:
            assert isinstance(kwargs['verbose'], bool), 'verbose needs to be True or False.'
            self._verbose = kwargs['verbose']
        if 'tol' in kwargs:
            assert isinstance(kwargs['tol'], int), 'tolernace needs to be an integer.'
            self.tol = kwargs['tol']
        if 'solver' in kwargs:
            if kwargs['solver'] in cvx.installed_solvers():
                self.solver = getattr(cvx, kwargs['solver'])                
            else:
                print ('you do not have this solver.')

        return kwargs

