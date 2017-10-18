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

from ..utils.invoutils import checkFeasibility, validateFOP


class RelativeDualityGap():
    """ Formulate an Absolute Duality Gap method of generalized linear inverse
    optimization.

    Args:
        tol (int): Sets number of significant digits. Default is 8.
        verbose (bool): Sets displays.  Default False.
        normalize_c: Set to either 1 or np.inf. Decides the normalization constraint on c
        ban_constraints (list): A list of constraint indices to force to zero when solving. Default is none.

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
        self.ban_constraints = []
        self.normalize_c = 1
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
        #self.A = np.mat(A)
        #self.b = np.mat(b)
        self.A, self.b = validateFOP(A, b)
        self._fop = True

    def solve(self, points, **kwargs):
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
        self._kwargs = self._initialize_kwargs(kwargs)

        points = [np.mat(point).T for point in points]
        assert self._fop, 'No forward model given.'
        self.error = self._solveRelativeDGLP(points)
        if (np.round(self.c, self.tol) == 0).all():
            # the relative problem returned c = 0 and we have to fix.
            # First we solve the auxiliary problem
            self.K = self._solve_auxiliary_problem()

            #Then we solve the "hard way"
            self.error = self._solveRelativeDG(points)
        return self.error

    def _solve_auxiliary_problem(self):
        """ Solves the auxiliary problem.

            max max(b'y, -b'y, y'1)
            st  ||A'y|| = 1
                y >= 0
        """
        if self.normalize_c == 1:
            res = self._solve_auxiliary_problemNorm1()
        elif self.normalize_c == np.inf:
            res = self._solve_auxiliary_problemNormInf()
        return 1 / res

    def _solve_auxiliary_problemNorm1(self):
        """ Solves the auxiliary problem with || A'y ||_1 = 1
        """
        m, n = self.A.shape
        nFormulations = 2**n
        bestResult = 0

        for formulation in range(nFormulations):
            binFormulation = format(formulation, '0{}b'.format(n))
            cSign = [int(i) for i in binFormulation]
            cSign = np.mat(cSign)
            cSign[cSign == 0] = -1

            y1 = cvx.Variable(m)
            c1 = cvx.Variable(n)
            cons1 = []
            cons1.append(self.A.T * y1 == c1)
            cons1.append(y1 >= 0)
            cons1.append(cSign * c1 == 1)

            for i in range(n):
                if cSign[0, i] == 1:
                    cons.append(c1[i] >= 0)
                else:
                    cons.append(c1[i] <= 0)

            obj1 = cvx.Maximize(self.b.T * y1)
            prob1 = cvx.Problem(obj1, cons2)
            result1 = prob1.solve(solver=self.solver)

            y2 = cvx.Variable(m)
            c2 = cvx.Variable(n)
            cons2 = []
            cons2.append(self.A.T * y2 == c2)
            cons2.append(y2 >= 0)
            cons1.append(cSign * c2 == 1)

            for i in range(n):
                if cSign[0, i] == 1:
                    cons.append(c2[i] >= 0)
                else:
                    cons.append(c2[i] <= 0)

            obj2 = cvx.Maximize(-1 * self.b.T * y2)
            prob2 = cvx.Problem(obj2, cons2)
            result2 = prob2.solve(solver=self.solver)

            y3 = cvx.Variable(m)
            c3 = cvx.Variable(n)
            cons3 = []
            cons3.append(self.A.T * y3 == c3)
            cons3.append(y3 >= 0)
            cons1.append(cSign * c3 == 1)

            for i in range(n):
                if cSign[0, i] == 1:
                    cons.append(c3[i] >= 0)
                else:
                    cons.append(c3[i] <= 0)

            allOnes = np.mat(np.ones(m)).T
            obj3 = cvx.Maximize(y3.T * allOnes)
            prob3 = cvx.Problem(obj3, cons3)
            result3 = prob3.solve(solver=self.solver)

            result = np.max([result1, result2, result3])

            if result > bestResult:
                bestResult = result
        return result

    def _solve_auxiliary_problemNormInf(self):
        """ Solves the auxiliary problem with || A'y ||_infinty = 1
        """
        m, n = self.A.shape
        bestResult = 0

        for j in range(n):
            y1 = cvx.Variable(m)
            c1 = cvx.Variable(n)
            cons1 = []
            cons1.append(self.A.T * y1 == c1)
            cons1.append(y1 >= 0)

            cons1.append(c1[j] == 1)
            cons1.append(c1 <= 1)
            cons1.append(c1 >= -1)

            obj1 = cvx.Maximize(self.b.T * y1)
            prob1 = cvx.Problem(obj1, cons1)
            result1 = prob1.solve(solver=self.solver)

            y2 = cvx.Variable(m)
            c2 = cvx.Variable(n)
            cons2 = []
            cons2.append(self.A.T * y2 == c2)
            cons2.append(y2 >= 0)

            cons2.append(c2[j] == 1)
            cons2.append(c2 <= 1)
            cons2.append(c2 >= -1)

            obj2 = cvx.Maximize(-1 * self.b.T * y2)
            prob2 = cvx.Problem(obj2, cons2)
            result2 = prob2.solve(solver=self.solver)

            y3 = cvx.Variable(m)
            c3 = cvx.Variable(n)
            cons3 = []
            cons3.append(self.A.T * y3 == c3)
            cons3.append(y3 >= 0)

            cons3.append(c3[j] == 1)
            cons3.append(c3 <= 1)
            cons3.append(c3 >= -1)

            allOnes = np.mat(np.ones(m)).T
            obj3 = cvx.Maximize(y3.T * allOnes)
            prob3 = cvx.Problem(obj3, cons3)
            result3 = prob3.solve(solver=self.solver)

            result = np.max([result1, result2, result3])

            if result > bestResult:
                bestResult = result
        return result

    def _baseBruteForceProblem(self, y, z, c):
        obj = cvx.Minimize(sum(z))
        cons = []
        cons.append(y >= 0)
        cons.append(self.A.T * y == c)
        for i in self.ban_constraints:
            cons.append(y[i] == 0)
        return obj, cons

    def _solveRelativeDG(self, points):
        """ Solves the norm constrained version of the problem.

            min sum z_q
            st  z_q >= c'x_q - 1
                z_q >= 1 - c'x_q
                A'y = c
                b'y = 1
                ||c|| = 1
                y >= 0
        """
        if self.normalize_c == 1:
            error = self._solveRelativeDGNorm1(points)
        elif self.normalize_c == np.inf:
            error = self._solveRelativeDGNormInf(points)
        return error

    def _solveRelativeDGNorm1(self, points):
        """ Solves the problem with the l-1 norm.
        """
        m, n = self.A.shape
        nPoints = len(points)
        nFormulations = 2**n
        bestResult = np.inf

        for formulation in range(nFormulations):
            binFormulation = format(formulation, '0{}b'.format(n))
            cSign = [int(i) for i in binFormulation]
            cSign = np.mat(cSign)
            cSign[cSign == 0] = -1

            # first solve the positive variant
            obj1, cons1, y1, z1, c1 = self._positiveProblem(
                m, n, nPoints, points)
            cons1.append(cSign * c1 >= self.K)
            for i in range(n):
                if cSign[0, i] == 1:
                    cons1.append(c1[i] >= 0)
                else:
                    cons1.append(c1[i] <= 0)
            prob1 = cvx.Problem(obj1, cons1)
            result1 = prob1.solve(solver=self.solver)

            # then solve the negative variant
            obj2, cons2, y2, z2, c2 = self._negativeProblem(
                m, n, nPoints, points)
            cons2.append(cSign * c2 >= self.K)
            for i in range(n):
                if cSign[0, i] == 1:
                    cons2.append(c2[i] >= 0)
                else:
                    cons2.append(c2[i] <= 0)
            prob2 = cvx.Problem(obj2, cons2)
            result2 = prob2.solve(solver=self.solver)

            # then solve the zero variant
            obj3, cons3, y3, c3 = self._zeroProblem(m, n, nPoints, points)
            cons3.append(cSign * c3 >= self.K)
            for i in range(n):
                if cSign[0, i] == 1:
                    cons3.append(c3[i] >= 0)
                else:
                    cons3.append(c3[i] <= 0)
            prob3 = cvx.Problem(obj3, cons3)
            result3 = prob3.solve(solver=self.solver)

            best = np.argmin([result1, result2, result3, bestResult])
            if best == 0:
                bestResult = result1
                self.c = c1.value / np.linalg.norm(c1.value, 1)
                self.dual = y1.value / np.linalg.norm(y1.value, 1)
            elif best == 1:
                bestResult = result2
                self.c = c2.value / np.linalg.norm(c2.value, 1)
                self.dual = y2.value / np.linalg.norm(y2.value, 1)
            elif best == 2:
                bestResult = result3
                self.c = c3.value / np.linalg.norm(c3.value, 1)
                self.dual = y3.value / np.linalg.norm(y3.value, 1)
        self._solved = True
        self.error = bestResult
        self.dual = self.dual.T.tolist()[0]  # reconvert to just a list
        self.c = self.c.T.tolist()[0]
        return self.error

    def _solveRelativeDGNormInf(self, points):
        """ Solves the problem with the l-infinity norm.
        """
        m, n = self.A.shape
        nPoints = len(points)
        bestResult = np.inf

        for j in range(n):
            # first solve the positive variant
            obj1p, cons1p, y1p, z1p, c1p = self._positiveProblem(
                m, n, nPoints, points)
            cons1p.append(c1p[j] >= self.K)
            prob1p = cvx.Problem(obj1p, cons1p)
            result1p = prob1p.solve(solver=self.solver)

            obj1n, cons1n, y1n, z1n, c1n = self._positiveProblem(
                m, n, nPoints, points)
            cons1n.append(c1n[j] <= -self.K)
            prob1n = cvx.Problem(obj1p, cons1p)
            result1n = prob1n.solve(solver=self.solver)

            # then solve the negative variant
            obj2p, cons2p, y2p, z2p, c2p = self._negativeProblem(
                m, n, nPoints, points)
            cons2p.append(c2p[j] >= self.K)
            prob2p = cvx.Problem(obj2p, cons2p)
            result2p = prob2p.solve(solver=self.solver)

            obj2n, cons2n, y2n, z2n, c2n = self._negativeProblem(
                m, n, nPoints, points)
            cons2n.append(c2n[j] <= -self.K)
            prob2n = cvx.Problem(obj1p, cons1p)
            result2n = prob2n.solve(solver=self.solver)

            # then solve the zero variant
            obj3p, cons3p, y3p, c3p = self._zeroProblem(m, n, nPoints, points)
            cons3p.append(c3p[j] >= self.K)
            prob3p = cvx.Problem(obj3p, cons3p)
            result3p = prob3p.solve(solver=self.solver)

            obj3n, cons3n, y3n, c3n = self._zeroProblem(m, n, nPoints, points)
            cons3n.append(c3n[j] <= -self.K)
            prob3n = cvx.Problem(obj1p, cons1p)
            result3n = prob3n.solve(solver=self.solver)

            best = np.argmin([
                result1p, result1n, result2p, result2n, result3p, result3n,
                bestResult
            ])
            if best == 0:
                bestResult = result1p
                self.c = c1p.value / np.linalg.norm(c1p.value, np.inf)
                self.dual = y1p.value / np.linalg.norm(y1p.value, np.inf)
            elif best == 1:
                bestResult = result1n
                self.c = c1n.value / np.linalg.norm(c1n.value, np.inf)
                self.dual = y1n.value / np.linalg.norm(y1n.value, np.inf)
            elif best == 2:
                bestResult = result2p
                self.c = c2p.value / np.linalg.norm(c2p.value, np.inf)
                self.dual = y2p.value / np.linalg.norm(y2p.value, np.inf)
            elif best == 3:
                bestResult = result2n
                self.c = c2n.value / np.linalg.norm(c2n.value, np.inf)
                self.dual = y2n.value / np.linalg.norm(y2n.value, np.inf)
            elif best == 4:
                bestResult = result3p
                self.c = c3p.value / np.linalg.norm(c3p.value, np.inf)
                self.dual = y3p.value / np.linalg.norm(y3p.value, np.inf)
            elif best == 5:
                bestResult = result3n
                self.c = c3n.value / np.linalg.norm(c3n.value, np.inf)
                self.dual = y3n.value / np.linalg.norm(y3n.value, np.inf)
        self._solved = True
        self.error = bestResult
        self.dual = self.dual.T.tolist()[0]  # reconvert to just a list
        self.c = self.c.T.tolist()[0]
        return self.error

    def _positiveProblem(self, m, n, nPoints, points):
        y = cvx.Variable(m)
        z = cvx.Variable(nPoints)
        c = cvx.Variable(n)
        obj, cons = self._baseBruteForceProblem(y, z, c)
        cons.append(y.T * self.b == 1)
        for i, point in enumerate(points):
            cons.append(z[i] >= c.T * point - 1)
            cons.append(z[i] >= 1 - c.T * point)
        return obj, cons, y, z, c

    def _negativeProblem(self, m, n, nPoints, points):
        y = cvx.Variable(m)
        z = cvx.Variable(nPoints)
        c = cvx.Variable(n)
        obj, cons = self._baseBruteForceProblem(y, z, c)
        cons.append(y.T * self.b == -1)
        for i, point in enumerate(points):
            cons.append(z[i] >= c.T * point + 1)
            cons.append(z[i] >= -1 - c.T * point)
        return obj, cons, y, z, c

    def _zeroProblem(self, m, n, nPoints, points):
        y = cvx.Variable(m)
        z = cvx.Variable(nPoints)
        c = cvx.Variable(n)
        _, cons = self._baseBruteForceProblem(y, z, c)
        obj = cvx.Minimize(0)
        cons.append(y.T * self.b == 0)
        allOnes = np.mat(np.ones(m)).T
        cons.append(y.T * allOnes == 1)
        for point in points:
            cons.append(c.T * point == 0)
        return obj, cons, y, c

    def _solveRelativeDGLP(self, points):
        """ Solves a linear program.

            min sum z_q
            st  [ z_q >= c'x_q - 1
            z_q >= 1 - c'x_q
            A'y = c
            b'y = 1
            y >= 0 ]
        """
        m, n = self.A.shape
        nPoints = len(points)

        # first solve the positive variant
        obj1, cons1, y1, z1, c1 = self._positiveProblem(m, n, nPoints, points)
        prob1 = cvx.Problem(obj1, cons1)
        result1 = prob1.solve(solver=self.solver)

        # then solve the negative variant
        obj2, cons2, y2, z2, c2 = self._negativeProblem(m, n, nPoints, points)
        prob2 = cvx.Problem(obj2, cons2)
        result2 = prob2.solve(solver=self.solver)

        # then solve the zero variant
        obj3, cons3, y3, c3 = self._zeroProblem(m, n, nPoints, points)
        prob3 = cvx.Problem(obj3, cons3)
        result3 = prob3.solve(solver=self.solver)

        optimalReform = np.argmin([result1, result2, result3])
        if optimalReform == 0:
            self.c = c1.value / np.linalg.norm(c1.value, self.normalize_c)
            self.dual = y1.value / np.linalg.norm(y1.value, self.normalize_c)
        elif optimalReform == 1:
            self.c = c2.value / np.linalg.norm(c2.value, self.normalize_c)
            self.dual = y2.value / np.linalg.norm(y2.value, self.normalize_c)
        elif optimalReform == 2:
            self.c = c3.value / np.linalg.norm(c3.value, self.normalize_c)
            self.dual = y3.value / np.linalg.norm(y3.value, self.normalize_c)

        self.c = self.c.T.tolist()[0]
        self.dual = self.dual.T.tolist()[0]
        self._solved = True
        return np.min([result1, result2, result3])

    def rho(self, points):
        """ Solves the goodness of fit.
        """
        assert self._solved, 'you need to solve first.'

        m, n = self.A.shape
        #numer = [ np.abs(np.dot(self.c, point) - np.dot(self.dual, self.b)) / np.abs(np.dot(self.dual, self.b)) for point in points ]
        numer = [np.abs(np.dot(self.c, point) - 1) for point in points]
        numer = sum(numer)
        denom = 0
        for i in range(m):
            #denomTerm = [ np.abs(np.dot(self.A[i], point) - self.b[i]) / np.abs(self.b[i]) for point in points ]
            denomTerm = [
                np.abs(
                    np.dot(self.A[i] / np.linalg.norm(
                        self.A[i].T, self.normalize_c), point) - 1)
                for point in points
            ]
            denom += sum(denomTerm)
        rho = 1 - numer / denom
        return rho[0, 0]

    def _initialize_kwargs(self, kwargs):
        if 'verbose' in kwargs:
            assert isinstance(kwargs['verbose'],
                              bool), 'verbose needs to be True or False.'
            self._verbose = kwargs['verbose']

        if 'tol' in kwargs:
            assert isinstance(kwargs['tol'],
                              int), 'tolernace needs to be an integer.'
            self.tol = kwargs['tol']

        if 'ban_constraints' in kwargs:
            assert isinstance(kwargs['ban_constraints'],
                              list), 'ban constraints needs to be a list.'
            self.ban_constraints = kwargs['ban_constraints']

        if 'normalize_c' in kwargs:
            assert kwargs['normalize_c'] == 1 or kwargs['normalize_c'] == np.inf, 'normalize c with 1 or infinity norm.'
            self.normalize_c = kwargs['normalize_c']

        if 'solver' in kwargs:
            if kwargs['solver'] in cvx.installed_solvers():
                self.solver = getattr(cvx, kwargs['solver'])
            else:
                print('you do not have this solver.')

        return kwargs
