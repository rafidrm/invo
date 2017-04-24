import numpy as np




def checkFeasibility(points, A, b, tol=8):
    """ Check if the set of points satisfy A x >= b.
    
    Args:
        points (list): set of numpy points.
        A (list): numpy matrix.
        b (list): numpy vector.
    """
    def _checkFeasibilityPoint(point, A, b, tol):
        check = np.round(A * point - b, tol)
        return (check >= 0).all()
    
    allPointsAreFeasible = np.array([ _checkFeasibilityPoint(point, A, b, tol) for point in points]).all()
    return allPointsAreFeasible
