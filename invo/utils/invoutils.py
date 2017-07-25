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


def validateFOP(A, b):
    """ Validate that A, b are numpy matrices with the appropriate structure.
    """
    aType = type(A)
    bType = type(b)
    if aType is np.matrixlib.defmatrix.matrix:
        m, n = A.shape
        Aout = A

    elif aType is np.ndarray:
        m, n = A.shape
        Aout = np.matrix(A)
    
    elif aType is list:
        m = len(A)
        n = len(A[0])
        for ai in A:
            if len(ai) != n:
                raise ValueError('Rows of A are incomplete')
        aRowType = type(A[0])
        
        if aRowType is np.matrixlib.defmatrix.matrix:
            Aout = []
            for ai in A:
                Aout.append(ai.tolist())[0]
        elif aRowType is np.ndarray:
            Aout = []
            for ai in A:
                Aout.append(ai.tolist())[0]
        elif aRowType is list:
            Aout = A
        
        else:
            raise TypeError('Type of A is incorrect.')
        Aout = np.matrix(Aout)

    else:
        raise TypeError('Type of A is incorrect.')

    if bType is np.matrixlib.defmatrix.matrix:
        m2, n2 = b.shape
        if m2 != m:
            bout = b.T
        else:
            bout = b

    elif bType is np.ndarray:
        if len(b.shape) == 1:
            bout = np.mat(b).T
        else:
            raise TypeError('Type of b is incorrect.')

    elif bType is list:
            bout = np.mat(b).T

    else:
        raise TypeError('Type of b is incorrect.')


    # do a final validation here.
    m, n = Aout.shape
    m2, n2 = bout.shape
    if m != m2:
        raise TypeError('Type of A or b is incorrect.')
    if n2 != 1:
        raise TypeError('Type of b is incorrect.')
    return Aout, bout



def validatePoints(points):
    pType = type(points)

    if pType is list:
        p2Type = type(points[0])
        pOut = []
        for point in points:
            if type(point) != p2Type:
                raise TypeError('Points are not all the same type.')
            if p2Type is list:
                pOut.append(np.mat(point).T)
            elif p2Type is np.matrixlib.defmatrix.matrix:
                if point.shape[0] == 1: 
                    pOut.append(point.T)
                else:
                    pOut.append(point)
            elif p2Type is np.ndarray:
                if len(p3.shape) == 1:
                    pOut.append(np.mat(point).T)
                else:
                    raise TypeError('Type of points is incorrect.')
            else:
                raise TypeError('Type of points is incorrect.')

    elif pType is np.matrixlib.defmatrix.matrix:
        pass

    elif pType is np.ndarray:
        pass

    else:
        raise TypeError('Type of points is incorrect.')
