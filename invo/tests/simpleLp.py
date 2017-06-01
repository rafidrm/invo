import numpy as np
import pudb

from ..LinearModels import AbsoluteDualityGap
from ..LinearModels import RelativeDualityGap
from ..LinearModels import pNorm
from ..utils.fwdutils import fit_convex_hull


def simpleLp(vertices, testPoints, model):
    adg = model 
    #adg.ForwardModel(forward='hull', points=vertices)
    adg.solve(testPoints)
    results = {
            'solved': adg._solved,
            'cost': adg.c,
            'error': adg.error,
            'dual': adg.dual
            }
    return results 
    


if __name__ == "__main__":
    
    vertices = [
            np.array([0.0, 0.0, 0.0, 0.0]),
            np.array([1.0, 0.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0, 0.0]),
            np.array([0.0, 0.0, 1.0, 0.0]),
            np.array([0.0, 0.0, 0.0, 1.0]),
            np.array([0.3, 0.6, 0.2, 0.4]),
            np.array([0.5, 0.2, 0.3, 0.5]),
            np.array([0.1, 0.5, 0.3, 0.7])
            ]
    A, b = fit_convex_hull(vertices)
    
    
    
    print('\n *** Testing ADG with feasible points, hull method *** \n')
    interiorPoints = [
            np.array([0.1, 0.1, 0.1, 0.1]),
            np.array([0.2, 0.4, 0.2, 0.3]),
            np.array([0.1, 0.3, 0.3, 0.5]),
            ]
    model = AbsoluteDualityGap()
    model.FOP(A, b)
    result = simpleLp(vertices, interiorPoints, model)
    for key, value in result.items():
        print ('{} = {}'.format(key, value))
    rho = model.rho(interiorPoints)
    print('rho = {}'.format(rho))
    
    print('\n *** Testing ADG with feasible and infeasible points, hull method *** \n')
    mixedPoints = [
            np.array([1.1, 0.1, 0.1, 0.1]),
            np.array([0.2, 0.4, 0.2, 0.3]),
            np.array([0.1, 4.2, 1.4, 0.5]),
            ]
    model = AbsoluteDualityGap()
    model.FOP(A, b)
    result = simpleLp(vertices, mixedPoints, model)
    for key, value in result.items():
        print ('{} = {}'.format(key, value))
    rho = model.rho(mixedPoints)
    print('rho = {}'.format(rho))
    
    print('\n *** Testing RDG with feasible points, hull method *** \n')
    interiorPoints = [
            np.array([0.1, 0.1, 0.1, 0.1]),
            np.array([0.2, 0.4, 0.2, 0.3]),
            np.array([0.1, 0.3, 0.3, 0.5]),
            ]
    model = RelativeDualityGap()
    model.FOP(A, b)
    result = simpleLp(vertices, interiorPoints, model)
    for key, value in result.items():
        print ('{} = {}'.format(key, value))
    rho = model.rho(interiorPoints)
    print('rho = {}'.format(rho))
    
    print('\n *** Testing RDG with feasible and infeasible points, hull method *** \n')
    mixedPoints = [
            np.array([1.1, 0.1, 0.1, 0.1]),
            np.array([0.2, 0.4, 0.2, 0.3]),
            np.array([0.1, 4.2, 1.4, 0.5]),
            ]
    model = RelativeDualityGap()
    model.FOP(A, b)
    result = simpleLp(vertices, mixedPoints, model)
    for key, value in result.items():
        print ('{} = {}'.format(key, value))
    rho = model.rho(mixedPoints)
    print('rho = {}'.format(rho))

    print('\n *** Testing pNorm with feasible points, hull method *** \n')
    interiorPoints = [
            np.array([0.1, 0.1, 0.1, 0.1]),
            np.array([0.2, 0.4, 0.2, 0.3]),
            np.array([0.1, 0.3, 0.3, 0.5]),
            ]
    model = pNorm(p=2)
    model.FOP(A, b)
    result = simpleLp(vertices, interiorPoints, model)
    for key, value in result.items():
        print ('{} = {}'.format(key, value))
    projections = model.optimal_points(interiorPoints)
    print(projections)
    rho = model.rho(interiorPoints)
    print('rho = {}'.format(rho))

    print('\n *** Testing pNorm with feasible and infeasible points, hull method *** \n')
    mixedPoints = [
            np.array([1.1, 0.1, 0.1, 0.1]),
            np.array([0.2, 0.4, 0.2, 0.3]),
            np.array([0.1, 4.2, 1.4, 0.5]),
            ]
    model = pNorm(p=2)
    model.FOP(A, b)
    result = simpleLp(vertices, mixedPoints, model)
    for key, value in result.items():
        print ('{} = {}'.format(key, value))
    projections = model.optimal_points(interiorPoints)
    print(projections)
    rho = model.rho(mixedPoints)
    print('rho = {}'.format(rho))

