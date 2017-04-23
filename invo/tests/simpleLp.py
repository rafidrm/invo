import numpy as np
import pudb

from ..LinearModels.AbsoluteDualityGap import AbsoluteDualityGap
from ..LinearModels.RelativeDualityGap import RelativeDualityGap




def simpleLp(vertices, testPoints, model):
    adg = model 
    adg.ForwardModel(forward='hull', points=vertices)
    adg.solve(testPoints)
    results = {
            'solved': adg.solved,
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
    
    
    
    print('\n *** Testing ADG with feasible points, hull method *** \n')
    interiorPoints = [
            np.array([0.1, 0.1, 0.1, 0.1]),
            np.array([0.2, 0.4, 0.2, 0.3]),
            np.array([0.1, 0.3, 0.3, 0.5]),
            ]
    model = AbsoluteDualityGap()
    result = simpleLp(vertices, interiorPoints, model)
    for key, value in result.items():
        print ('{} = {}'.format(key, value))
    
    print('\n *** Testing ADG with feasible and infeasible points, hull method *** \n')
    mixedPoints = [
            np.array([1.1, 0.1, 0.1, 0.1]),
            np.array([0.2, 0.4, 0.2, 0.3]),
            np.array([0.1, 4.2, 1.4, 0.5]),
            ]
    model = AbsoluteDualityGap()
    result = simpleLp(vertices, mixedPoints, model)
    for key, value in result.items():
        print ('{} = {}'.format(key, value))
    
    print('\n *** Testing RDG with feasible points, hull method *** \n')
    interiorPoints = [
            np.array([0.1, 0.1, 0.1, 0.1]),
            np.array([0.2, 0.4, 0.2, 0.3]),
            np.array([0.1, 0.3, 0.3, 0.5]),
            ]
    model = RelativeDualityGap()
    result = simpleLp(vertices, interiorPoints, model)
    for key, value in result.items():
        print ('{} = {}'.format(key, value))
    
    print('\n *** Testing RDG with feasible and infeasible points, hull method *** \n')
    mixedPoints = [
            np.array([1.1, 0.1, 0.1, 0.1]),
            np.array([0.2, 0.4, 0.2, 0.3]),
            np.array([0.1, 4.2, 1.4, 0.5]),
            ]
    model = RelativeDualityGap()
    result = simpleLp(vertices, mixedPoints, model)
    for key, value in result.items():
        print ('{} = {}'.format(key, value))
