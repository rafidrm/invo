import numpy as np
import pudb

from ..LinearModels.AbsoluteDualityGap import AbsoluteDualityGap




def simpleLp():
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

    interiorPoints = [
            np.array([0.1, 0.1, 0.1, 0.1]),
            np.array([0.2, 0.4, 0.2, 0.3]),
            np.array([0.1, 0.3, 0.3, 0.5]),
            ]

    adg = AbsoluteDualityGap()
    adg.ForwardModel(forward='hull', points=vertices)
    adg.solve(interiorPoints)
    results = {
            'solved': adg.solved,
            'cost': adg.c,
            'error': adg.error,
            'dual': adg.dual
            }
    return results 
    



if __name__ == "__main__":
    result = simpleLp()
    for key, value in result.items():
        print ('{} = {}'.format(key, value))
