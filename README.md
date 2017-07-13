# invo - An Inverse Optimization Library

## Setup

Invo is available on PyPi, so you can just pip install it.

```python
pip install invo
```

Invo uses numpy, scipy, and cvxpy as dependencies, so you may need to install them first.


## Usage

For more detailed documentation, refer to our [ReadTheDocs](http://pyinvo.readthedocs.io/en/latest/)


An invo problem has two stages. You first define a forward model, and then you solve the corresponding inverse optimization problem. Currently, we assume forward problems are given in inequality form as follows:
<p align="center"><img src="https://rawgit.com/rafidrm/invo/master/svgs/98a1e418f729823bb9c36e247ad53bcc.svg?invert_in_darkmode" align=middle width=96.70749pt height=16.376943pt/></p> 


```python
import numpy as np
from invo.LinearModels import AbsoluteDualityGap

# Construct a random forward problem.
vertices = [ np.random.rand(4) for i in range(8) ]
from invo.utils.fwdutils import fit_convex_hull
A, b = fit_convex_hull(vertices)

# Construct a set of optimal observed decisions.
optimalPoints = [ np.random.rand(4) for i in range(4) ]


# Add the forward problem, then run inverse optimization.
model = AbsoluteDualityGap()
model.FOP(A, b)
model.solve(optimalPoints)
print (model.c)
```


## Currently completed

* Absolute Duality Gap linear model
* Relative Duality Gap linear model
* pNorm linear model




