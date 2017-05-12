""" Inverse optimization for Linear Models

This module contains a collection of linear inverse optimization models. It
includes the ones covered by generalized inverse optimization, including the
absolute and relative duality gap, and p-norm methods. As we add new models,
we should include them in the __main__ file.
"""



from .RelativeDualityGap import RelativeDualityGap
from .AbsoluteDualityGap import AbsoluteDualityGap
from .pNorm import pNorm
