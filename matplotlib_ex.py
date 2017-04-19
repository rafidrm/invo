import numpy as np
import pandas as pd
from scipy import stats, integrate
import matplotlib.pyplot as plt
from sys import platform as sys_pf
if sys_pf == 'darwin':
    import matplotlib
    matplotlib.use("TkAgg")

import seaborn as sns
sns.set(color_codes=True)


np.random.seed(sum(map(ord, "distributions")))



x = np.random.normal(size=100)
sns.distplot(x);


