import numpy as np
from collections import Counter

#NORMALISE COLUMN (value to range [0,1])
def normalise_column(_col):
    max_val = np.amax(_col)
    min_val = np.amin(_col)
    new_col = []
    for _val in _col:
        _newVal = (_val - min_val) / (max_val - min_val)
        new_col.append(_newVal)
    return np.array(new_col)