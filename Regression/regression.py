from sklearn.metrics import r2_score
import numpy as np
import pandas as pd

def regression_r2(x,y):  # x is true labels and y is predicted
    return r2_score(x, y)
