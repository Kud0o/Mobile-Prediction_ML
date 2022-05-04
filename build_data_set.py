import numpy as np

def save_to_csv(X):
    """
    - we assume that the in comming data is numpy array
    """
    print(X.shape)
    np.savetxt("age_rate.csv" , X , header="age_rate")
    return