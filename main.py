import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from preprocessing import *
from build_data_set import *
from Visualization import *
from ML_Model import *


FILE_NAME = "final.csv"

def ReadData(file_name):
   data_set = pd.read_csv(file_name)
   data_set.drop(data_set.columns[data_set.columns.str.contains('unnamed', case=False)], axis=1, inplace=True)
   # drop Unnamed problem
   return data_set

def nan_statistics(Data):
    total_missing = Data.isnull().values.sum()
    print("total number of rows is : {}".format(Data.shape[0]))
    print("total number of missing entries is : {}".format(total_missing))

    print("\n statistics for each feature and number of examples that miss that feature \n")
    print(Data.isnull().sum())
    return


Data = ReadData(FILE_NAME)


print("\n\n")
X_train , X_valid , Y_train , Y_valid  = split_data(Data)

train_data = pd.concat([Y_train , X_train] , axis=1)
valid_data = pd.concat([Y_valid , X_valid] , axis=1)
print(train_data)

reg = Linear_Regression(train_data)
print(reg.score(X_valid , Y_valid))

Get_Means(Data)