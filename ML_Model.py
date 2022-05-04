import numpy as np
import pandas as pd
from sklearn import linear_model
from preprocessing import prepare_Data

def Linear_Regression(Data):
    X = Data.iloc[: , 1:]
    Y = Data["Average User Rating"]
    reg = linear_model.LinearRegression()
    reg.fit(X , Y)  # i guess it do the trainig
    print("acuracy is : {}".format(reg.score(X , Y)))
    return reg


def Get_Means(Data):
    Means = dict()
    Means["Developer"]= Data["Developer"].values.mean()
    Means["Price"]= Data["Price"].values.mean()
    Means["Age Rating"]= Data["Age Rating"].values.mean()
    Means["Size"]= Data["Size"].values.mean()
    Means["Age Rating"]= Data["Age Rating"].values.mean()
    Means["Languages"]= Data["Languages"].mean()
    Means["Genres"]= Data["Genres"].mean()
    print(Means.keys(),Means.values())
    return Means


def Predict_Linear_Regression(input , reg):
    """
    :param input:  supposed to be pandas frame
    :param reg:
    :return:
    """
    input = prepare_Data(input)
    Avg_user_rate = reg.predict([input])
    print("average user rate is " , Avg_user_rate)
    return Avg_user_rate