import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from docutils.nodes import label
from sklearn import linear_model
import seaborn as sns
from sklearn import metrics
import statsmodels.api as sm
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import seaborn as sb
from sklearn.preprocessing import Imputer
def NormalizeA2BMulti(input , target):
    my_map = {}
    count_map = {}
    n = len(input)
    for j in range(n):
        row = input[j]
        for i in range(len(row)):
            x=row[i]
            if my_map.__contains__(x) == False:
                my_map.setdefault(x, 0)
                count_map.setdefault(x, 0)
            my_map[x]+= target[i]
            count_map[x]+=1

    for i in my_map.keys():
        my_map[i]/=count_map[i]

    Result_list = []
    for i in range(n):
        row = input[i]
        Total =0
        for j in range(len(row)):
            Total += my_map[row[j]]
        Result_list.append(Total)

    return Result_list
#################################################
def NormalizeA2B(input , target):
    my_map = {}
    lenght_map = {}
    n = len(input)
    for i in range(n):
        x = input[i]
        if my_map.__contains__(x) == False:
            my_map.setdefault(x, 0)
            lenght_map.setdefault(x, 0)
        my_map[x]+= target[i]
        lenght_map[x]+=1

    for i in my_map.keys():
        my_map[i]/=lenght_map[i]

    Result_list = []
    for i in range(n):
        Result_list.append(my_map[input[i]])

    return Result_list
#############################################

def split(N):
   res = []
   values_2Darr = []
   label_encoder= LabelEncoder()
   x= R[N].values

   for i in range(len(x)):
       values_2Darr.append(x[i].split(','))
   for i in values_2Darr:
       res.append(label_encoder.fit_transform(i))

   for i in range(len(res)):
       res[i] =sum(res[i])/len(res[i])

   return res

def Correlation():
    corr = R.corr()
    #top_feature = corr.index[abs(corr['Value'] > 0.5)]
    # Correlation plot
    plt.subplots(figsize=(12, 8))
    #top_corr = R[top_feature].corr()
    sns.heatmap(corr, annot=True)
    plt.show()

def Get_NO(C):
    G = []
    for i in C:
        i=i[:-1]
        i=(int)(i)
        G.append(i)

    return G



def normalization(DataName):
    Max=R[DataName].max()
    Min=R[DataName].min()
    Variance=Max-Min
    R[DataName]=(R[DataName]-Min)/Variance


def Feature_Encoder(X,cols):
    for c in cols:
        lbl = LabelEncoder()
        lbl.fit(list(X[c].values))
        X[c] = lbl.transform(list(X[c].values))
    return X


def ReadData():
   data_set = pd.read_csv("./predicting_mobile_game_success_train_set.csv")
   data_set.drop(data_set.columns[data_set.columns.str.contains('unnamed', case=False)], axis=1, inplace=True)
   # drop Unnamed18 problem
   return data_set


def Model(X,Y,LabelX,LabelY):
    cls = linear_model.LinearRegression()
    X = np.expand_dims(X, axis=1)
    Y = np.expand_dims(Y, axis=1)
    cls.fit(X, Y)  # Fit method is used for fitting your training data into the model
    prediction = cls.predict(X)
    plt.scatter(X, Y)
    plt.xlabel(LabelX, fontsize=20)
    plt.ylabel(LabelY, fontsize=20)
    plt.plot(X, prediction, color='red', linewidth=3)
    print('Mean Square Error', metrics.mean_squared_error(Y, prediction))
    plt.show()




R=ReadData()

'''''
label_encoder = LabelEncoder()

R.iloc[:,0] = label_encoder.fit_transform(R.iloc[:,0]).astype('float64')*R.iloc[:,1]*0.1 #name
R.iloc[:,5] = label_encoder.fit_transform(R.iloc[:,5]).astype('float64')*R.iloc[:,1]*0.1#Description
R.iloc[:,6] = label_encoder.fit_transform(R.iloc[:,6]).astype('float64')*R.iloc[:,1]*0.1 #Developer
R['Languages'] = split("Languages")#language
R['Genres'] = split("Genres")#Genres
R.iloc[:,10] = label_encoder.fit_transform(R.iloc[:,10]).astype('float64')*R.iloc[:,1]*0.1#Primary Genre
R.iloc[:,12] = label_encoder.fit_transform(R.iloc[:,12]).astype('float64')*R.iloc[:,1]*0.1#Original Release Date
R.iloc[:,13] = label_encoder.fit_transform(R.iloc[:,13]).astype('float64')*R.iloc[:,1]*0.1#Current Version Release Date

'''
''''
normalization("Average User Rating")
normalization("User Rating Count")
normalization("Name")
normalization("Description")
normalization("Developer")
normalization("Age Rating")

'''''

R.iloc[:,9]=Get_NO(R.iloc[:,9]) #Age Rating
#print(R["Age Rating"])

#########################################

R["Developer"]=NormalizeA2B(R["Developer"].values,R["Average User Rating"].values)#develoepr
R["Description"]=NormalizeA2B(R["Description"].values,R["Average User Rating"].values)#Description
R["Primary Genre"]=NormalizeA2B(R["Primary Genre"].values,R["Average User Rating"].values)#Description
R["Name"]=NormalizeA2B(R["Name"].values,R["Average User Rating"].values)#Name
R["URL"]=NormalizeA2B(R["URL"].values,R["Average User Rating"].values)#URL
R["Icon URL"]=NormalizeA2B(R["Icon URL"].values,R["Average User Rating"].values)#Icon URL

R["Original Release Date"]=NormalizeA2B(R["Original Release Date"].values,R["Average User Rating"].values)#Original Release Date
R["Current Version Release Date"]=NormalizeA2B(R["Current Version Release Date"].values,R["Average User Rating"].values)#Current Version Release Date

R["Languages"]=NormalizeA2BMulti(R["Languages"].values,R["Average User Rating"].values)#Languages
R["Genres"]=NormalizeA2BMulti(R["Genres"].values,R["Average User Rating"].values)#Genres

Correlation()

