import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split

def normalization(DataName , Data):
    Max=Data[DataName].max()
    Min=Data[DataName].min()
    Variance=Max-Min
    Data[DataName]=(Data[DataName]-Min)/Variance
    return  Data

"""
    we will bring price using regression 
"""
def drop_cols(Data):
    """
    been droped based on the correlation
    or to be replaced by infered value
    """
    droped_cols = ["ID" ,"URL" ,  "Icon URL"  , "Original Release Date"  , 'Description',
                   "Current Version Release Date"  , "Subtitle","Name" , 'In-app Purchases']
    Data = Data.drop(droped_cols , axis=1 , inplace=True)

def drop_rows(Data):

    # we need test it
    l = ["Languages"]
    for tmp in l:
        data = Data[ Data[tmp].notnull() ]
    return  data

def replace_NaNs(Data):
    Data["Price"].replace(np.NaN, Data["Price"].mean(), inplace=True)  # Replaced with avg
    Data["Average User Rating"].replace(np.NaN, Data["Average User Rating"].mean(),inplace=True)  # replace with average
    Data["User Rating Count"].replace(np.NaN, Data["User Rating Count"].mean(),inplace=True)  # Replaced with average


def convert_date(str):
    """
    :param str:  single data in string format
    :return:  list of 3 values (day , month , year)
    """
    l = str.split('/')
    l = np.array(l , dtype=np.int)
    return l

def subtract_dates(l1 , l2):
    # year , month , day
    org = datetime(l1[2] , l1[1] , l1[0])
    cur = datetime(l2[2] , l2[1] , l2[0])
    diff = cur-org
    return diff.days

def encoding():
    pass

def ordinal_encoding(Data):
    """
        we will convert the data into np array
        and we will insure that it's all is strings
    """
    Data.fillna(0, inplace=True)
    Data = Data.values  # return np array
    Data = Data.astype(str)
    oe = OrdinalEncoder()
    oe.fit(Data)
    Data = oe.transform(Data)
    return Data

def get_diff_days(d1 , d2):
    """
     more days mean more support , more popularity and successes
    :param d1:  orig data  list  as strings
    :param d2:  cur date list as strings
    :return: list of diff in days as ints
    """

    assert (d1.shape == d2.shape)

    orig = []
    cur = []
    diff_days = []
    # convert the dates
    for i in range(0 , d1.shape[0]):
        orig_date = convert_date(d1[i])
        cur_date  = convert_date(d2[i])
        orig.append(orig_date)
        cur.append(cur_date)
    assert (len(orig)==len(cur))
    # get the difference in days
    for i in range(0 , len(orig)):
        diff_days.append(subtract_dates(orig[i] , cur[i]))
    return diff_days  #actual days


def convert_age_rating(C):
    """
        - delete the + char from the string
        return : l is list of age limit as ints
    """
    G = []
    for i in C:
        i = i[:-1]
        i = (int)(i)
        G.append(i)

    return G

# this function should do get values for the generas and any one that look like it
"""
def get_average(Data , X):
    # encode the feature 
    Data.fillna(0, inplace=True)
    freq = dict()
    cnt = dict()
    # initialize the dicts
    for i in range(0 , X.shape[0]):
        freq[X[i]] = 0
        cnt[X[i]] = 0
    for i in range(0 , Data.shape[0]):
        val = float(Data["Average User Rating"][i])
        ty = Data["Genres"][i]
        freq[ty] +=val  # sum the avg user rates
        cnt[ty] +=1  # count how many belong to this ty
    # get average
    for i in range(0 , X.shape[0]):
        if freq[X[i]] ==0:
            print(X[i])
        freq[X[i]] = freq[X[i]] / cnt[X[i]]
    return freq
"""

def Encode_Multiple_Features(input,target):
    # read it again
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


def Encode_Single_Features(input , target):
    # test it
    my_map = dict()
    lenght_map = dict()
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


def features_encoding(Data):
    # test it
    ls = [('Developer' , "Average User Rating") , ("Primary Genre" , "Average User Rating")]
    # encode single variable
    for i in range(0 , len(ls)):
        dep  , indep= ls[i]
        Data[dep] = Encode_Single_Features(Data[dep].values , Data[indep].values)

    # encode multiple feature
    lm = [('Languages', "Average User Rating"), ("Genres", "Average User Rating")]
    for i  in range(0 , len(lm)):
        dep, indep = lm[i]
        print("what I use " , dep , indep)
        Data[dep] = Encode_Multiple_Features(Data[dep].values, Data[indep].values)
    return Data

def prepare_Data(Data , save=False , name="newfile"):
    """
    :param Data:  data frame of pandas.
    :return:  Data frame of pandas with all the of my changes applied on it.
    """
    replace_NaNs(Data)
    Data = drop_rows(Data)
    Data["Age Rating"] = convert_age_rating(Data["Age Rating"])
    Data["game_age"] = get_diff_days(Data['Original Release Date'].values,
                                     Data['Current Version Release Date'].values)
    drop_cols(Data)
    Data = features_encoding(Data)  # it have to come after the drop_cols why !???

    # save the file
    if save==True:
        Data.to_csv(name+".csv")

    return Data

def split_data(Data):
    """
    :param Data:  pandas data frame
    :return:  pandas data frame
    """
    X = Data.iloc[: , 1:]
    Y = Data.iloc[: , 0]

    X_train , X_test , Y_train , Y_test = train_test_split(X , Y , test_size=0.2)
    return X_train , X_test , Y_train , Y_test


# list version **********************
#R["Developer"]=NormalizeA2B(R["Developer"].values,R["Average User Rating"].values)#develoepr