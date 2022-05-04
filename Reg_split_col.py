import pandas as pd

SetGenres = [] # Hold the Names of Genres
HoldValues=[]
newHoldeValues=[]

Data = pd.read_csv("./predicting_mobile_game_success_train_set.csv")
Data.drop(Data.columns[Data.columns.str.contains('unnamed', case=False)], axis=1, inplace=True)


for i in Data['Genres']:
    values=i.split(',')
    SetGenres.append(values)


my_map = {}
n = len(SetGenres)
for j in range(n):
    row = SetGenres[j]
    for i in range(len(row)):
        x = row[i]
        if my_map.__contains__(x) == False:
            my_map.setdefault(x, 0)
names=[]
for Name in my_map.keys():
    my_map[Name]=[]
    #names.append(Name)

for i in Data['Genres']:
    values=i.split(',')
    for j in my_map.keys():
        if j in values:
            my_map[j].append(1)
        else:
            my_map[j].append(0)

v=pd.DataFrame()

for i in my_map.keys():
    v[i]= my_map[i]
##############################################

#cc=pd.DataFrame(newHoldeValues)
v.to_csv("FinalGenres.csv")

