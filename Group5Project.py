#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 14:35:03 2019

@author: ximenamartinez
"""


import pandas as pd
import matplot.lib.pyplot as plt
from sklearn.model_selection import train_test_split
from scipy import stats
import numpy as np

#imported data set
sf = pd.read_csv("/Users/ximenamartinez/Desktop/San_Francisco_Communitywide_Greenhouse_Gas_Inventory.csv")



############################################### Exploritory Data Phase :)###############################################################

#Columns 
list(sf)

#Array for the years
years = [1990,2000,2005,2010,2012,2015,2016,2017]


#Eliminates rows with zeros
sf = sf[sf.Quantity != 0]
sf = sf[sf.Emissions_mtCO2e != 0]

#Quick grouping to visualize
sf.groupby('Commodity_Type').size()
sf.groupby('Quantity_Units').size()

#eliminated 0 emission records
sf = sf[sf.Emissions_mtCO2e != 0]
sf

#Division of emission by quantitiy and add as new column 
d = sf['Emissions_mtCO2e'].div(sf['Quantity'])
d
sf['Emission per Unit'] = (d).map(str) + " Emissions per "+ sf['Quantity_Units']
sf['Emission per Unit']

sort = sf # for future sorting

#Quick grouping to visualize
list(sf)
sf.groupby(['Quantity_Units']).size()
sf.groupby(['Commodity_Type']).size()
sf.groupby(['Calendar_Year']).size()
sf

#use for decision tree (Shows range of Emissions per Unit for each type)
#Grouping gets a little odd for scientific notation, make note
grouping = sf.groupby(['Quantity_Units','Emission per Unit']).size()
grouping #all
grouping[0:8] #Avg kgN/day
grouping[8:16] #MMBTU
grouping[16:100] #Therm
grouping[100:199] #gallons
grouping[199:289] #kWh
grouping[289:297] #miles
grouping[297:298] #This shows infinity as a ratio b/c quantity is zero, we may have to disregard this data ??
grouping[298:306] #people
grouping[306:315] #tons


# seperating to test and trianing
sf_train, sf_test = train_test_split(sf, test_size = 0.5, random_state = 8)

#check size
sf.shape
sf_train.shape
sf_test.shape



#indexing
index1 = list(range(197))
index2 = list(range(198))

sf_train['index'] = index1
sf_test['index'] = index2

sf_train
sf_test

#standardizing
list(sf)
sf_train['Emissions_mtCO2e_z'] = stats.zscore(sf_train['Emissions_mtCO2e'])
sf_test['Emissions_mtCO2e_z'] = stats.zscore(sf_test['Emissions_mtCO2e'])

sf_train['Quantity_z'] = stats.zscore(sf_train['Quantity'])
sf_test['Quantity_z'] = stats.zscore(sf_test['Quantity'])

#outliers
sf_train.query('Emissions_mtCO2e_z > 3 | Emissions_mtCO2e_z < -3')
sf_train.query('Quantity_z > 3 | Quantity_z < -3')

sf_test.query('Emissions_mtCO2e_z > 3 | Emissions_mtCO2e_z < -3')
sf_test.query('Quantity_z > 3 | Quantity_z < -3')


sort.sort_values('Emissions_mtCO2e', inplace = True, ascending = False)
a = sort['Emissions_mtCO2e']

#bins
sf_train['em_binned'] = pd.cut(x = sf_train['Emissions_mtCO2e'], bins = [0, 5000, 50000, 500000, 1000000, 2500000], labels=["Under 5000", "5000 to 50000", "50000 to 500000",  "500000 to 1000000", "Over 1000000"], right = False)
list(sf)
crosstab_01 = pd.crosstab(sf_train['em_binned'], sf_train['Quantity'])
crosstab_01.plot(kind='bar', stacked = True, title = 'Bar Graph of Emissions (Binned) with Quantity Overlay', legend = None)



###############################################End Exploritory Data Phase :D###############################################################


############################################## Regression Model#############################################################################

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import random
import statsmodels.tools.tools as stattools
import statsmodels.api as sm
import sklearn.metrics as met
from scipy import stats
import matplotlib.pyplot as plt
from sklearn import linear_model
import sklearn.linear_model as sk
from sklearn.linear_model import LinearRegression

sf = pd.read_csv("/users/sapnaraom/Downloads/San_Francisco_Communitywide_Greenhouse_Gas_Inventory.csv")
sf_train, sf_test = train_test_split(sf, test_size = 0.5, random_state = 7)

Sector_General_np = np.array(sf_train['Sector_General'])
(Sector_General_cat, Sector_General_cat_dict) = stattools.categorical(Sector_General_np, drop = True, dictnames = True)
inv_map = {v: k for k, v in Sector_General_cat_dict.items()}
Sector_General_cat_pd = sf_train['Sector_General'].apply(lambda r: inv_map[r])

Sector_Detail1_np = np.array(sf_train['Sector_Detail1'])
(Sector_Detail1_cat, Sector_Detail1_cat_dict) = stattools.categorical(Sector_Detail1_np, drop = True, dictnames = True)
inv_map = {v: k for k, v in Sector_Detail1_cat_dict.items()}
Sector_Detail1_cat_pd = sf_train['Sector_Detail1'].apply(lambda r: inv_map[r])

Sector_Detail2_np = np.array(sf_train['Sector_Detail2'])
(Sector_Detail2_cat, Sector_Detail2_cat_dict) = stattools.categorical(Sector_Detail2_np, drop = True, dictnames = True)
inv_map = {v: k for k, v in Sector_Detail2_cat_dict.items()}
Sector_Detail2_cat_pd = sf_train['Sector_Detail2'].apply(lambda r: inv_map[r])

Sector_GPC_np = np.array(sf_train['Sector_GPC'])
(Sector_GPC_cat, Sector_GPC_cat_dict) = stattools.categorical(Sector_GPC_np, drop = True, dictnames = True)
inv_map = {v: k for k, v in Sector_GPC_cat_dict.items()}
Sector_GPC_cat_pd = sf_train['Sector_GPC'].apply(lambda r: inv_map[r])

Sector_GPC_Detail_np = np.array(sf_train['Sector_GPC_Detail'])
(Sector_GPC_Detail_cat, Sector_GPC_Detail_cat_dict) = stattools.categorical(Sector_GPC_Detail_np, drop = True, dictnames = True)
inv_map = {v: k for k, v in Sector_GPC_Detail_cat_dict.items()}
Sector_GPC_Detail_cat_pd = sf_train['Sector_GPC_Detail'].apply(lambda r: inv_map[r])

Commodity_Type_np = np.array(sf_train['Commodity_Type'])
(Commodity_Type_cat, Commodity_Type_cat_dict) = stattools.categorical(Commodity_Type_np, drop = True, dictnames = True)
inv_map = {v: k for k, v in Commodity_Type_cat_dict.items()}
Commodity_Type_cat_pd = sf_train['Commodity_Type'].apply(lambda r: inv_map[r])

Quantity_Units_np = np.array(sf_train['Quantity_Units'])
(Quantity_Units_cat, Quantity_Units_cat_dict) = stattools.categorical(Quantity_Units_np, drop = True, dictnames = True)
inv_map = {v: k for k, v in Quantity_Units_cat_dict.items()}
Quantity_Units_cat_pd = sf_train['Quantity_Units'].apply(lambda r: inv_map[r])

X = pd.concat((sf_train[['Calendar_Year']], Sector_General_cat_pd, Sector_Detail1_cat_pd, Sector_Detail2_cat_pd, Sector_GPC_cat_pd, Sector_GPC_Detail_cat_pd, Commodity_Type_cat_pd,sf_train[['Quantity']], Quantity_Units_cat_pd), axis = 1)
y = pd.DataFrame(sf_train[['Emissions_mtCO2e']])

X = sm.add_constant(X)
model01 = sm.OLS(y, X).fit()
model01.summary() # Prints summary of regression results of training data
reg = LinearRegression().fit(X, y)
print(reg.score(X, y))


Sector_General_np = np.array(sf_test['Sector_General'])
(Sector_General_cat, Sector_General_cat_dict) = stattools.categorical(Sector_General_np, drop = True, dictnames = True)
inv_map = {v: k for k, v in Sector_General_cat_dict.items()}
Sector_General_cat_pd1 = sf_test['Sector_General'].apply(lambda r: inv_map[r])

Sector_Detail1_np = np.array(sf_test['Sector_Detail1'])
(Sector_Detail1_cat, Sector_Detail1_cat_dict) = stattools.categorical(Sector_Detail1_np, drop = True, dictnames = True)
inv_map = {v: k for k, v in Sector_Detail1_cat_dict.items()}
Sector_Detail1_cat_pd1 = sf_test['Sector_Detail1'].apply(lambda r: inv_map[r])

Sector_Detail2_np = np.array(sf_test['Sector_Detail2'])
(Sector_Detail2_cat, Sector_Detail2_cat_dict) = stattools.categorical(Sector_Detail2_np, drop = True, dictnames = True)
inv_map = {v: k for k, v in Sector_Detail2_cat_dict.items()}
Sector_Detail2_cat_pd1 = sf_test['Sector_Detail2'].apply(lambda r: inv_map[r])

Sector_GPC_np = np.array(sf_test['Sector_GPC'])
(Sector_GPC_cat, Sector_GPC_cat_dict) = stattools.categorical(Sector_GPC_np, drop = True, dictnames = True)
inv_map = {v: k for k, v in Sector_GPC_cat_dict.items()}
Sector_GPC_cat_pd1 = sf_test['Sector_GPC'].apply(lambda r: inv_map[r])

Sector_GPC_Detail_np = np.array(sf_test['Sector_GPC_Detail'])
(Sector_GPC_Detail_cat, Sector_GPC_Detail_cat_dict) = stattools.categorical(Sector_GPC_Detail_np, drop = True, dictnames = True)
inv_map = {v: k for k, v in Sector_GPC_Detail_cat_dict.items()}
Sector_GPC_Detail_cat_pd1 = sf_test['Sector_GPC_Detail'].apply(lambda r: inv_map[r])

Commodity_Type_np = np.array(sf_test['Commodity_Type'])
(Commodity_Type_cat, Commodity_Type_cat_dict) = stattools.categorical(Commodity_Type_np, drop = True, dictnames = True)
inv_map = {v: k for k, v in Commodity_Type_cat_dict.items()}
Commodity_Type_cat_pd1 = sf_test['Commodity_Type'].apply(lambda r: inv_map[r])

Quantity_Units_np = np.array(sf_test['Quantity_Units'])
(Quantity_Units_cat, Quantity_Units_cat_dict) = stattools.categorical(Quantity_Units_np, drop = True, dictnames = True)
inv_map = {v: k for k, v in Quantity_Units_cat_dict.items()}
Quantity_Units_cat_pd1 = sf_test['Quantity_Units'].apply(lambda r: inv_map[r])

X_test = pd.concat((sf_test[['Calendar_Year']], Sector_General_cat_pd1, Sector_Detail1_cat_pd1, Sector_Detail2_cat_pd1, Sector_GPC_cat_pd1, Sector_GPC_Detail_cat_pd1, Commodity_Type_cat_pd1,sf_test[['Quantity']], Quantity_Units_cat_pd1), axis = 1)
np.isnan(X_test)
np.where(np.isnan(X_test))
np.nan_to_num(X_test)

y_test = pd.DataFrame(sf_test[['Emissions_mtCO2e']])
np.isnan(y_test)
np.where(np.isnan(y_test))
np.nan_to_num(y_test)

sf_train.to_csv("sf_train.csv")
X_test = sm.add_constant(X_test)
model01_test = sm.OLS(y_test, X_test).fit()
model01_test.summary()# Prints summary of regression results of test data

#MAEBaseline and MAERegression, to determine whether the regression model
#outperformed its baseline model.
ypred = model01.predict(X_test)
ytrue = sf_test[['Emissions_mtCO2e']]
MAE_Regression = met.mean_absolute_error(y_true = ytrue, y_pred = ypred)
print (MAE_Regression) 

y_values = sf_test[['Emissions_mtCO2e']]
y_values.mean()
y_values_abs = np.abs(y_values - y_values.mean())
count = sf_test[['Emissions_mtCO2e']].count()
MAE_baseline = np.sum(y_values_abs)/count
print (MAE_baseline)

"""
 (MAE_Regression) 
Emissions_mtCO2e  94238
 (MAE_baseline)
Emissions_mtCO2e    124753.2

MAE Regression < MAE baseline, therefore our model is beating the baseline.

"""

# The print function would print the predictions for y based on the linear model
lm = linear_model.LinearRegression()
model = lm.fit(X,y)
predictions = lm.predict(X)
print(predictions)

# This is the R² score of our model. As you probably remember, this the percentage of explained 
# variance of the predictions.
lm.score(X,y) # Around 0.7251451201206117

# The coefficients for the predictors
lm.coef_

"""
array([[ 0.00000000e+00, -1.98789122e+03,  8.37131826e+03,
        -1.78425274e+04, -3.68929734e+02, -4.59007477e+04,
         1.65511154e+04,  1.09844443e+04,  3.60342652e-04,
        -1.58758253e+04]])

"""
# the intercept:
lm.intercept_  

"""array([4051764.003046])"""

############################################### End Regression Model###############################################################

############################################### Decision Tree ####################################################################


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from scipy import stats
import numpy as np
import statsmodels.tools.tools as stattools


#Import for model: decision tree
from sklearn.tree import DecisionTreeClassifier, export_graphviz

#imported data set
sf = pd.read_csv("/Users/bonbon/downloads/USF/2019 Fall/MSIS678_DataWarehousing/Project/San_Francisco_Communitywide_Greenhouse_Gas_Inventory.csv")
#print(sf.to_string())


#print(sf_train.to_string())


# X = sf.drop('Emissions_mtCO2e', 1)
#transfer the continous varibles into categorical
Sector_General_np = np.array(sf['Sector_General'])
(Sector_General_cat, Sector_General_cat_dict) = stattools.categorical(Sector_General_np, drop = True, dictnames = True)
inv_map = {v: k for k, v in Sector_General_cat_dict.items()}
Sector_General_cat_pd = sf['Sector_General'].apply(lambda r: inv_map[r])
#print(Sector_General_cat_pd.to_string())

Sector_Detail1_np = np.array(sf['Sector_Detail1'])
(Sector_Detail1_cat, Sector_Detail1_cat_dict) = stattools.categorical(Sector_Detail1_np, drop = True, dictnames = True)
inv_map = {v: k for k, v in Sector_Detail1_cat_dict.items()}
Sector_Detail1_cat_pd = sf['Sector_Detail1'].apply(lambda r: inv_map[r])

Sector_Detail2_np = np.array(sf['Sector_Detail2'])
(Sector_Detail2_cat, Sector_Detail2_cat_dict) = stattools.categorical(Sector_Detail2_np, drop = True, dictnames = True)
inv_map = {v: k for k, v in Sector_Detail2_cat_dict.items()}
Sector_Detail2_cat_pd = sf['Sector_Detail2'].apply(lambda r: inv_map[r])

Sector_GPC_np = np.array(sf['Sector_GPC'])
(Sector_GPC_cat, Sector_GPC_cat_dict) = stattools.categorical(Sector_GPC_np, drop = True, dictnames = True)
inv_map = {v: k for k, v in Sector_GPC_cat_dict.items()}
Sector_GPC_cat_pd = sf['Sector_GPC'].apply(lambda r: inv_map[r])

Sector_GPC_Detail_np = np.array(sf['Sector_GPC_Detail'])
(Sector_GPC_Detail_cat, Sector_GPC_Detail_cat_dict) = stattools.categorical(Sector_GPC_Detail_np, drop = True, dictnames = True)
inv_map = {v: k for k, v in Sector_GPC_Detail_cat_dict.items()}
Sector_GPC_Detail_cat_pd = sf['Sector_GPC_Detail'].apply(lambda r: inv_map[r])

Commodity_Type_np = np.array(sf['Commodity_Type'])
(Commodity_Type_cat, Commodity_Type_cat_dict) = stattools.categorical(Commodity_Type_np, drop = True, dictnames = True)
inv_map = {v: k for k, v in Commodity_Type_cat_dict.items()}
Commodity_Type_cat_pd = sf['Commodity_Type'].apply(lambda r: inv_map[r])

Quantity_Units_np = np.array(sf['Quantity_Units'])
(Quantity_Units_cat, Quantity_Units_cat_dict) = stattools.categorical(Quantity_Units_np, drop = True, dictnames = True)
inv_map = {v: k for k, v in Quantity_Units_cat_dict.items()}
Quantity_Units_cat_pd = sf['Quantity_Units'].apply(lambda r: inv_map[r])

#concatenate categorical X variables
sf = pd.concat((sf[['Calendar_Year']], Sector_General_cat_pd, Sector_Detail1_cat_pd, Sector_Detail2_cat_pd, Sector_GPC_cat_pd, Sector_GPC_Detail_cat_pd, Commodity_Type_cat_pd,sf[['Quantity']], Quantity_Units_cat_pd, sf[['Emissions_mtCO2e']]), axis = 1)
sf_train, sf_test = train_test_split(sf, test_size = 0.5, random_state = 8)

#define axises
X = sf_train.drop('Emissions_mtCO2e', 1)
Xvl = sf_test.drop('Emissions_mtCO2e', 1)
y = sf_train.Emissions_mtCO2e
yvl = sf_test.Emissions_mtCO2e

X_names = ["Calendar_Year", " Sector_General", "Sector_Detail1", "Sector_Detail2", "Sector_GPC", "Sector_GPC_Detail", "Commodity_Type", "Quantity", "Quantity_Units"]

# y = sf.Emissions_mtCO2e
# define the interval and range of y
numOfCat = 3
y_names = [str(i) for i in range(numOfCat)]
print(y.to_string())
y = pd.qcut(y, numOfCat, labels=y_names)
yvl = pd.qcut(yvl, numOfCat, labels=y_names)

#set up the decission tree model with CART
cart_sf_train = DecisionTreeClassifier(criterion = "gini", min_samples_leaf = 5).fit(X,y)
export_graphviz(cart_sf_train, out_file = "/Users/bonbon/downloads/USF/2019 Fall/MSIS678_DataWarehousing/Project/cart_sf_train.dot", feature_names=X_names, class_names = y_names)

#compute the accuracy score of the decision tree
from sklearn.metrics import accuracy_score
model = DecisionTreeClassifier(random_state = 1)
model.fit(X,y)
pred_test = model.predict(Xvl)
score = accuracy_score(yvl, pred_test)
print('accuracy_score', score)


#compute the feature importances ranking by decision tree model
importances = pd.Series(cart_sf_train.feature_importances_, index=X.columns)
importances.plot(kind='barh', figsize=(12,8))
plt.show()


#Build a forest and compute the feature importances
from sklearn.ensemble import ExtraTreesClassifier

forest = ExtraTreesClassifier(n_estimators=250,
                              random_state=0)
forest.fit(X, y)

#compute the accuracy score of the random forest
pred_test_forest = forest.predict(Xvl)
score_forest = accuracy_score(yvl, pred_test_forest)
print('accuracy_score of random forest', score_forest)

importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")

for f in range(X.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

# Plot the feature importances of the forest
plt.figure()
plt.title("Feature importances")
plt.bar(range(X.shape[1]), importances[indices],
       color="r", yerr=std[indices], align="center")
plt.xticks(range(X.shape[1]), indices)
plt.xlim([-1, X.shape[1]])
plt.show()

############################################### End Decision Tree ####################################################################


############################################### Additional Exploration (disregard) ################################################


sf = pd.read_csv("/Users/ximenamartinez/Desktop/San_Francisco_Communitywide_Greenhouse_Gas_Inventory.csv")


sf_train, sf_test = train_test_split(sf, test_size = 0.5, random_state = 8)

sf.shape
sf_train.shape
sf_test.shape


#################### The next few blocks will calculate total emmisions and seperate by it's respective type #######################

diesel = sf[sf.Commodity_Type == 'Diesel']#Seperates the data assosiated with diesel
total1 = [0,0,0,0,0,0,0,0]#Sets up array of zeros

diesel1 = diesel[diesel.Calendar_Year == 1990]#Further seperates the data by year
total1[0]= diesel1['Emissions_mtCO2e'].sum()# Claculates sum
total1[0]#check

diesel2 = diesel[diesel.Calendar_Year == 2000]
total1[1]= diesel2['Emissions_mtCO2e'].sum()
total1[1]

diesel3 = diesel[diesel.Calendar_Year == 2005]
total1[2]= diesel3['Emissions_mtCO2e'].sum()
total1[2]

diesel4 = diesel[diesel.Calendar_Year == 2010]
total1[3]= diesel4['Emissions_mtCO2e'].sum()
total1[3]

diesel5 = diesel[diesel.Calendar_Year == 2012]
total1[4]= diesel5['Emissions_mtCO2e'].sum()
total1[4]

diesel6 = diesel[diesel.Calendar_Year == 2015]
total1[5]= diesel6['Emissions_mtCO2e'].sum()
total1[5]

diesel7 = diesel[diesel.Calendar_Year == 2016]
total1[6]= diesel7['Emissions_mtCO2e'].sum()
total1[6]

diesel8 = diesel[diesel.Calendar_Year == 2017]
total1[7]= diesel8['Emissions_mtCO2e'].sum()
total1[7]
total1





########

B100 = sf[sf.Commodity_Type == 'B100']#Seperates the data assosiated with
total2 = [0,0,0,0,0,0,0,0]#Sets up array of zeros


B100_1 = B100[B100.Calendar_Year == 1990]
total2[0]= B100_1['Emissions_mtCO2e'].sum()
total2[0]

B100_2 = B100[B100.Calendar_Year == 2000]
total2[1]= B100_2['Emissions_mtCO2e'].sum()
total2[1]

B100_3 = B100[B100.Calendar_Year == 2005]
total2[2]= B100_3['Emissions_mtCO2e'].sum()
total2[2]

B100_4 = B100[B100.Calendar_Year == 2010]
total2[3]= B100_4['Emissions_mtCO2e'].sum()
total2[3]

B100_5 = B100[B100.Calendar_Year == 2012]
total2[4]= B100_5['Emissions_mtCO2e'].sum()
total2[4]

B100_6 = B100[B100.Calendar_Year == 2015]
total2[5]= B100_6['Emissions_mtCO2e'].sum()
total2[5]

B100_7 = B100[B100.Calendar_Year == 2016]
total2[6]= B100_7['Emissions_mtCO2e'].sum()
total2[6]

B100_8 = B100[B100.Calendar_Year == 2017]
total2[7]= B100_8['Emissions_mtCO2e'].sum()
total2[7]
total2






######

B20 = sf[sf.Commodity_Type == 'B20']#Seperates the data assosiated with
total3 = [0,0,0,0,0,0,0,0]


B20_1 = B20[B20.Calendar_Year == 1990]
total3[0]= B20_1['Emissions_mtCO2e'].sum()
total3[0]

B20_2 = B20[B20.Calendar_Year == 2000]
total3[1]= B20_2['Emissions_mtCO2e'].sum()
total3[1]

B20_3 = B20[B20.Calendar_Year == 2005]
total3[2]= B20_3['Emissions_mtCO2e'].sum()
total3[2]

B20_4 = B20[B20.Calendar_Year == 2010]
total3[3]= B20_4['Emissions_mtCO2e'].sum()
total3[3]

B20_5 = B20[B20.Calendar_Year == 2012]
total3[4]= B20_5['Emissions_mtCO2e'].sum()
total3[4]

B20_6 = B20[B20.Calendar_Year == 2015]
total3[5]= B20_6['Emissions_mtCO2e'].sum()
total3[5]

B20_7 = B20[B20.Calendar_Year == 2016]
total3[6]= B20_7['Emissions_mtCO2e'].sum()
total3[6]

B20_8 = B20[B20.Calendar_Year == 2017]
total3[7]= B20_8['Emissions_mtCO2e'].sum()
total3[7]
total3






#########
B5 = sf[sf.Commodity_Type == 'B5']#Seperates the data assosiated with
total4 = [0,0,0,0,0,0,0,0]#Sets up array of zeros

B5_1 = B5[B5.Calendar_Year == 1990]
total4[0]= B5_1['Emissions_mtCO2e'].sum()
total4[0]

B5_2 = B5[B5.Calendar_Year == 2000]
total4[1]= B5_2['Emissions_mtCO2e'].sum()
total4[1]

B5_3 = B5[B5.Calendar_Year == 2005]
total4[2]= B5_3['Emissions_mtCO2e'].sum()
total4[2]

B5_4 = B5[B5.Calendar_Year == 2010]
total4[3]= B5_4['Emissions_mtCO2e'].sum()
total4[3]

B5_5 = B5[B5.Calendar_Year == 2012]
total4[4]= B5_5['Emissions_mtCO2e'].sum()
total4[4]

B5_6 = B5[B5.Calendar_Year == 2015]
total4[5]= B5_6['Emissions_mtCO2e'].sum()
total4[5]

B5_7 = B5[B5.Calendar_Year == 2016]
total4[6]= B5_7['Emissions_mtCO2e'].sum()
total4[6]

B5_8 = B5[B5.Calendar_Year == 2017]
total4[7]= B5_8['Emissions_mtCO2e'].sum()
total4[7]
total4







######

CNG = sf[sf.Commodity_Type == 'CNG']#Seperates the data assosiated with
total5 = [0,0,0,0,0,0,0,0]#Sets up array of zeros

CNG_1 = CNG[CNG.Calendar_Year == 1990]
total5[0]= CNG_1['Emissions_mtCO2e'].sum()
total5[0]

CNG_2 = CNG[CNG.Calendar_Year == 2000]
total5[1]= CNG_2['Emissions_mtCO2e'].sum()
total5[1]

CNG_3 = CNG[CNG.Calendar_Year == 2005]
total5[2]= CNG_3['Emissions_mtCO2e'].sum()
total5[2]

CNG_4 = CNG[CNG.Calendar_Year == 2010]
total5[3]= CNG_4['Emissions_mtCO2e'].sum()
total5[3]

CNG_5 = CNG[CNG.Calendar_Year == 2012]
total5[4]= CNG_5['Emissions_mtCO2e'].sum()
total5[4]

CNG_6 = CNG[CNG.Calendar_Year == 2015]
total5[5]= CNG_6['Emissions_mtCO2e'].sum()
total5[5]

CNG_7 = CNG[CNG.Calendar_Year == 2016]
total5[6]= CNG_7['Emissions_mtCO2e'].sum()
total5[6]

CNG_8 = CNG[CNG.Calendar_Year == 2017]
total5[7]= CNG_8['Emissions_mtCO2e'].sum()
total5[7]
total5






#######
Electricity = sf[sf.Commodity_Type == 'Electricity']#Seperates the data assosiated with
total6 = [0,0,0,0,0,0,0,0]#Sets up array of zeros

Electricity_1 = Electricity[Electricity.Calendar_Year == 1990]
total6[0]= Electricity_1['Emissions_mtCO2e'].sum()
total6[0]

Electricity_2 = Electricity[Electricity.Calendar_Year == 2000]
total6[1]= Electricity_2['Emissions_mtCO2e'].sum()
total6[1]

Electricity_3 = Electricity[Electricity.Calendar_Year == 2005]
total6[2]= Electricity_3['Emissions_mtCO2e'].sum()
total6[2]

Electricity_4 = Electricity[Electricity.Calendar_Year == 2010]
total6[3]= Electricity_4['Emissions_mtCO2e'].sum()
total6[3]

Electricity_5 = Electricity[Electricity.Calendar_Year == 2012]
total6[4]=  Electricity_5['Emissions_mtCO2e'].sum()
total6[4]

Electricity_6 = Electricity[Electricity.Calendar_Year == 2015]
total6[5]= Electricity_6['Emissions_mtCO2e'].sum()
total6[5]

Electricity_7 = Electricity[Electricity.Calendar_Year == 2016]
total6[6]= Electricity_7['Emissions_mtCO2e'].sum()
total6[6]

Electricity_8 = Electricity[Electricity.Calendar_Year == 2017]
total6[7]= Electricity_8['Emissions_mtCO2e'].sum()
total6[7]
total6





########
Fug_Em = sf[sf.Commodity_Type == 'Fugitive Emissions']#Seperates the data assosiated with
total7 = [0,0,0,0,0,0,0,0]#Sets up array of zeros

Fug_Em_1 = Fug_Em[Fug_Em.Calendar_Year == 1990]
total7[0]= Fug_Em_1['Emissions_mtCO2e'].sum()
total7[0]

Fug_Em_2 = Fug_Em[Fug_Em.Calendar_Year == 2000]
total7[1]= Fug_Em_2['Emissions_mtCO2e'].sum()
total7[1]

Fug_Em_3 = Fug_Em[Fug_Em.Calendar_Year == 2005]
total7[2]= Fug_Em_3['Emissions_mtCO2e'].sum()
total7[2]

Fug_Em_4 = Fug_Em[Fug_Em.Calendar_Year == 2010]
total7[3]= Fug_Em_4['Emissions_mtCO2e'].sum()
total7[3]

Fug_Em_5 = Fug_Em[Fug_Em.Calendar_Year == 2012]
total7[4]=  Fug_Em_5['Emissions_mtCO2e'].sum()
total7[4]

Fug_Em_6 = Fug_Em[Fug_Em.Calendar_Year == 2015]
total7[5]= Fug_Em_6['Emissions_mtCO2e'].sum()
total7[5]

Fug_Em_7 = Fug_Em[Fug_Em.Calendar_Year == 2016]
total7[6]= Fug_Em_7['Emissions_mtCO2e'].sum()
total7[6]

Fug_Em_8 = Fug_Em[Fug_Em.Calendar_Year == 2017]
total7[7]= Fug_Em_8['Emissions_mtCO2e'].sum()
total7[7]
total7







#######
Gasoline = sf[sf.Commodity_Type == 'Gasoline']#Seperates the data assosiated with
total8 = [0,0,0,0,0,0,0,0]#Sets up array of zeros

Gasoline_1 = Gasoline[Gasoline.Calendar_Year == 1990]
total8[0]= Gasoline_1['Emissions_mtCO2e'].sum()
total8[0]

Gasoline_2 = Gasoline[Gasoline.Calendar_Year == 2000]
total8[1]= Gasoline_2['Emissions_mtCO2e'].sum()
total8[1]

Gasoline_3 = Gasoline[Gasoline.Calendar_Year == 2005]
total8[2]= Gasoline_3['Emissions_mtCO2e'].sum()
total8[2]

Gasoline_4 = Gasoline[Gasoline.Calendar_Year == 2010]
total8[3]= Gasoline_4['Emissions_mtCO2e'].sum()
total8[3]

Gasoline_5 = Gasoline[Gasoline.Calendar_Year == 2012]
total8[4]=  Gasoline_5['Emissions_mtCO2e'].sum()
total8[4]

Gasoline_6 = Gasoline[Gasoline.Calendar_Year == 2015]
total8[5]= Gasoline_6['Emissions_mtCO2e'].sum()
total8[5]

Gasoline_7 = Gasoline[Gasoline.Calendar_Year == 2016]
total8[6]= Gasoline_7['Emissions_mtCO2e'].sum()
total8[6]

Gasoline_8 = Gasoline[Gasoline.Calendar_Year == 2017]
total8[7]= Gasoline_8['Emissions_mtCO2e'].sum()
total8[7]
total8






######
Natural_Gas = sf[sf.Commodity_Type == 'Natural Gas']#Seperates the data assosiated with
total9 = [0,0,0,0,0,0,0,0]#Sets up array of zeros

Natural_Gas_1 = Natural_Gas[Natural_Gas.Calendar_Year == 1990]
total9[0]= Natural_Gas_1['Emissions_mtCO2e'].sum()
total9[0]

Natural_Gas_2 = Natural_Gas[Natural_Gas.Calendar_Year == 2000]
total9[1]= Natural_Gas_2['Emissions_mtCO2e'].sum()
total9[1]

Natural_Gas_3 = Natural_Gas[Natural_Gas.Calendar_Year == 2005]
total9[2]= Natural_Gas_3['Emissions_mtCO2e'].sum()
total9[2]

Natural_Gas_4 = Natural_Gas[Natural_Gas.Calendar_Year == 2010]
total9[3]= Natural_Gas_4['Emissions_mtCO2e'].sum()
total9[3]

Natural_Gas_5 = Natural_Gas[Natural_Gas.Calendar_Year == 2012]
total9[4]=  Natural_Gas_5['Emissions_mtCO2e'].sum()
total9[4]

Natural_Gas_6 = Natural_Gas[Natural_Gas.Calendar_Year == 2015]
total9[5]= Natural_Gas_6['Emissions_mtCO2e'].sum()
total9[5]

Natural_Gas_7 = Natural_Gas[Natural_Gas.Calendar_Year == 2016]
total9[6]= Natural_Gas_7['Emissions_mtCO2e'].sum()
total9[6]

Natural_Gas_8 = Natural_Gas[Natural_Gas.Calendar_Year == 2017]
total9[7]= Natural_Gas_8['Emissions_mtCO2e'].sum()
total9[7]
total9






#######
Process_Em = sf[sf.Commodity_Type == 'Process Emissions']#Seperates the data assosiated with
total10 = [0,0,0,0,0,0,0,0]#Sets up array of zeros

Process_Em_1 = Process_Em[Process_Em.Calendar_Year == 1990]
total10[0]= Process_Em_1['Emissions_mtCO2e'].sum()
total10[0]

Process_Em_2 = Process_Em[Process_Em.Calendar_Year == 2000]
total10[1]= Process_Em_2['Emissions_mtCO2e'].sum()
total10[1]

Process_Em_3 = Process_Em[Process_Em.Calendar_Year == 2005]
total10[2]= Process_Em_3['Emissions_mtCO2e'].sum()
total10[2]

Process_Em_4 = Process_Em[Process_Em.Calendar_Year == 2010]
total10[3]= Process_Em_4['Emissions_mtCO2e'].sum()
total10[3]

Process_Em_5 = Process_Em[Process_Em.Calendar_Year == 2012]
total10[4]=  Process_Em_5['Emissions_mtCO2e'].sum()
total10[4]

Process_Em_6 = Process_Em[Process_Em.Calendar_Year == 2015]
total10[5]= Process_Em_6['Emissions_mtCO2e'].sum()
total10[5]

Process_Em_7 = Process_Em[Process_Em.Calendar_Year == 2016]
total10[6]= Process_Em_7['Emissions_mtCO2e'].sum()
total10[6]

Process_Em_8 = Process_Em[Process_Em.Calendar_Year == 2017]
total10[7]= Process_Em_8['Emissions_mtCO2e'].sum()
total10[7]
total10








########
Propane = sf[sf.Commodity_Type == 'Propane']#Seperates the data assosiated with
total11 = [0,0,0,0,0,0,0,0]#Sets up array of zeros

Propane_1 = Propane[Propane.Calendar_Year == 1990]
total11[0]= Propane_1['Emissions_mtCO2e'].sum()
total11[0]

Propane_2 = Propane[Propane.Calendar_Year == 2000]
total11[1]= Propane_2['Emissions_mtCO2e'].sum()
total11[1]

Propane_3 = Propane[Propane.Calendar_Year == 2005]
total11[2]= Propane_3['Emissions_mtCO2e'].sum()
total11[2]

Propane_4 = Propane[Propane.Calendar_Year == 2010]
total11[3]= Propane_4['Emissions_mtCO2e'].sum()
total11[3]

Propane_5 = Propane[Propane.Calendar_Year == 2012]
total11[4]=  Propane_5['Emissions_mtCO2e'].sum()
total11[4]

Propane_6 = Propane[Propane.Calendar_Year == 2015]
total11[5]= Propane_6['Emissions_mtCO2e'].sum()
total11[5]

Propane_7 = Propane[Propane.Calendar_Year == 2016]
total11[6]= Propane_7['Emissions_mtCO2e'].sum()
total11[6]

Propane_8 = Propane[Propane.Calendar_Year == 2017]
total11[7]= Propane_8['Emissions_mtCO2e'].sum()
total11[7]
total11





######
Solid_Waste = sf[sf.Commodity_Type == 'Solid Waste']#Seperates the data assosiated with
total12 = [0,0,0,0,0,0,0,0]#Sets up array of zeros

Solid_Waste_1 = Solid_Waste[Solid_Waste.Calendar_Year == 1990]
total12[0]= Solid_Waste_1['Emissions_mtCO2e'].sum()
total12[0]

Solid_Waste_2 = Solid_Waste[Solid_Waste.Calendar_Year == 2000]
total12[1]= Solid_Waste_2['Emissions_mtCO2e'].sum()
total12[1]

Solid_Waste_3 = Solid_Waste[Solid_Waste.Calendar_Year == 2005]
total12[2]= Solid_Waste_3['Emissions_mtCO2e'].sum()
total12[2]

Solid_Waste_4 = Solid_Waste[Solid_Waste.Calendar_Year == 2010]
total12[3]= Solid_Waste_4['Emissions_mtCO2e'].sum()
total12[3]

Solid_Waste_5 = Solid_Waste[Solid_Waste.Calendar_Year == 2012]
total12[4]=  Solid_Waste_5['Emissions_mtCO2e'].sum()
total12[4]

Solid_Waste_6 = Solid_Waste[Solid_Waste.Calendar_Year == 2015]
total12[5]= Solid_Waste_6['Emissions_mtCO2e'].sum()
total12[5]

Solid_Waste_7 = Solid_Waste[Solid_Waste.Calendar_Year == 2016]
total12[6]= Solid_Waste_7['Emissions_mtCO2e'].sum()
total12[6]

Solid_Waste_8 = Solid_Waste[Solid_Waste.Calendar_Year == 2017]
total12[7]= Solid_Waste_8['Emissions_mtCO2e'].sum()
total12[7]
total12







#######
Renewable_CNG = sf[sf.Commodity_Type == 'Renewable CNG']#Seperates the data assosiated with
total13 = [0,0,0,0,0,0,0,0]#Sets up array of zeros

Renewable_CNG_1 = Renewable_CNG[Renewable_CNG.Calendar_Year == 1990]
total13[0]= Renewable_CNG_1['Emissions_mtCO2e'].sum()
total13[0]

Renewable_CNG_2 = Renewable_CNG[Renewable_CNG.Calendar_Year == 2000]
total13[1]= Renewable_CNG_2['Emissions_mtCO2e'].sum()
total13[1]

Renewable_CNG_3 = Renewable_CNG[Renewable_CNG.Calendar_Year == 2005]
total13[2]= Renewable_CNG_3['Emissions_mtCO2e'].sum()
total13[2]

Renewable_CNG_4 = Renewable_CNG[Renewable_CNG.Calendar_Year == 2010]
total13[3]= Renewable_CNG_4['Emissions_mtCO2e'].sum()
total13[3]

Renewable_CNG_5 = Renewable_CNG[Renewable_CNG.Calendar_Year == 2012]
total13[4]=  Renewable_CNG_5['Emissions_mtCO2e'].sum()
total13[4]

Renewable_CNG_6 = Renewable_CNG[Renewable_CNG.Calendar_Year == 2015]
total13[5]= Renewable_CNG_6['Emissions_mtCO2e'].sum()
total13[5]

Renewable_CNG_7 = Renewable_CNG[Renewable_CNG.Calendar_Year == 2016]
total13[6]= Renewable_CNG_7['Emissions_mtCO2e'].sum()
total13[6]

Renewable_CNG_8 = Renewable_CNG[Renewable_CNG.Calendar_Year == 2017]
total13[7]= Renewable_CNG_8['Emissions_mtCO2e'].sum()
total13[7]
total13





######
Renewable_Diesel = sf[sf.Commodity_Type == 'Renewable Diesel']#Seperates the data assosiated with
total14 = [0,0,0,0,0,0,0,0]#Sets up array of zeros

Renewable_Diesel_1 = Renewable_Diesel[Renewable_Diesel.Calendar_Year == 1990]
total14[0]= Renewable_Diesel_1['Emissions_mtCO2e'].sum()
total14[0]

Renewable_Diesel_2 = Renewable_Diesel[Renewable_Diesel.Calendar_Year == 2000]
total14[1]= Renewable_Diesel_2['Emissions_mtCO2e'].sum()
total14[1]

Renewable_Diesel_3 = Renewable_Diesel[Renewable_Diesel.Calendar_Year == 2005]
total14[2]= Renewable_Diesel_3['Emissions_mtCO2e'].sum()
total14[2]

Renewable_Diesel_4 = Renewable_Diesel[Renewable_Diesel.Calendar_Year == 2010]
total14[3]= Renewable_Diesel_4['Emissions_mtCO2e'].sum()
total14[3]

Renewable_Diesel_5 = Renewable_Diesel[Renewable_Diesel.Calendar_Year == 2012]
total14[4]=  Renewable_Diesel_5['Emissions_mtCO2e'].sum()
total14[4]

Renewable_Diesel_6 = Renewable_Diesel[Renewable_Diesel.Calendar_Year == 2015]
total14[5]= Renewable_Diesel_6['Emissions_mtCO2e'].sum()
total14[5]

Renewable_Diesel_7 = Renewable_Diesel[Renewable_Diesel.Calendar_Year == 2016]
total14[6]= Renewable_Diesel_7['Emissions_mtCO2e'].sum()
total14[6]

Renewable_Diesel_8 = Renewable_Diesel[Renewable_Diesel.Calendar_Year == 2017]
total14[7]= Renewable_Diesel_8['Emissions_mtCO2e'].sum()
total14[7]
total14

#####
Sludge_gas = sf[sf.Commodity_Type == 'Sludge gas']#Seperates the data assosiated with
total15 = [0,0,0,0,0,0,0,0]#Sets up array of zeros

Sludge_gas_1 = Sludge_gas[Sludge_gas.Calendar_Year == 1990]
total15[0]= Sludge_gas_1['Emissions_mtCO2e'].sum()
total15[0]

Sludge_gas_2 = Sludge_gas[Sludge_gas.Calendar_Year == 2000]
total15[1]= Sludge_gas_2['Emissions_mtCO2e'].sum()
total15[1]

Sludge_gas_3 = Sludge_gas[Sludge_gas.Calendar_Year == 2005]
total15[2]= Sludge_gas_3['Emissions_mtCO2e'].sum()
total15[2]

Sludge_gas_4 = Sludge_gas[Sludge_gas.Calendar_Year == 2010]
total15[3]= Sludge_gas_4['Emissions_mtCO2e'].sum()
total15[3]

Sludge_gas_5 = Sludge_gas[Sludge_gas.Calendar_Year == 2012]
total15[4]=  Sludge_gas_5['Emissions_mtCO2e'].sum()
total15[4]

Sludge_gas_6 = Sludge_gas[Sludge_gas.Calendar_Year == 2015]
total15[5]= Sludge_gas_6['Emissions_mtCO2e'].sum()
total15[5]

Sludge_gas_7 = Sludge_gas[Sludge_gas.Calendar_Year == 2016]
total15[6]= Sludge_gas_7['Emissions_mtCO2e'].sum()
total15[6]

Sludge_gas_8 = Sludge_gas[Sludge_gas.Calendar_Year == 2017]
total15[7]= Sludge_gas_8['Emissions_mtCO2e'].sum()
total15[7]
total15

#################### End of calculation and seperation #######################



#Plots each category against their emissions
plt.plot(years, total1) #plots tota1 data with years
plt.ylabel('Diesel Emissions_mtCO2e')#Y label
plt.xlabel('Year')#X label
plt.show()#shows plot

plt.plot(years, total2)
plt.ylabel('B100 Emissions_mtCO2e')
plt.xlabel('Year')
plt.show()

plt.plot(years, total3)
plt.ylabel('B20 Emissions_mtCO2e')
plt.xlabel('Year')
plt.show()

plt.plot(years, total4)
plt.ylabel('B5 Emissions_mtCO2e')
plt.xlabel('Year')
plt.show()

plt.plot(years, total5)
plt.ylabel('CNG Emissions_mtCO2e')
plt.xlabel('Year')
plt.show()

plt.plot(years, total6)
plt.ylabel('Electricity Emissions_mtCO2e')
plt.xlabel('Year')
plt.show()

plt.plot(years, total7)
plt.ylabel('Fugitive Emissions_mtCO2e')
plt.xlabel('Year')
plt.show()

plt.plot(years, total8)
plt.ylabel('Gasoline Emissions_mtCO2e')
plt.xlabel('Year')
plt.show()

plt.plot(years, total9)
plt.ylabel('Natural Gas Emissions_mtCO2e')
plt.xlabel('Year')
plt.show()

plt.plot(years, total10)
plt.ylabel('Process Emissions_mtCO2e')
plt.xlabel('Year')
plt.xlab()

plt.plot(years, total11)
plt.ylabel('Propane Emissions_mtCO2e')
plt.xlabel('Year')
plt.show()

plt.plot(years, total12)
plt.ylabel('Solid Waste Emissions_mtCO2e')
plt.xlabel('Year')
plt.show()

plt.plot(years, total13)
plt.ylabel('Renewable CNG Emissions_mtCO2e')
plt.xlabel('Year')
plt.show()

plt.plot(years, total14)
plt.ylabel('Renewable Diesel Emissions_mtCO2e')
plt.xlabel('Year')
plt.show()

plt.plot(years, total15)
plt.ylabel('Fugitive Emissions_mtCO2e')
plt.xlabel('Year')
plt.show()


#Plots all together
plt.plot(years, total1) 
plt.plot(years, total2)
plt.plot(years, total3)
plt.plot(years, total4) 
plt.plot(years, total5) 
plt.plot(years, total6) 
plt.plot(years, total7) 
plt.plot(years, total8) 
plt.plot(years, total9) 
plt.plot(years, total10) 
plt.plot(years, total11) 
plt.plot(years, total12) 
plt.plot(years, total13) 
plt.plot(years, total14) 
plt.plot(years, total15)
plt.ylabel('Emissions_mtCO2e')
plt.xlabel('Year')
plt.show()

array2 = [0,0,0,0,0,0,0,0]
for i in range(8): 
    array2[i] = total1[i]+ total2[i]+ total3[i]+ total4[i]+ total5[i]+ total6[i]+ total7[i]+total8[i]+ total9[i]+ total10[i]+ total11[i]+ total12[i]+ total13[i]+ total14[i]+ total15[i]  

plt.plot(years, array2)
plt.ylabel('Total Emissions_mtCO2e')
plt.xlabel('Year')
plt.show()



#Division of emission by quantitiy and add as new column 
d = sf['Emissions_mtCO2e'].div(sf['Quantity'])
d
sf['Emision per Unit'] = (d).map(str) + " Emissions per "+ sf['Quantity_Units']
sf['Emision per Unit']



#Cross tab of comodity and emision
crosstab_01 = pd.crosstab(sf['Commodity_Type'], sf['Emissions_mtCO2e'])
crosstab_01#counts
#percentages
round(crosstab_01.div(crosstab_01.sum(0), axis = 1) *100,1)
#bar graph
crosstab_01.plot(kind='bar', stacked = True, legend = None)
plt.show()
#normalized graph
norm1 = crosstab_01.div(crosstab_01.sum(1), axis = 0)
norm1.plot(kind='bar', stacked = True, legend = None)
plt.show()




#Cross tab of comodity and emision
crosstab_02 = pd.crosstab(sf['Calendar_Year'], sf['Emissions_mtCO2e'])
crosstab_02#counts
#percentages
round(crosstab_02.div(crosstab_02.sum(0), axis = 1) *100,1)
#bar graph
crosstab_02.plot(kind='bar', stacked = True, legend = None)
plt.show()
#normalized graph
norm2 = crosstab_02.div(crosstab_02.sum(1), axis = 0)
norm2.plot(kind='bar', stacked = True, legend = None)
plt.show()






#Cross tab of comodity and emision
crosstab_03 = pd.crosstab(sf['Sector_General'], sf['Emissions_mtCO2e'])
crosstab_03#counts
#percentages
round(crosstab_03.div(crosstab_03.sum(0), axis = 1) *100,1)
#bar graph
crosstab_03.plot(kind='bar', stacked = True, legend = None)
plt.show()
#normalized graph
norm3 = crosstab_03.div(crosstab_03.sum(1), axis = 0)
norm3.plot(kind='bar', stacked = True, legend = None)
plt.show()




#Cross tab of comodity and emision
crosstab_04 = pd.crosstab(sf['Sector_Detail1'], sf['Emissions_mtCO2e'])
crosstab_04#counts
#percentages
round(crosstab_04.div(crosstab_04.sum(0), axis = 1) *100,1)
#bar graph
crosstab_04.plot(kind='bar', stacked = True, legend = None)
plt.show()
#normalized graph
norm4 = crosstab_01.div(crosstab_04.sum(1), axis = 0)
norm4.plot(kind='bar', stacked = True, legend = None)
plt.show()



#Cross tab of comodity and emision
crosstab_05 = pd.crosstab(sf['Sector_Detail2'], sf['Emissions_mtCO2e'])
crosstab_05#counts
#percentages
round(crosstab_05.div(crosstab_05.sum(0), axis = 1) *100,1)
#bar graph
crosstab_05.plot(kind='bar', stacked = True, legend = None)
plt.show()
#normalized graph
norm5 = crosstab_05.div(crosstab_05.sum(1), axis = 0)
norm5.plot(kind='bar', stacked = True, legend = None)
plt.show()





#Cross tab of comodity and emision
crosstab_06 = pd.crosstab(sf['Sector_GPC'], sf['Emissions_mtCO2e'])
crosstab_06#counts
#percentages
round(crosstab_06.div(crosstab_06.sum(0), axis = 1) *100,1)
#bar graph
crosstab_06.plot(kind='bar', stacked = True, legend = None)
plt.show()
#normalized graph
norm6 = crosstab_06.div(crosstab_06.sum(1), axis = 0)
norm6.plot(kind='bar', stacked = True, legend = None)
plt.show()





#Cross tab of comodity and emision
crosstab_07 = pd.crosstab(sf['Sector_GPC_Detail'], sf['Emissions_mtCO2e'])
crosstab_07#counts
#percentages
round(crosstab_07.div(crosstab_07.sum(0), axis = 1) *100,1)
#bar graph
crosstab_07.plot(kind='bar', stacked = True, legend = None)
plt.show()
#normalized graph
norm7 = crosstab_07.div(crosstab_07.sum(1), axis = 0)
norm7.plot(kind='bar', stacked = True, legend = None)
plt.show()





#Cross tab of comodity and emision
crosstab_08 = pd.crosstab(sf['Emissions_mtCO2e'], sf['Emissions_mtCO2e'])
crosstab_08#counts
#percentages
round(crosstab_08.div(crosstab_08.sum(0), axis = 1) *100,1)
#bar graph
crosstab_08.plot(kind='bar', stacked = True, legend = None)
plt.show()
#normalized graph
norm8 = crosstab_08.div(crosstab_08.sum(1), axis = 0)
norm8.plot(kind='bar', stacked = True, legend = None)
plt.show()




#Cross tab of comodity and emision
crosstab_09 = pd.crosstab(sf['Biogenic_Emissions_mtCO2e'], sf['Emissions_mtCO2e'])
crosstab_09#counts
#percentages
round(crosstab_09.div(crosstab_09.sum(0), axis = 1) *100,1)
#bar graph
crosstab_09.plot(kind='bar', stacked = True, legend = None)
plt.show()
#normalized graph
norm9 = crosstab_09.div(crosstab_09.sum(1), axis = 0)
norm9.plot(kind='bar', stacked = True, legend = None)
plt.show()




#Cross tab of comodity and emision
crosstab_10 = pd.crosstab(sf['Quantity'], sf['Emissions_mtCO2e'])
crosstab_10#counts
#percentages
round(crosstab_10.div(crosstab_10.sum(0), axis = 1) *100,1)
#bar graph
crosstab_10.plot(kind='bar', stacked = True, legend = None)
plt.show()
#normalized graph
norm10 = crosstab_10.div(crosstab_10.sum(1), axis = 0)
norm10.plot(kind='bar', stacked = True, legend = None)
plt.show()



#Cross tab of comodity and emision
crosstab_11 = pd.crosstab(sf['Quantity_Units'], sf['Emissions_mtCO2e'])
crosstab_11#counts
#percentages
round(crosstab_11.div(crosstab_11.sum(0), axis = 1) *100,1)
#bar graph
crosstab_11.plot(kind='bar', stacked = True, legend = None)
plt.show()
#normalized graph
norm11 = crosstab_11.div(crosstab_11.sum(1), axis = 0)
norm11.plot(kind='bar', stacked = True, legend = None)
plt.show()



#Cross tab of comodity and emision
crosstab_12 = pd.crosstab(sf['Emissions_binned'], sf['Emissions_mtCO2e'])
crosstab_12#counts
#percentages
round(crosstab_12.div(crosstab_12.sum(0), axis = 1) *100,1)
#bar graph
crosstab_12.plot(kind='bar', stacked = True, legend = None)
plt.show()
#normalized graph
norm12 = crosstab_12.div(crosstab_12.sum(1), axis = 0)
norm12.plot(kind='bar', stacked = True, legend = None)
plt.show()






#sorts by assending emision per quantity
sf1 = sf.sort_values(by = ['Emision per Quantitiy'],ascending = True)
sf1['Commodity_Type']
sf1





#Bins emmisions and plot
sf['Emissions_binned'] = pd.cut(x = sf['Emissions_mtCO2e'], bins = [0, 500000, 1000000, 1500000, 2000000, 2500000], labels=["Under 500000", "500000 to 1000000", "1000000 to 1500000", "1500000 to 2500000", "Over 2500000"], right = False)
crosstab_02 = pd.crosstab(sf['Emissions_binned'], sf['Commodity_Type'])
crosstab_02.plot(kind='bar', stacked = True, title = 'Commodity Type with Emissions (Binned) Overlay', legend = None)


####################################### End Disregarded #########################################





