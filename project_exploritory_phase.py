#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 14:35:03 2019

@author: ximenamartinez
"""


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from scipy import stats
import numpy as np

#Import for model: decision tree
from sklearn import tree

#imported data set
sf = pd.read_csv("/Users/bonbon/downloads/USF/2019 Fall/MSIS678_DataWarehousing/Project/San_Francisco_Communitywide_Greenhouse_Gas_Inventory.csv")




###############################################Exploritory Data Phase :)###############################################################

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


print("###############################################End Exploritory Data Phase :D###############################################################")

print('###############START DECISION TREE#######################')
model = tree.DecisionTreeClassifier(random_state=1)
















