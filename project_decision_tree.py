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

Sector_General_np = np.array(sf['Sector_General'])
(Sector_General_cat, Sector_General_cat_dict) = stattools.categorical(Sector_General_np, drop = True, dictnames = True)
inv_map = {v: k for k, v in Sector_General_cat_dict.items()}
Sector_General_cat_pd = sf['Sector_General'].apply(lambda r: inv_map[r])
print(Sector_General_cat_pd.to_string())

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

sf = pd.concat((sf[['Calendar_Year']], Sector_General_cat_pd, Sector_Detail1_cat_pd, Sector_Detail2_cat_pd, Sector_GPC_cat_pd, Sector_GPC_Detail_cat_pd, Commodity_Type_cat_pd,sf[['Quantity']], Quantity_Units_cat_pd, sf[['Emissions_mtCO2e']]), axis = 1)
sf_train, sf_test = train_test_split(sf, test_size = 0.5, random_state = 8)

X = sf_train.drop('Emissions_mtCO2e', 1)
Xvl = sf_test.drop('Emissions_mtCO2e', 1)
y = sf_train.Emissions_mtCO2e
yvl = sf_test.Emissions_mtCO2e

X_names = ["Calendar_Year", " Sector_General", "Sector_Detail1", "Sector_Detail2", "Sector_GPC", "Sector_GPC_Detail", "Commodity_Type", "Quantity", "Quantity_Units"]
y_names = ["Emissions_mtCO2e"]

# y = sf.Emissions_mtCO2e

y_names = [str(i) for i in range(10)]
y = pd.cut(y, 10, labels=y_names)
yvl = pd.cut(yvl, 10, labels=y_names)

cart_sf_train = DecisionTreeClassifier(criterion = "gini", min_samples_leaf = 5).fit(X,y)
export_graphviz(cart_sf_train, out_file = "/Users/bonbon/downloads/USF/2019 Fall/MSIS678_DataWarehousing/Project/cart_sf_train.dot", feature_names=X_names, class_names = y_names)

#importances = pd.Series(cart_sf_train.feature_importances_, index=X.columns)
#importances.plot(kind='barh', figsize=(12,8))
#plt.show()

from sklearn.metrics import accuracy_score
model = DecisionTreeClassifier(random_state = 1)
model.fit(X,y)
pred_test = model.predict(Xvl)
score = accuracy_score(yvl, pred_test)
#print('accuracy_score', score)



from sklearn.ensemble import ExtraTreesClassifier
# Build a forest and compute the feature importances
forest = ExtraTreesClassifier(n_estimators=250,
                              random_state=0)

forest.fit(X, y)
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
#plt.show()



# from sklearn.linear_model import LinearRegression
# X = pd.concat((sf[['Calendar_Year']], Sector_General_cat_pd, Sector_Detail1_cat_pd, Sector_Detail2_cat_pd, Sector_GPC_cat_pd, Sector_GPC_Detail_cat_pd, Commodity_Type_cat_pd,sf[['Quantity']], Quantity_Units_cat_pd), axis = 1)
# y = sf.Emissions_mtCO2e
# reg = LinearRegression().fit(X, y)
# print(reg.score(X, y))
# print(y.to_string())
#
# fig = plt.figure()
# plt.scatter(sf["Calendar_Year"], y)
# plt.show()

#Linear Regression
Xlr = as_matrix(X)
