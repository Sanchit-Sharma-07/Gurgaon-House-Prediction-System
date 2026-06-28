import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

#1 Load The Data
data = pd.read_csv("housing.csv")

# 2. Create a stratified test set based on income category
data["income_cat"] = pd.cut(data["median_income"],bins=[0,1.5,3.0,4.5,6.0,np.inf],labels=[1,2,3,4,5])
split = StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=42)
for train_index,test_index in split.split(data,data["income_cat"]):
  strat_train_set = data.loc[train_index].drop("income_cat",axis = 1)
  strat_test_set = data.loc[test_index].drop("income_cat",axis = 1)

#Work on Copy Data
housing = strat_train_set.copy()

#3 Separate Predictors and labels
housing_labels = housing["median_house_value"].copy()
housing = housing.drop("median_house_value",axis = 1)

#Separate numerical and categorical values
num_attribs = housing.drop("ocean_proximity",axis = 1).columns.tolist()
cat_attribs = ["ocean_proximity"]

#5 Pipelines
#Number Pipeline
num_pipeline = Pipeline([("imputer",SimpleImputer(strategy="median")),("Scaler",StandardScaler())])
#Categorical Pipeline
cat_pipeline = Pipeline([("onehot",OneHotEncoder(handle_unknown="ignore"))])
#Full Pipeline
full_pipeline = ColumnTransformer([("num",num_pipeline,num_attribs),("cat",cat_pipeline,cat_attribs)])

#6 Transform The Data
housing_prepared = full_pipeline.fit_transform(housing)

#7 Train the model

# Linear Regression
lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)
 
# Decision Tree
tree_reg = DecisionTreeRegressor(random_state=42)
tree_reg.fit(housing_prepared, housing_labels)
 
# Random Forest
forest_reg = RandomForestRegressor(random_state=42)
forest_reg.fit(housing_prepared, housing_labels)
 
# Evaluate LinearRegression with cross-validation
lin_rmses = -cross_val_score(
    lin_reg,
    housing_prepared,
    housing_labels,
    scoring="neg_root_mean_squared_error",
    cv=10
)
# Evaluate Decision Tree with cross-validation
tree_rmses = -cross_val_score(
    tree_reg,
    housing_prepared,
    housing_labels,
    scoring="neg_root_mean_squared_error",
    cv=10
)
# Evaluate RandomForest with cross-validation
forest_rmses = -cross_val_score(
    forest_reg,
    housing_prepared,
    housing_labels,
    scoring="neg_root_mean_squared_error",
    cv=10
)
print(pd.Series(lin_rmses).describe())
print(pd.Series(tree_rmses).describe())
print(pd.Series(forest_rmses).describe())
 