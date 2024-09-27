import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from statsmodels.api import OLS, add_constant
from scipy import stats

cwd = os.path.dirname(__file__)
path_to_data = os.path.join(cwd, "data", "housing_processed.csv")

housing = pd.read_csv(path_to_data)
housing.dropna(inplace=True)
housing.info()
housing.shape
housing.drop(columns=['ocean_proximity', 'Unnamed: 0', 'NEAR BAY'],axis=0,  inplace= True)
features = housing.drop(['median_house_value'], axis=1)
target = housing['median_house_value']


feature_train, feature_test, target_train, target_test = train_test_split(
    features, target, test_size=0.33)



feature_train = add_constant(feature_train)
model2 = OLS(endog=target_train, exog=feature_train).fit()
res_summary = model2.summary()
print(res_summary)

