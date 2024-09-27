import os
import pandas as pd
from sklearn.model_selection import train_test_split
from statsmodels.api import OLS, add_constant
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

cwd = os.path.dirname(__file__)
path_to_data = os.path.join(cwd, "data", "housing_processed.csv")

housing = pd.read_csv(path_to_data)
housing.dropna(inplace=True)
housing.drop(columns=['Unnamed: 0', 'NEAR BAY'],axis=0,  inplace= True)
features = housing.drop(['median_house_value'], axis=1)
target = housing['median_house_value']
correlation = features.corrwith(target)
correlation = correlation.apply(lambda x: abs(x))
most_correlated = correlation.sort_values(ascending=False).index[0:11]

feature_train, feature_test, target_train, target_test = train_test_split(
    features[most_correlated], target, test_size=0.33)



feature_train = add_constant(feature_train)
model = OLS(endog=target_train, exog=feature_train).fit()
res_summary = model.summary()
pred = model.predict(add_constant(feature_test))
print(res_summary)
print(mean_squared_error(target_test, pred))
print(mean_absolute_percentage_error(target_test, pred))

