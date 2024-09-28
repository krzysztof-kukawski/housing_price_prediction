import os
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
cwd = os.path.dirname(__file__)
path_to_data = os.path.join(cwd, "data", "housing_processed.csv")

housing = pd.read_csv(path_to_data)
housing.dropna(inplace=True)
housing.drop(columns=['Unnamed: 0'],axis=0,  inplace= True)
features = housing.drop(['median_house_value'], axis=1)
target = housing['median_house_value']
correlation = features.corrwith(target)
correlation = correlation.apply(lambda x: abs(x))

most_correlated = correlation.sort_values(ascending=False).index[0:13]

feature_train, feature_test, target_train, target_test = train_test_split(
    features[most_correlated], target, test_size=0.33)

for regressor in [LinearRegression, DecisionTreeRegressor, XGBRegressor, RandomForestRegressor]:
    for scaling in [StandardScaler, MinMaxScaler, RobustScaler]:
        pipe = Pipeline([('scaler', scaling()), ('regressor', regressor())])

        fir = pipe.fit(feature_train, target_train)
        pred = fir.predict(feature_test)
        print(mean_absolute_percentage_error(target_test, pred), scaling.__name__, regressor.__name__)
        print(mean_squared_error(target_test, pred), scaling.__name__, regressor.__name__)