import os
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
cwd = os.path.dirname(__file__)
path_to_data = os.path.join(cwd, "data", "housing.csv")

housing = pd.read_csv(path_to_data)


enc = pd.get_dummies(housing['ocean_proximity'], drop_first= True, dtype='float')

housing_encoded = housing.merge(enc, 'inner', left_index=True, right_index=True)
housing_encoded.drop('ocean_proximity',axis=1,  inplace= True)
scaler = StandardScaler()
scaler.set_output(transform='pandas')
housing_transformed = scaler.fit_transform(housing_encoded)
print(housing_transformed)
print(scaler.get_feature_names_out())

housing_encoded.to_csv(os.path.join(cwd, "data", "housing_processed.csv"))

