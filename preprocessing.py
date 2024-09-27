import os
import pandas as pd

cwd = os.path.dirname(__file__)
path_to_data = os.path.join(cwd, "data", "housing.csv")

housing = pd.read_csv(path_to_data)


enc = pd.get_dummies(housing['ocean_proximity'], drop_first= True, dtype='float')

housing_enc = housing.merge(enc, 'inner', left_index=True, right_index=True)

housing_enc.to_csv(os.path.join(cwd, "data", "housing_processed.csv"))