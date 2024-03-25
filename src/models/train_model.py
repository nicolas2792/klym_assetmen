from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.model_selection import GridSearchCV, RepeatedKFold, train_test_split, cross_val_score
from build_features import feature_selection
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
from sklearn.decomposition import PCA
from pycaret.regression import *
import pickle

path =r'C:\Users\super\OneDrive\Escritorio\klym_project\data\processed\final_data_set.csv'
df = pd.read_csv(path)
df = df.drop(columns='Unnamed: 0')
cat_list = [i for i in df.columns if df[i].dtype=='O']
num_list = [i for i in df.columns if df[i].dtype!='O']

run_test = feature_selection(df)
run_test.potencial_corr(cat_list)
run_test.numercial_selection(num_list)
run_test.cat_selection_var()
run_test.business_variables()
list_variables = run_test.final_selection()

df = df[list_variables]

df = pd.get_dummies(df, run_test.cat_selection_var())
for i in ['year', '#crashes', 'poverty']:
    df[i] =  (df[i]-df[i].mean())/df[i].std()
    

s = setup(data= df, target = 'price', session_id=123)
best = compare_models()
model = tune_model(best)



with open(r"C:\Users\super\OneDrive\Escritorio\klym_project\models\model.pkl", "wb") as f:
    pickle.dump(model, f)
