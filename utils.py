import os
import pandas as pd


    
def model_raw(x:str):
    list_model = x.split(' ')
    return list_model[0]

def path_all():
    return os.getcwd()

def handle_nas(df,model=None, brand= None, col_fill=None):
    if col_fill in ['manufacturer', 'model']:
        return 'otros'
    elif col_fill in ['type', 'cylinders', 'drive', 'size']:
        cond = (df['model']==model) & (df['manufacturer']==brand)
        if df.loc[cond,col_fill].shape[0]==0:
            return 'otros'
        else:
            var = df.loc[cond,col_fill].iloc[0]
            return var

def normalization(x, col):
    path = path_all()
    path = path+r'\data\external\valores_stant.csv'
    df_vals = pd.read_csv(path)
    mean= df_vals.loc[df_vals['col']==col,'media'].iloc[0]
    std = df_vals.loc[df_vals['col']==col,'desviacion'].iloc[0]
    val = (x- mean)/std
    return val


def odometer(x):
    if x <0:
        return 'no_data'
    elif x >=0 and x<= 30000: 
        return '0-30ml'
    elif x>30000 and x <= 60000:
        return '31-60ml'
    elif x > 60000 and x <= 90000:
        return '61-90ml'
    elif x > 90000 and x <= 120000:
        return '91-120ml' 
    elif x > 120000 and x <= 150000:
        return '121mil-150mil'
    else: 
        return'+151ml'