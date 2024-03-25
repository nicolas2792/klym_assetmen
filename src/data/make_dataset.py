import logging
from pathlib import Path
import pandas as pd

def fill_mode(group):
    mode_values = group.mode()
    if not mode_values.empty:
        return group.fillna(mode_values.iloc[0])
    else:
        return group
    
def model_raw(x:str):
    list_model = x.split(' ')
    return list_model[0]

def main():
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    path = r'c:\Users\super\OneDrive\Escritorio\klym_project\data'
    # initial data sets
    car_df = pd.read_csv(path+r'\raw\vehicles.csv')
    crashes_df = pd.read_csv(path+r'\raw\crashes_poverty.csv', sep=';')
    counties_df = pd.read_csv(path+r'\raw\counties.csv', sep=';')
    
    # fill missing data
    car_df['cylinders'] = car_df.groupby(['manufacturer', 'model'])['cylinders'].transform(fill_mode)
    car_df['drive'] = car_df.groupby(['manufacturer', 'model'])['drive'].transform(fill_mode)
    car_df['type'] = car_df.groupby(['manufacturer', 'model'])['type'].transform(fill_mode)

    crashes_df = pd.merge(crashes_df,counties_df[['State','Postal\ncode']], on='State', \
    how='left')
    
    crashes_df['Postal\ncode'] = crashes_df['Postal\ncode'].astype(str)
    crashes_df['Postal\ncode'] = crashes_df['Postal\ncode'].str.lower()

    crashes_df = crashes_df.rename(columns={'State':'State_com', 'Postal\ncode':'state',\
    'Number of Crashes': '#crashes', 'Poverty':'poverty'})
    crashes_df = crashes_df[['state', '#crashes', 'poverty']]

    car_df = pd.merge(car_df, crashes_df, on='state', how='left')

    cols_remove =['region', 'vin', 'size', 'paint_color', 'county', 'lat', 'long','condition']
    car_df = car_df.drop(columns=cols_remove)
    
    car_df['model'] = car_df['model'].apply(lambda x: model_raw(str(x)))
    
    for col in car_df.columns:
        if car_df[col].dtype == 'O':
            car_df[col] = car_df[col].fillna('otros')
            car_df[col]= car_df[col].astype('str')
    
    car_df['model_brand'] = car_df['model']+'-'+car_df['manufacturer']

    car_df = car_df.drop(columns=['manufacturer', 'model', 'description', 'odometer'])
    
    car_df.loc[car_df['state']=='dc','poverty']=car_df.loc[car_df['state']=='ma','poverty'].iloc[0]
    car_df.loc[car_df['state']=='dc','#crashes']=car_df.loc[car_df['state']=='ma','#crashes'].iloc[0]

    car_df.to_csv(path+r'\processed\final_data_set.csv', index=False)
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)



    main()
