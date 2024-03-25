# -*- coding: utf-8 -*-

import pandas as pd
import pickle
from pycaret.regression import predict_model
import uvicorn
from pydantic import create_model
from utils import path_all, normalization, handle_nas, odometer
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Create the app
app = FastAPI()


class CarData(BaseModel):
    type: str
    year: int
    model: str
    fuel:str
    manufacturer: str
    drive: str
    odometer: int
    poverty:float
    crashes:float
    title_status:str
    transmission:str
    cylinders:str
    
# Define predict function
@app.post("/predict")

async def predict_car_price(car_data: CarData):
    
    try:
        #input_data = car_data.model_dump()

        df = pd.DataFrame([car_data.model_dump()])
        
        for i in df.columns:
            if pd.isna(df[i].loc[0])==True:
                if i in ['model', 'manufacturer']:
                    df[i] = handle_nas(df, col_fill=i)
                else:
                    model = df['model'].iloc[0]
                    brand = df['manufacturer'].iloc[0]
                    df[i] = handle_nas(df, model=model, brand=brand, col_fill=i) 
        
        df['odometer_cat'] = odometer(df['odometer'].iloc[0])
        
        df = df.rename(columns={'crashes':'#crashes'})
    
        for i in ['year', 'poverty', '#crashes']:
            df[i] = normalization(df[i].iloc[0], i)
            
        df = pd.get_dummies(df, columns= ['cylinders', 'transmission', 'title_status', 'odometer_cat', 'type'])   
        df = df.drop(columns=['manufacturer', 'model'])
        
        
                
        path = path_all()
        with open(path+r"\models\model.pkl", "rb") as f:
            model_pred = pickle.load(f)
        
        set_df = set(df.columns.tolist())
        set_mol = set(model_pred.feature_names_in_)
        
        cols_add = list(set_df.difference(set_mol))

        dic_change_cols ={}
        for i in range(len(cols_add)):
            col_n = cols_add[i].split('_')[0]
            col_n = col_n+'_otros'
            dic_change_cols[cols_add[i]]=col_n
            
        df = df.rename(columns=dic_change_cols)
        
        cols_missing = list(set_mol.difference(set_df))
        
        for col  in cols_missing:
            df[col] = False
        
        df = df[list(model_pred.feature_names_in_)]
        
        prediction = model_pred.predict(df)
        
        return {"predicted_price": prediction[0]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(status_code=exc.status_code, content={"message": exc.detail})

@app.exception_handler(Exception)
async def generic_exception_handler(request, exc):
    return JSONResponse(status_code=500, content={"message": "Internal server error"})




if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
