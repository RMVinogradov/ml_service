import joblib
import numpy as np
from pydantic import BaseModel, Field
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
import io
import pandas as pd
 

class Model(BaseModel):
    X: list[str] 

app = FastAPI()
loaded_model = joblib.load("model_baseline.pkl")

 
@app.post("/predict")
def predict_model(model:Model):
   
    string_data = """datetime,Accelerometer1RMS,Accelerometer2RMS,Current,Pressure,Temperature,Thermocouple,Voltage,Volume Flow RateRMS
""" +'\r\n'.join(model.X)
     
    df = pd.read_csv(io.StringIO(string_data), sep = ',', index_col = 'datetime')
    df.drop(columns=['Thermocouple'], inplace = True)
    result = loaded_model.predict(df)
    return {"result": ','.join(map(str,result))}

def main():
    import uvicorn
    uvicorn.run(app, host="127.0.0.1",port =8000)

if __name__ == "__main__":
    main()
