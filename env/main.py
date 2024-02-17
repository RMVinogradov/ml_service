import joblib
import numpy as np
from pydantic import BaseModel
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
import io


class Model(BaseModel):
    X: list[str]

app = FastAPI()
loaded_model = joblib.load("model_baseline.pkl")

@app.post("/predict")
def predict_model(model:Model):
    #print(Model.X)
    #samples_to_predict = np.array(Model.X).reshape(1,-1)
    string_data = """Accelerometer1RMS, Accelerometer2RMS, Current, Pressure,
       Temperature, 'hermocouple, Voltage, Volume Flow RateRMS,
       anomaly, changepoint""" + '\r\n'.join(Model.X)
    df = pd.read_csv(io.StringIO(string_data), sep = ',', index_col = 'datetime')
    df.drop(columns=['Thermocouple'], inplace = True)
    result = loaded_model.predict(df)
    return {result:result[0]}

def main():
    import uvicorn
    uvicorn.run(app, host="127.0.0.1",port =8000)

if __name__ == "__main__":
    main()
