from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd 



app = FastAPI()

# Load Model
model = joblib.load('./model/titanic_model.pkl')

# Define input structure
class PassengerFeatures(BaseModel):
    Pclass: int
    Age: float 
    Fare: float
    FamilySize: int
    Sex_male: int
    Embarked_Q: int
    Embarked_S: int
    Title_Miss: int
    Title_Mr: int
    Title_Mrs: int
    Title_Rare: int

@app.get('/')
async def home():
    return{'message':'Titanic Prediction API'}

@app.post('/predict')
async def predict(passenger:PassengerFeatures):
    data = pd.DataFrame([passenger.dict()])
    prediction = model.predict(data)[0]
    return {
        "survived":int(prediction),
        "message": 'Survived' if prediction ==1 else 'Did not survive'
        }
    

