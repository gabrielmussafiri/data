from services.service import PredictionService
import pandas as pd
from pydantic import BaseModel
from fastapi import APIRouter
import os

router = APIRouter()

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


artifact_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', 'artifact', 'titanic_model.pkl')
)
model = PredictionService(model_path=artifact_path)


@router.post('/predict')
async def predict(passenger:PassengerFeatures):
    data = pd.DataFrame([passenger.dict()])

    prediction = model.predict(data)
    return {
        "survived":int(prediction),
        "message": 'Survived' if prediction ==1 else 'Did not survive'
        }