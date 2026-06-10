from pydantic import BaseModel

class FertilizerInput(BaseModel):
    Nitrogen: float
    Phosphorus: float
    Potassium: float
    pH: float
    Temperature: float
    Crop: str