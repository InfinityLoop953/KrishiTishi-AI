from pydantic import BaseModel

class CropInput(BaseModel):
    Soil: float
    Soil_Moisture: float
    Temperature: float
    Humidity: float