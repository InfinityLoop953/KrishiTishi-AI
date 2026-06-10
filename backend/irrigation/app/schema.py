from pydantic import BaseModel

class IrrigationInput(BaseModel):

    temperature: float
    location: str
    crop_type: str
    season: str
    humidity: float
    rainfall: float
    soil_moisture: float
    soil_ph: float
    light_intensity: float
    fertilizer_used: float