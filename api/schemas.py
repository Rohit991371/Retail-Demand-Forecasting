from pydantic import BaseModel, Field

class PredictionRequest(BaseModel):
    store_id: str = Field(..., description="Store identifier")
    product_id: str = Field(..., description="Product identifier")
    
    
class PredictionResponse(BaseModel):
    store_id: str
    product_id: str
    prediction_date: str
    predicted_demand: float