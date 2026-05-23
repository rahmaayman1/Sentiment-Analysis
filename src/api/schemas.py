from pydantic import BaseModel

class PredictRequest(BaseModel):
    text: str

class PredictResponse(BaseModel):
    sentiment  : str
    confidence : float
    model      : str