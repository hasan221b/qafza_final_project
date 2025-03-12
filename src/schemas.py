from pydantic import BaseModel

class TextInput(BaseModel):
    text: str

class PredictionOutput(BaseModel):
    sentiment: str
    confidence: float
    probabilities: dict[str, float]

class PredictionOutput2(BaseModel):
    sentiment: str
    confidence: float
    compound: float
    rank_score: float

class CompoundInput(BaseModel):
    positive_prob: float
    negative_prob: float
    neutral_prob: float

class RankingInput(BaseModel):
    positive_prob: float
    negative_prob: float
    neutral_prob: float
    comp: float
    text: str

