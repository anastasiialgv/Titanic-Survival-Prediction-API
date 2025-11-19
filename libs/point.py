from pydantic import BaseModel

class TitanicFeatures(BaseModel):
    pclass: int
    age: float
    sibsp: int
    parch: int
    fare: float
    embarked: int
    sex: int