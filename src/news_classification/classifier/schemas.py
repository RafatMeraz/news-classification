from pydantic import BaseModel


class ClassifierIn(BaseModel):
    text: str

class ClassifierOut(BaseModel):
    category_id: int
    category_name: str