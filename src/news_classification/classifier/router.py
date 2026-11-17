from fastapi import APIRouter, HTTPException, status

from news_classification.classifier.schemas import ClassifierIn, ClassifierOut
from news_classification.classifier.service import predict

classifier_router = APIRouter(prefix="/classifier")

@classifier_router.post("/")
async def classify(classifier_in: ClassifierIn) -> ClassifierOut:
    try:
        category_id, category_name = predict(classifier_in.text)
        return ClassifierOut(category_id=category_id, category_name=category_name)
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
