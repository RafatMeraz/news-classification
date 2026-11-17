import pickle
import string

import nltk
import mlflow
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

from news_classification.core.logger_utils import get_logger

nltk.download("stopwords", quiet=True)
nltk.download("punkt_tab", quiet=True)

logger = get_logger("classifier.service")

LABEL_MAP = {
    0: "World",
    1: "Sports",
    2: "Business",
    3: "Science/Technology",
}

_stemmer = SnowballStemmer("english")
_stop_words = set(stopwords.words("english"))

_model = None
_vectorizer = None

mlflow.set_tracking_uri("sqlite:///mlflow.db")
client = mlflow.tracking.MlflowClient()


def _transform_text(text: str) -> str:
    text = text.lower()
    tokens = nltk.word_tokenize(text)
    tokens = [t for t in tokens if t.isalnum()]
    tokens = [t for t in tokens if t not in _stop_words and t not in string.punctuation]
    tokens = [_stemmer.stem(t) for t in tokens]
    return " ".join(tokens)


def _load_model():
    global _model, _vectorizer

    versions = client.get_latest_versions("news_classification")
    if not versions:
        raise RuntimeError("No versions found for registered model 'news_classification'")

    latest = max(versions, key=lambda v: int(v.version))
    model_uri = f"models:/news_classification/{latest.version}"
    logger.info(f"Loading registered model version {latest.version} (run {latest.run_id})")
    _model = mlflow.sklearn.load_model(model_uri)

    with open("models/tfidf_vectorizer.pkl", "rb") as f:
        _vectorizer = pickle.load(f)

    logger.info("Model and vectorizer loaded")


def get_model_and_vectorizer():
    if _model is None or _vectorizer is None:
        _load_model()
    return _model, _vectorizer


def predict(text: str) -> tuple[int, str]:
    model, vectorizer = get_model_and_vectorizer()

    transformed = _transform_text(text)
    features = vectorizer.transform([transformed])

    feature_names = vectorizer.get_feature_names_out()
    feature_array = features.toarray()
    import pandas as pd
    X = pd.DataFrame(feature_array, columns=[f"tfidf_{n}" for n in feature_names])

    category_id = int(model.predict(X)[0])
    category_name = LABEL_MAP.get(category_id, "Unknown")
    return category_id, category_name
