import os

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000/classifier/")

CATEGORY_ICONS = {
    "World": "🌍",
    "Sports": "⚽",
    "Business": "💼",
    "Science/Technology": "🔬",
}

st.set_page_config(page_title="News Classifier", page_icon="📰", layout="centered")

st.title("📰 News Classifier")
st.caption("Classifies news articles into World · Sports · Business · Science/Technology")

st.divider()

text = st.text_area(
    "Paste a news article or headline",
    placeholder="e.g. Apple reports record quarterly earnings driven by iPhone sales...",
    height=180,
)

if st.button("Classify", type="primary", use_container_width=True):
    if not text.strip():
        st.warning("Enter some text first.")
    else:
        with st.spinner("Classifying..."):
            try:
                response = requests.post(API_URL, json={"text": text}, timeout=30)
                response.raise_for_status()
                result = response.json()

                category_name = result["category_name"]
                category_id = result["category_id"]
                icon = CATEGORY_ICONS.get(category_name, "📄")

                st.success("Classification complete")
                col1, col2 = st.columns(2)
                col1.metric("Category", f"{icon} {category_name}")
                col2.metric("Label ID", category_id)

            except requests.exceptions.ConnectionError:
                st.error("Cannot connect to API. Make sure the FastAPI server is running at localhost:8000.")
            except requests.exceptions.HTTPError as e:
                st.error(f"API error: {e.response.json().get('detail', str(e))}")
            except Exception as e:
                st.error(f"Unexpected error: {e}")

st.divider()
st.caption("Model: SVC and TF-IDF (300 features)")
st.caption("Note: Model can improve with more data training and no of features increment.")