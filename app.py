import html
import re
from pathlib import Path

import streamlit as st
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


MAX_LENGTH = 128
PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATHS = (
    PROJECT_ROOT / "bert-sentiment-model",
)


def clean_text(text):
    text = html.unescape(text)
    text = re.sub(r"<br\s*/?>", " ", text, flags=re.IGNORECASE)
    text = text.replace("\\n", " ").replace("\\t", " ")
    return re.sub(r"\s+", " ", text).strip()


@st.cache_resource
def load_model(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path, local_files_only=True
    ).to("cpu")
    model.eval()
    return tokenizer, model


def predict_sentiment(text, tokenizer, model):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH,
    )
    with torch.inference_mode():
        logits = model(**inputs).logits
        probabilities = torch.softmax(logits, dim=-1)[0].tolist()

    predicted_id = max(range(len(probabilities)), key=probabilities.__getitem__)
    scores = {
        model.config.id2label[index]: probability
        for index, probability in enumerate(probabilities)
    }
    return model.config.id2label[predicted_id], scores


def main():
    st.set_page_config(page_title="BERT Sentiment Classifier", page_icon="💬")
    st.title("BERT Sentiment Classifier")

    model_path = next(
        (path for path in MODEL_PATHS if (path / "config.json").is_file()), None
    )
    if model_path is None:
        st.info(
            "Run the notebook's model-saving cell first. Keep the "
            "bert_sentiment_model folder beside app.py or inside notebooks/."
        )
        st.stop()

    with st.form("sentiment_form"):
        review = st.text_area("Enter a review", height=160)
        submitted = st.form_submit_button("Analyze")

    if submitted:
        text = clean_text(review)
        if not text:
            st.warning("Enter a review before analyzing.")
            st.stop()

        try:
            with st.spinner("Analyzing..."):
                tokenizer, model = load_model(str(model_path))
                sentiment, scores = predict_sentiment(text, tokenizer, model)
        except (OSError, ValueError, RuntimeError) as error:
            st.error(f"Could not analyze the review: {error}")
            st.stop()

        st.metric("Sentiment", sentiment)
        st.write("Class probabilities")
        for label, probability in scores.items():
            st.progress(probability, text=f"{label}: {probability:.1%}")


if __name__ == "__main__":
    main()
