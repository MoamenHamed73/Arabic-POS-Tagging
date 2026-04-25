import streamlit as st
import numpy as np
import pickle
import json
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# -----------------------------
# Load saved model + tokenizers
# -----------------------------
@st.cache_resource
def load_artifacts():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    model = load_model(os.path.join(base_dir, "pos_bilstm_model.keras"))

    with open(os.path.join(base_dir, "word_tokenizer.pkl"), "rb") as f:
        word_tokenizer = pickle.load(f)

    with open(os.path.join(base_dir, "tag_tokenizer.pkl"), "rb") as f:
        tag_tokenizer = pickle.load(f)

    with open(os.path.join(base_dir, "config.json"), "r", encoding="utf-8") as f:
        config = json.load(f)

    max_len = config["MAX_LEN"]

    return model, word_tokenizer, tag_tokenizer, max_len


model, word_tokenizer, tag_tokenizer, MAX_LEN = load_artifacts()
idx2tag = {v: k for k, v in tag_tokenizer.word_index.items()}


# -----------------------------
# Prediction Function
# -----------------------------
def predict_pos(sentence):
    words = sentence.strip().split()

    if not words:
        return []

    seq = word_tokenizer.texts_to_sequences([words])
    padded = pad_sequences(
        seq,
        maxlen=MAX_LEN,
        padding="post",
        truncating="post"
    )

    pred = model.predict(padded, verbose=0)
    pred = np.argmax(pred, axis=-1)[0]

    results = []
    for word, pred_idx in zip(words, pred):
        tag = idx2tag.get(pred_idx, "O")
        results.append((word, tag))

    return results


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(
    page_title="Arabic POS Tagging",
    page_icon="🧠",
    layout="centered"
)

st.title("🧠 Arabic POS Tagging using BiLSTM")
st.markdown("Predict Part-of-Speech (POS) tags for Arabic text using a trained BiLSTM model.")

sentence = st.text_area(
    "Enter Arabic Sentence:",
    placeholder="مثال: برلين ترفض حصول شركة أمريكية على رخصة تصنيع دبابة"
)

if st.button("Predict POS Tags"):
    if not sentence.strip():
        st.warning("Please enter a sentence first.")
    else:
        results = predict_pos(sentence)

        st.subheader("Prediction Results")

        for word, tag in results:
            st.write(f"**{word}** → {tag}")

        st.subheader("Tabular View")
        st.table({
            "Word": [w for w, _ in results],
            "POS Tag": [t for _, t in results]
        })


st.markdown("---")
st.caption("Built with Streamlit + TensorFlow + BiLSTM for Arabic POS Tagging")
