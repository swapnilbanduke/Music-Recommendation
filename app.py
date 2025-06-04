"""
Emotion-Based Music Recommender – Streamlit
------------------------------------------
Run locally:
    pip install -r requirements.txt
    streamlit run app.py

Folder structure (relative to this file):
├── model
│   ├── CNN_Model.h5
│   └── ResNet50V2_Model.h5
├── dataset
│   └── data_moods.csv   # cols: name/​title, artist, mood, preview_url, …
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from tensorflow.keras.models import load_model

# ──────────────────────────────────────────────
# Config & paths (independent of CWD)
# ──────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
MODEL_DIR  = BASE_DIR / "model"
DATA_PATH  = BASE_DIR / "dataset" / "data_moods.csv"
DEFAULT_MODEL = "ResNet50V2_Model.h5"

CLASS_NAMES = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Sad",
    "Surprise",
    "Neutral",
]

# Map CSV moods → model moods
MOOD_ALIASES = {
    "energetic": "happy",
    "calm": "neutral",
    "melancholy": "sad",
    "relaxed": "neutral",
}

DISPLAY_COLS = ["name", "artist", "popularity", "preview_url"]  # order

# ──────────────────────────────────────────────
# Sidebar – model selector
# ──────────────────────────────────────────────
st.sidebar.title("🎶 Emotion-Based Music Recommender")

available_models = [p.name for p in MODEL_DIR.glob("*.h5")]
if not available_models:
    st.sidebar.error("No .h5 models found inside `/model`.")
    st.stop()

model_name = st.sidebar.selectbox(
    "Choose a model",
    options=available_models,
    index=available_models.index(DEFAULT_MODEL)
    if DEFAULT_MODEL in available_models
    else 0,
)

# cache the heavy model load
@st.cache_resource(show_spinner="Loading model…")
def load_selected_model(path: Path):
    return load_model(path, compile=False)  # compile=False → fewer warnings

model = load_selected_model(MODEL_DIR / model_name)

_, H, W, C = model.input_shape  # (None, H, W, C)
st.sidebar.markdown(
    f"**Input shape expected by {model_name}:** `{H}×{W}×{C}`",
)

# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────
def preprocess_image(img: Image.Image) -> np.ndarray:
    """Resize + scale → (1, H, W, C)"""
    img = img.convert("L" if C == 1 else "RGB").resize((W, H))
    arr = np.array(img).astype("float32") / 255.0
    if C == 1:
        arr = np.expand_dims(arr, -1)
    return np.expand_dims(arr, 0)

def make_clickable(url: str | None) -> str:
    return f'<a href="{url}" target="_blank">Play ▶️</a>' if url else ""

# ──────────────────────────────────────────────
# UI – upload, predict, recommend
# ──────────────────────────────────────────────
st.title("Upload a face image to get mood-based song suggestions")

uploaded = st.file_uploader(
    "Choose an image (jpg / png)",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=False,
)

# ╭──────────────────────────────────────────────────────────────╮
# │                    NEW  helper function                     │
# ╰──────────────────────────────────────────────────────────────╯
def recommend_songs(pred_class: str, df: pd.DataFrame) -> pd.DataFrame:
    """Return top-5 songs for the *mapped* target mood."""
    mood_map = {
        "disgust":   "sad",
        "happy":     "happy",
        "sad":       "happy",
        "fear":      "calm",
        "angry":     "calm",
        "surprise":  "energetic",
        "neutral":   "energetic",

    }
    target = mood_map.get(pred_class.lower())
    if not target:
        return pd.DataFrame()

    # filter & rank
    songs = df[df["mood"] == target]          # df.mood already lower-case
    if songs.empty:
        return songs
    songs = (
        songs.sort_values("popularity", ascending=False)
             .head(5)
             .reset_index(drop=True)
    )
    return songs

# ---------------  main logic  ---------------
if uploaded:
    image = Image.open(uploaded)
    st.image(image, caption="Original upload", width=250)

    preds   = model.predict(preprocess_image(image), verbose=0)[0]
    idx     = int(np.argmax(preds))
    conf    = float(preds[idx])
    emotion = CLASS_NAMES[idx]

    # confidence gate
    CONF_THRESHOLD = 0.50
    if conf < CONF_THRESHOLD:
        st.warning(
            f"I am not confidant about your expressions as confidence level is  {conf*100:.1f}% – "
            "try a clearer real face image or better lighting."
        )
        st.stop()            # nothing else runs

    st.subheader(f"Detected emotion: **{emotion}** – {conf*100:.1f}% confidence")

    # load & tidy catalogue
    if DATA_PATH.exists():
        df = pd.read_csv(DATA_PATH)
        if "mood" not in df.columns:
            st.warning("`mood` column missing in CSV.")
        else:
            df["mood"] = (
                df["mood"]
                  .astype(str)
                  .str.strip()
                  .str.lower()
                  .replace(MOOD_ALIASES)      # Energetic→happy, etc.
            )
            # normalise 'title'→'name'
            if "title" in df.columns and "name" not in df.columns:
                df = df.rename(columns={"title": "name"})

            tracks = recommend_songs(emotion, df)

            if tracks.empty:
                st.info(f"🙁 No songs mapped for {emotion}.")
            else:
                # make Play link if preview_url exists
                if "preview_url" in tracks.columns:
                    tracks["Play"] = tracks["preview_url"].apply(make_clickable)

                order = [c for c in ["name", "artist", "popularity", "Play"] if c in tracks.columns]
                st.success("Here are some suggestions:")
                st.write(tracks[order].to_html(escape=False, index=False), unsafe_allow_html=True)
    else:
        st.warning("Place your catalogue at `dataset/data_moods.csv`.")
