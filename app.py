
import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import json
from PIL import Image


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="نباتات | NABATAT",
    page_icon="🌿",
    layout="centered",
    initial_sidebar_state="collapsed"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
    <style>

    /* Main page */

    .main {
        padding-top: 1rem;
    }


    /* NABATAT title */

    .nabatat-title {
        text-align: center;
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 0px;
    }


    .nabatat-subtitle {
        text-align: center;
        font-size: 22px;
        margin-top: 0px;
    }


    .description {
        text-align: center;
        font-size: 16px;
        margin-bottom: 30px;
    }


    /* Result card */

    .result-card {
        padding: 20px;
        border-radius: 15px;
        border: 1px solid #dddddd;
        margin-top: 20px;
    }


    .result-title {
        font-size: 26px;
        font-weight: 700;
        margin-bottom: 15px;
    }


    .species-name {
        font-size: 24px;
        font-weight: 600;
    }


    .arabic-name {
        font-size: 24px;
        direction: rtl;
        text-align: right;
    }


    .confidence {
        font-size: 18px;
        font-weight: 600;
    }


    /* Footer */

    .footer {
        text-align: center;
        margin-top: 40px;
        padding: 20px;
        font-size: 14px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# FILE PATHS
# =========================================================

MODEL_PATH = "model/plant_species_final_model.keras"

CLASS_NAMES_PATH = (
    "model/plant_species_class_names.json"
)

INFO_PATH = (
    "data/Species_names_Arabic.xlsx"
)


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():

    return tf.keras.models.load_model(
        MODEL_PATH
    )


# =========================================================
# LOAD CLASS NAMES
# =========================================================

@st.cache_data
def load_class_names():

    with open(
        CLASS_NAMES_PATH,
        "r",
        encoding="utf-8"
    ) as f:

        classes = json.load(f)

    return [
        str(x).strip()
        for x in classes
    ]


# =========================================================
# LOAD SPECIES INFORMATION
# =========================================================

@st.cache_data
def load_species_database():

    df = pd.read_excel(
        INFO_PATH
    )

    # Standardize column names

    df.columns = (
        df.columns
        .astype(str)
        .str.strip()
        .str.lower()
    )

    # Clean model class

    df["model_class"] = (
        df["model_class"]
        .astype(str)
        .str.strip()
    )

    return df


# =========================================================
# LOAD EVERYTHING
# =========================================================

model = load_model()

class_names = load_class_names()

species_df = load_species_database()


# =========================================================
# PREDICTION FUNCTION
# =========================================================

def identify_plant(image):

    # Convert image

    image = image.convert("RGB")


    # Resize

    image = image.resize(
        (224, 224)
    )


    # Convert to array

    image_array = np.array(
        image
    )


    # Add batch dimension

    image_array = np.expand_dims(
        image_array,
        axis=0
    )


    # Predict

    predictions = model.predict(
        image_array,
        verbose=0
    )[0]


    # Highest probability

    predicted_index = int(
        np.argmax(predictions)
    )


    predicted_class = str(
        class_names[predicted_index]
    ).strip()


    confidence = (
        float(
            predictions[predicted_index]
        )
        * 100
    )


    # Find species in Excel database

    match = species_df[
        species_df["model_class"]
        == predicted_class
    ]


    # =====================================================
    # DATABASE INFORMATION AVAILABLE
    # =====================================================

    if not match.empty:

        row = match.iloc[0]

        scientific_name = str(
            row["scientific_name"]
        )

        arabic_name = str(
            row["arabic_name"]
        )

        family = str(
            row["family"]
        )

        notes = str(
            row["notes"]
        )

    else:

        scientific_name = (
            predicted_class
            .replace("_", " ")
        )

        arabic_name = "Not available"

        family = "Not available"

        notes = (
            "Additional information "
            "is not currently available."
        )


    return (
        scientific_name,
        arabic_name,
        family,
        notes,
        confidence
    )


# =========================================================
# HEADER
# =========================================================

st.markdown(
    '<div class="nabatat-title">🌿 نباتات</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="nabatat-subtitle">NABATAT</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="description">'
    'AI-Powered Plant Species Identification'
    '<br>'
    'التعرف على أنواع النباتات باستخدام الذكاء الاصطناعي'
    '</div>',
    unsafe_allow_html=True
)


# =========================================================
# IMAGE UPLOAD
# =========================================================

st.subheader(
    "📷 Upload or Take a Plant Photograph"
)


uploaded_file = st.file_uploader(
    "Choose a plant photograph",
    type=[
        "jpg",
        "jpeg",
        "png",
        "webp"
    ]
)


# =========================================================
# DISPLAY IMAGE
# =========================================================

if uploaded_file is not None:

    image = Image.open(
        uploaded_file
    )

    st.image(
        image,
        caption="Selected plant photograph",
        width="stretch"
    )


    # =====================================================
    # IDENTIFY BUTTON
    # =====================================================

    identify_button = st.button(
        "🔍 IDENTIFY PLANT",
        type="primary",
        use_container_width=True
    )


    if identify_button:

        with st.spinner(
            "NABATAT is identifying the plant..."
        ):

            (
                scientific_name,
                arabic_name,
                family,
                notes,
                confidence
            ) = identify_plant(image)


        # =================================================
        # RESULT
        # =================================================

        st.markdown(
            '<div class="result-card">',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="result-title">'
            '🌿 Identification Result'
            '</div>',
            unsafe_allow_html=True
        )


        # Scientific name

        st.markdown(
            "**Scientific name**"
        )

        st.markdown(
            f'<div class="species-name">'
            f'{scientific_name}'
            f'</div>',
            unsafe_allow_html=True
        )


        st.divider()


        # Arabic name

        st.markdown(
            "**الاسم العربي**"
        )

        st.markdown(
            f'<div class="arabic-name">'
            f'{arabic_name}'
            f'</div>',
            unsafe_allow_html=True
        )


        st.divider()


        # Family

        st.markdown(
            "**Family**"
        )

        st.write(
            family
        )


        st.divider()


        # Confidence

        st.markdown(
            "**AI confidence**"
        )

        st.markdown(
            f'<div class="confidence">'
            f'{confidence:.2f}%'
            f'</div>',
            unsafe_allow_html=True
        )


        st.progress(
            min(
                confidence / 100,
                1.0
            )
        )


        st.divider()


        # Notes

        st.markdown(
            "**📝 Notes**"
        )

        st.write(
            notes
        )


        st.markdown(
            '</div>',
            unsafe_allow_html=True
        )


# =========================================================
# INFORMATION WHEN NO IMAGE
# =========================================================

else:

    st.info(
        "Upload a plant photograph above "
        "to begin identification."
    )


# =========================================================
# ABOUT NABATAT
# =========================================================

with st.expander(
    "ℹ️ About NABATAT"
):

    st.write(
        """
**نباتات | NABATAT** is an AI-assisted
plant species identification prototype.

The current version is trained to identify
13 plant species.

Additional species and photographs can be
incorporated as the training dataset and
machine-learning model are expanded.

Identification results should be verified
by a qualified botanist.
"""
    )


# =========================================================
# FOOTER
# =========================================================

st.markdown(
    """
    <div class="footer">
    🌿 <b>نباتات | NABATAT</b><br>
    AI-assisted plant species identification
    <br><br>
    Research Prototype
    </div>
    """,
    unsafe_allow_html=True
)
