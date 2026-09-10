import streamlit as st
import os
import numpy as np
import time
from PIL import Image
import tensorflow as tf
import pandas as pd
from pathlib import Path
from huggingface_hub import hf_hub_download


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Brain Tumor AI Prediction",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

.stApp {
    background: #eef4f2;
    color: #172331;
    font-family: "Trebuchet MS", "Segoe UI", sans-serif;
}

[data-testid="stHeader"] { background: transparent; }

[data-testid="stSidebar"] {
    background: #17343a;
    border-right: 1px solid rgba(255, 255, 255, 0.08);
}

[data-testid="stSidebar"] * { color: #eaf4f2; }
[data-testid="stSidebar"] hr { border-color: rgba(255, 255, 255, 0.16); }

.block-container {
    max-width: 1180px;
    padding: 3.5rem 3rem 2.5rem;
}

.hero {
    position: relative;
    overflow: hidden;
    padding: 42px 46px 38px;
    border-radius: 4px;
    background: linear-gradient(
        120deg,
        #17343a 0%,
        #0b6d70 67%,
        #0f8980 100%
    );
    color: white;
    margin-bottom: 30px;
    box-shadow: 0 16px 32px rgba(23, 52, 58, 0.16);
}

.hero::after {
    content: "";
    position: absolute;
    width: 220px;
    height: 220px;
    right: -65px;
    top: -90px;
    border: 1px solid rgba(255, 255, 255, 0.25);
    border-radius: 50%;
    box-shadow: 0 0 0 24px rgba(255, 255, 255, 0.04), 0 0 0 48px rgba(255, 255, 255, 0.04);
}

.hero h1 {
    position: relative;
    z-index: 1;
    max-width: 700px;
    font-family: Georgia, serif;
    font-size: clamp(2.2rem, 5vw, 4.1rem);
    line-height: 1.02;
    margin: 12px 0 14px;
}

.hero p {
    position: relative;
    z-index: 1;
    max-width: 580px;
    font-size: 1.05rem;
    line-height: 1.6;
    color: #d9efeb;
    margin: 0;
}

.eyebrow, .section-label {
    color: #a5eee2;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.16em;
    text-transform: uppercase;
}

.eyebrow { position: relative; z-index: 1; }

.section-label {
    color: #0b7d78;
    margin-bottom: 0.3rem;
}

.upload-panel, .result-card, .chart-panel {
    background: #fbfcfa;
    border: 1px solid #dfe7e5;
    border-radius: 4px;
    box-shadow: 0 10px 24px rgba(23, 52, 58, 0.06);
}

.upload-panel {
    padding: 22px 24px 8px;
    border-top: 4px solid #ee765d;
    margin: 10px 0 26px;
}

.upload-panel h3 {
    margin: 0 0 3px;
    color: #172331;
    font-family: Georgia, serif;
    font-size: 1.45rem;
}

.result-card {
    padding: 24px;
    margin-top: 15px;
}

.result-card h2 {
    color: #075b5d;
    font-family: Georgia, serif;
    font-size: 2.25rem;
    margin: 5px 0 10px;
}

.result-card p { color: #657384; }

.snapshot-card {
    display: flex;
    gap: 24px;
    align-items: center;
    padding: 24px;
    background: #fbfcfa;
    border: 1px solid #dfe7e5;
    border-radius: 4px;
    box-shadow: 0 10px 24px rgba(23, 52, 58, 0.06);
}

.confidence-ring {
    position: relative;
    display: grid;
    width: 112px;
    height: 112px;
    flex: 0 0 112px;
    place-items: center;
    border-radius: 50%;
    background: conic-gradient(#0b7d78 var(--confidence), #dfe7e5 0);
}

.confidence-ring::after {
    content: "";
    position: absolute;
    width: 86px;
    height: 86px;
    background: #fbfcfa;
    border-radius: 50%;
}

.confidence-value {
    position: relative;
    z-index: 1;
    color: #075b5d;
    font-family: Georgia, serif;
    font-size: 1.35rem;
    font-weight: 700;
}

.snapshot-copy h3 {
    margin: 0 0 4px;
    color: #075b5d;
    font-family: Georgia, serif;
}

.snapshot-copy p { margin: 0; color: #657384; }

.review-note {
    margin-top: 12px;
    padding-top: 12px;
    border-top: 1px solid #dfe7e5;
    color: #172331;
    font-size: 0.9rem;
}

[data-testid="stFileUploaderDropzone"] {
    background: #f2f8f6;
    border: 1px dashed #8abbb3;
    border-radius: 3px;
}

[data-testid="stMetric"] {
    background: #fbfcfa;
    border: 1px solid #dfe7e5;
    border-radius: 4px;
    padding: 12px 14px;
}

[data-testid="stMetricLabel"] p { color: #657384; }
[data-testid="stMetricValue"] { color: #075b5d; }

.chart-panel { padding: 8px 18px 14px; margin: 10px 0 24px; }

.info-card {
    padding: 20px 24px;
    border-radius: 4px;
    background-color: #f4faf8;
    border-left: 4px solid #0b7d78;
    margin-bottom: 15px;
}

.information-grid {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 16px;
    margin-top: 14px;
}

.information-card {
    min-height: 176px;
    padding: 24px;
    border: 1px solid #dfe7e5;
    border-top: 4px solid #0b7d78;
    border-radius: 4px;
    background: #fbfcfa;
    box-shadow: 0 10px 24px rgba(23, 52, 58, 0.06);
}

.information-card.next-step {
    border-top-color: #ee765d;
    background: #fffaf5;
}

.information-kicker {
    margin-bottom: 10px;
    color: #0b7d78;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.14em;
    text-transform: uppercase;
}

.next-step .information-kicker { color: #c65b47; }

.information-card h3 {
    margin: 0 0 12px;
    color: #172331;
    font-family: Georgia, serif;
    font-size: 1.35rem;
}

.information-card p {
    margin: 0;
    color: #657384;
    font-size: 0.95rem;
    line-height: 1.65;
}

.warning-card {
    padding: 20px 24px;
    border-radius: 4px;
    background-color: #fff7ed;
    border-left: 4px solid #ee765d;
}

.normal-card {
    padding: 20px 24px;
    border-radius: 4px;
    background-color: #edf8f1;
    border-left: 4px solid #3a9a67;
}

.tumor-card {
    padding: 20px 24px;
    border-radius: 4px;
    background-color: #fff1ef;
    border-left: 4px solid #d95b4d;
}

.small-text {
    color: #657384;
    font-size: 14px;
}

.class-item {
    display: flex;
    align-items: center;
    gap: 10px;
    margin: 9px 0;
    color: #eaf4f2;
    font-size: 0.92rem;
}

.class-mark {
    display: inline-grid;
    width: 27px;
    height: 27px;
    flex: 0 0 27px;
    place-items: center;
    border: 1px solid rgba(255, 255, 255, 0.32);
    border-radius: 50%;
    color: #17343a;
    background: #a5eee2;
    font-size: 0.72rem;
    font-weight: 800;
}

.class-mark.normal { background: #b9e7c9; }

.footer-bar {
    margin-top: 2.2rem;
    padding: 1.1rem 1.5rem 1.5rem;
    border-top: 1px solid rgba(23, 52, 58, 0.08);
    background: rgba(255, 255, 255, 0.28);
    border-radius: 10px 10px 0 0;
    text-align: center;
    box-shadow: 0 -6px 18px rgba(23, 52, 58, 0.04);
}

.footer-content {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    gap: 0.7rem;
    flex-wrap: wrap;
    color: #3c5264;
    font-size: 0.82rem;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    font-weight: 700;
}

.footer-badge {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 32px;
    height: 32px;
    border-radius: 50%;
    background: linear-gradient(135deg, #17343a, #0b7d78);
    color: #fff;
    font-size: 1rem;
    box-shadow: 0 8px 20px rgba(11, 125, 120, 0.22);
}

.footer-subtext {
    margin-top: 0.4rem;
    color: #657384;
    font-size: 0.76rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

@media (max-width: 700px) {
    .block-container { padding: 2rem 1rem; }
    .hero { padding: 30px 24px; }
    .snapshot-card { align-items: flex-start; flex-direction: column; }
    .information-grid { grid-template-columns: 1fr; }
    .footer-content { gap: 0.45rem; }
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# HERO SECTION
# ============================================================

st.markdown("""
<div class="hero">

<div class="eyebrow">Research prototype · MRI analysis</div>

<h1>Brain MRI, read with a second set of eyes.</h1>

<p>
An AI-assisted classification tool designed to surface patterns across four brain MRI categories.
</p>

</div>
""", unsafe_allow_html=True)


# ============================================================
# LOAD MODEL
# ============================================================

MODEL_DIR = Path(__file__).parent / "saved_models"
CSV_PATH = Path(__file__).parent / "model_comparison_results.csv"


def normalize_model_name(name):
    return "".join(ch for ch in str(name) if ch.isalnum()).lower()


def find_best_model():
    default_model_path = MODEL_DIR / "InceptionResNetV2.h5"
    default_model_name = "InceptionResNetV2"
    default_model_accuracy = 0.9871

    if not MODEL_DIR.exists():
        return default_model_path, default_model_name, default_model_accuracy

    if not CSV_PATH.exists():
        return default_model_path, default_model_name, default_model_accuracy

    try:
        comparison_df = pd.read_csv(CSV_PATH)
    except Exception:
        return default_model_path, default_model_name, default_model_accuracy

    if comparison_df.empty or "Model" not in comparison_df.columns or "Accuracy" not in comparison_df.columns:
        return default_model_path, default_model_name, default_model_accuracy

    valid_df = comparison_df.dropna(subset=["Model", "Accuracy"]).copy()
    if valid_df.empty:
        return default_model_path, default_model_name, default_model_accuracy

    best_row = valid_df.loc[valid_df["Accuracy"].idxmax()]
    best_model_name = str(best_row["Model"]).strip()
    best_model_accuracy = float(best_row["Accuracy"])

    target_name = normalize_model_name(best_model_name)
    model_files = sorted(MODEL_DIR.glob("*.h5"))

    for model_file in model_files:
        if normalize_model_name(model_file.stem) == target_name:
            return model_file, best_model_name, best_model_accuracy

    for model_file in model_files:
        if target_name in normalize_model_name(model_file.stem):
            return model_file, best_model_name, best_model_accuracy

    return default_model_path, default_model_name, default_model_accuracy


MODEL_PATH, MODEL_NAME, MODEL_ACCURACY = find_best_model()


def get_model_repo_id():
    model_repo_id = os.getenv("HF_MODEL_REPO", "").strip()

    if model_repo_id:
        return model_repo_id

    try:
        return str(st.secrets.get("HF_MODEL_REPO", "")).strip()
    except Exception:
        return ""


def get_hf_token():
    model_token = os.getenv("HF_TOKEN", "").strip()

    if model_token:
        return model_token

    try:
        return str(st.secrets.get("HF_TOKEN", "")).strip()
    except Exception:
        return ""


model_repo_id = get_model_repo_id()

if model_repo_id:
    try:
        MODEL_PATH = Path(
            hf_hub_download(
                repo_id=model_repo_id,
                filename=MODEL_PATH.name,
                repo_type="model",
                token=get_hf_token() or None
            )
        )
    except Exception as e:
        st.error("❌ Model could not be downloaded from Hugging Face.")
        st.code(str(e))
        st.stop()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("🧠 About the system")

    st.write(
        """
        This AI-assisted application analyzes
        brain MRI images and predicts one of
        four categories using a trained deep
        learning model.
        """
    )

    st.divider()

    st.subheader("Supported classes")

    st.markdown(
        """
        <div class="class-item"><span class="class-mark">G</span><span>Glioma</span></div>
        <div class="class-item"><span class="class-mark">M</span><span>Meningioma</span></div>
        <div class="class-item"><span class="class-mark">P</span><span>Pituitary Tumor</span></div>
        <div class="class-item"><span class="class-mark normal">N</span><span>No Tumor</span></div>
        """,
        unsafe_allow_html=True
    )

    st.divider()

    st.subheader("⚙️ Model Information")

    st.write(f"Architecture: {MODEL_NAME}")
    st.write(f"Benchmark Accuracy: {MODEL_ACCURACY:.2%}")
    st.write("Input Size: 224 × 224")
    st.write("Classes: 4")
    st.write("Framework: TensorFlow / Keras")

    st.divider()

    st.warning(
        "This application is intended for "
        "research and educational purposes only. "
        "It is not a substitute for professional "
        "medical diagnosis."
    )


class CustomScaleLayer(tf.keras.layers.Layer):
    def __init__(self, scale, **kwargs):
        super().__init__(**kwargs)
        self.scale = scale

    def get_config(self):
        config = super().get_config()
        config.update({"scale": self.scale})
        return config

    def call(self, inputs):
        return inputs[0] + inputs[1] * self.scale


@st.cache_resource
def load_model():

    return tf.keras.models.load_model(
        MODEL_PATH,
        compile=False,
        custom_objects={"CustomScaleLayer": CustomScaleLayer}
    )


try:

    model = load_model()

except Exception as e:

    st.error(
        "❌ Model could not be loaded."
    )

    st.code(str(e))

    st.stop()


# ============================================================
# CLASS NAMES
# ============================================================

CLASS_NAMES = [
    "Glioma",
    "Meningioma",
    "No Tumor",
    "Pituitary Tumor"
]


# ============================================================
# MEDICAL INFORMATION
# ============================================================

TUMOR_INFO = {

    "Glioma": {

        "description":
        "Gliomas are tumors that originate from glial cells "
        "within the brain or spinal cord. Their behavior can "
        "vary considerably depending on tumor type and grade.",

        "recommendation":
        "The image should be reviewed by a qualified radiologist "
        "or neurologist. Further clinical assessment and "
        "appropriate MRI evaluation may be recommended."
    },


    "Meningioma": {

        "description":
        "Meningiomas develop from the meninges, the protective "
        "membranes surrounding the brain and spinal cord. "
        "Many are slow-growing, although their clinical "
        "significance varies.",

        "recommendation":
        "Consult a qualified healthcare professional for "
        "clinical correlation and evaluation. Additional "
        "imaging or follow-up may be recommended depending "
        "on the clinical situation."
    },


    "Pituitary Tumor": {

        "description":
        "Pituitary tumors develop in or around the pituitary "
        "gland, which plays an important role in hormone "
        "regulation.",

        "recommendation":
        "A healthcare professional may recommend further "
        "evaluation, potentially including endocrine assessment "
        "and dedicated imaging depending on the patient's symptoms."
    },


    "No Tumor": {

        "description":
        "The AI model did not identify features corresponding "
        "to the four tumor categories in this MRI image.",

        "recommendation":
        "This result does not rule out every possible brain "
        "condition. If symptoms or clinical concerns exist, "
        "consult a qualified healthcare professional."
    }
}


# ============================================================
# IMAGE UPLOADER
# ============================================================

st.markdown('<div class="section-label">Start here</div>', unsafe_allow_html=True)
st.markdown('<div class="upload-panel"><h3>Upload a brain MRI</h3><p class="small-text">Use a clear JPG or PNG image for the best model response.</p></div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Upload an MRI image for analysis",
    type=["jpg", "jpeg", "png"]
)


# ============================================================
# MAIN PREDICTION
# ============================================================

if uploaded_file is not None:

    image = Image.open(
        uploaded_file
    ).convert("RGB")

    col1, col2 = st.columns(
        [1, 1]
    )


    # --------------------------------------------------------
    # DISPLAY IMAGE
    # --------------------------------------------------------

    with col1:

        st.markdown('<div class="section-label">Input image</div>', unsafe_allow_html=True)
        st.subheader("Uploaded MRI")

        st.image(
            image,
            caption="Uploaded Brain MRI",
            use_container_width=True
        )


    # --------------------------------------------------------
    # PREPROCESSING
    # --------------------------------------------------------

    img = image.resize(
        (224, 224)
    )

    img_array = np.array(
        img
    ).astype(
        np.float32
    )

    img_array = img_array / 255.0

    img_array = np.expand_dims(
        img_array,
        axis=0
    )


    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    with st.status(
        "Preparing MRI analysis...",
        expanded=True
    ) as analysis_status:

        progress = st.progress(0)
        st.write("Checking uploaded image")
        time.sleep(0.25)
        progress.progress(25)

        st.write(
            f"Best model selected: {MODEL_NAME} "
            f"(benchmark accuracy: {MODEL_ACCURACY:.2%})"
        )
        time.sleep(0.3)
        progress.progress(48)

        st.write("Analyzing MRI pattern")
        prediction = model.predict(
            img_array,
            verbose=0
        )[0]
        progress.progress(86)
        time.sleep(0.25)

        st.write("Preparing result summary")
        progress.progress(100)
        analysis_status.update(
            label="Analysis complete",
            state="complete",
            expanded=False
        )


    predicted_index = np.argmax(
        prediction
    )

    predicted_class = CLASS_NAMES[
        predicted_index
    ]

    confidence = prediction[
        predicted_index
    ]


    # --------------------------------------------------------
    # RESULT
    # --------------------------------------------------------

    with col2:

        st.markdown('<div class="section-label">Model output</div>', unsafe_allow_html=True)
        st.subheader("AI prediction")

        st.markdown(
            f"""
            <div class="result-card">

            <h2>{predicted_class}</h2>

            <p>
            AI Confidence:
            <strong>{confidence:.2%}</strong>
            </p>

            <p>
            Model selected:
            <strong>{MODEL_NAME}</strong>
            </p>

            </div>
            """,
            unsafe_allow_html=True
        )


        # ----------------------------------------------------
        # STATUS
        # ----------------------------------------------------

        if predicted_class == "No Tumor":

            st.success(
                "✅ No tumor pattern detected by the model."
            )

        else:

            st.error(
                f"⚠️ {predicted_class} pattern detected."
            )

        if confidence < 0.65:

            st.warning(
                "Low-confidence result. The model is uncertain, so this image should receive careful professional review."
            )

        else:

            st.info(
                "The model's confidence is above the review threshold. This remains an AI-assisted result, not a diagnosis."
            )


# ============================================================
# PREDICTION SNAPSHOT
# ============================================================

if uploaded_file is not None:

    st.divider()

    st.markdown(
        '<div class="section-label">Decision snapshot</div>',
        unsafe_allow_html=True
    )

    st.subheader("A clearer read on this result")

    confidence_percent = confidence * 100
    confidence_label = "Strong signal" if confidence >= 0.65 else "Needs review"
    review_message = (
        "The leading category is clearly ahead of the alternatives."
        if confidence >= 0.65
        else "The leading category is not decisive; compare the probabilities and seek professional review."
    )

    st.markdown(
        f"""
        <div class="snapshot-card">
        <div class="confidence-ring" style="--confidence: {confidence_percent:.1f}%">
            <div class="confidence-value">{confidence_percent:.0f}%</div>
        </div>
        <div class="snapshot-copy">
            <h3>{confidence_label} · {predicted_class}</h3>
            <p>Probability assigned to the model's leading category.</p>
            <div class="review-note">{review_message}</div>
        </div>
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# PROBABILITY DISTRIBUTION
# ============================================================

if uploaded_file is not None:

    st.divider()

    st.subheader(
        "📊 Prediction Probability"
    )

    probability_df = pd.DataFrame({

        "Class": CLASS_NAMES,

        "Probability": prediction

    })

    probability_df["Probability"] = (
        probability_df["Probability"] * 100
    )

    st.vega_lite_chart(
        probability_df,
        {
            "height": 300,
            "mark": "bar",
            "encoding": {
                "x": {
                    "field": "Class",
                    "type": "nominal",
                    "sort": CLASS_NAMES,
                    "title": None,
                    "axis": {
                        "labelAngle": 0,
                        "labelLimit": 180,
                        "labelOverlap": False
                    }
                },
                "y": {
                    "field": "Probability",
                    "type": "quantitative",
                    "scale": {"domain": [0, 105]},
                    "axis": {
                        "values": [0, 20, 40, 60, 80, 100]
                    },
                    "title": "Probability (%)"
                },
                "color": {
                    "field": "Class",
                    "type": "nominal",
                    "legend": None
                }
            }
        },
        use_container_width=True
    )


# ============================================================
# CLASS-BY-CLASS PROBABILITY
# ============================================================

if uploaded_file is not None:

    cols = st.columns(4)

    for i, class_name in enumerate(
        CLASS_NAMES
    ):

        with cols[i]:

            st.metric(
                class_name,
                f"{prediction[i]*100:.2f}%"
            )


# ============================================================
# MEDICAL INFORMATION
# ============================================================

if uploaded_file is not None:

    st.divider()

    st.markdown(
        '<div class="section-label">Read the result</div>',
        unsafe_allow_html=True
    )

    st.subheader("Result information")

    info = TUMOR_INFO[
        predicted_class
    ]


    category_label = (
        "Model result: no tumor pattern"
        if predicted_class == "No Tumor"
        else f"Model result: {predicted_class}"
    )

    st.markdown(
        f"""
        <div class="information-grid">
        <div class="information-card">
            <div class="information-kicker">Interpretation</div>
            <h3>{category_label}</h3>
            <p>{info["description"]}</p>
        </div>
        <div class="information-card next-step">
            <div class="information-kicker">Recommended next step</div>
            <h3>Continue with clinical context</h3>
            <p>{info["recommendation"]}</p>
        </div>
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# IMPORTANT DISCLAIMER
# ============================================================

st.divider()

st.markdown(
    """
    <div class="warning-card">

    <h3>⚠️ Important Medical Disclaimer</h3>

    <p>
    This application is a research and educational
    prototype developed for automated brain MRI
    classification. The prediction generated by the
    system should not be considered a medical diagnosis.
    MRI interpretation must be performed by qualified
    healthcare professionals using clinical history,
    radiological findings, and other appropriate
    diagnostic information.
    </p>

    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer-bar">
        <div class="footer-content">
            <span class="footer-badge">🧠</span>
            <span>Brain Tumor AI Prediction</span>
        </div>
        <div class="footer-subtext">Deep Learning • TensorFlow • Streamlit</div>
    </div>
    """,
    unsafe_allow_html=True
)