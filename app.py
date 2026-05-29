from datetime import datetime
from pathlib import Path
from textwrap import dedent
import re

import pandas as pd
import streamlit as st


# =========================
# Page configuration
# =========================
st.set_page_config(
    page_title="Mental Health Text Screening",
    page_icon="🧠",
    layout="wide",
)


# =========================
# App constants
# =========================
APP_TITLE = "Mental Health Text Screening"
BASE_DIR = Path(__file__).parent
LOG_FILE = BASE_DIR / "text_screening_log.csv"
MODEL_DIR = BASE_DIR / "best_model" / "best_model"

TARGET_LABELS = [
    "Anxiety",
    "Bipolar",
    "Stress",
    "Personality Disorder",
    "Normal",
    "Depression",
    "Suicidal",
]

PAGES = [
    "Home",
    "Personal Details",
    "Text Screening",
    "Result",
    "History",
    "About",
    "Finish",
]

LOG_COLUMNS = [
    "timestamp",
    "user_id",
    "age",
    "gender",
    "input_text",
    "word_count",
    "prediction",
    "confidence",
]

DISCLAIMER = (
    "This application is a mental-health text screening support tool. It does not "
    "provide a medical diagnosis, treatment plan, or crisis support. If someone "
    "may be in immediate danger, contact local emergency services or a qualified "
    "mental-health professional."
)


# =========================
# Session state setup
# =========================
DEFAULT_STATE = {
    "page_index": 0,
    "user_id": "",
    "age": 20,
    "gender": "Prefer not to say",
    "input_text": "",
    "cleaned_text": "",
    "prediction_ready": False,
    "prediction": None,
    "confidence": None,
    "probabilities": None,
}

for key, value in DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = value


# =========================
# HTML helper
# =========================
def render_html(html):
    st.markdown(dedent(html).strip(), unsafe_allow_html=True)


# =========================
# Styling
# =========================
def load_custom_css():
    render_html(
        """
        <style>
        .hero-card {
            background: linear-gradient(135deg, #e0f2fe 0%, #f0f9ff 45%, #ecfdf5 100%);
            border: 1px solid #dbeafe;
            border-radius: 24px;
            padding: 34px;
            margin-bottom: 24px;
            box-shadow: 0 10px 30px rgba(15, 23, 42, 0.08);
        }

        .hero-title {
            font-size: 38px;
            font-weight: 800;
            color: #0f172a;
            margin-bottom: 8px;
        }

        .hero-subtitle {
            font-size: 18px;
            color: #334155;
            line-height: 1.6;
            max-width: 980px;
        }

        .mini-badge {
            display: inline-block;
            background: #0ea5e9;
            color: white;
            padding: 6px 12px;
            border-radius: 999px;
            font-size: 13px;
            font-weight: 700;
            margin-bottom: 14px;
        }

        .card-grid {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 16px;
            margin: 20px 0;
        }

        .label-grid {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 12px;
            margin: 18px 0;
        }

        .info-card, .label-card, .journey-card, .status-card, .pipeline-card,
        .soft-panel, .credit-card {
            background: #ffffff;
            border: 1px solid #e2e8f0;
            box-shadow: 0 8px 20px rgba(15, 23, 42, 0.06);
        }

        .info-card {
            border-radius: 18px;
            padding: 20px;
        }

        .label-card {
            border-radius: 16px;
            padding: 14px;
            text-align: center;
            font-weight: 800;
            color: #0f172a;
        }

        .info-card-title {
            font-size: 18px;
            font-weight: 800;
            color: #0f172a;
            margin-bottom: 6px;
        }

        .info-card-text {
            color: #475569;
            font-size: 14px;
            line-height: 1.5;
        }

        .journey-card {
            border-radius: 24px;
            padding: 24px;
            margin-top: 18px;
        }

        .journey-title {
            font-size: 24px;
            font-weight: 900;
            color: #0f172a;
            margin-bottom: 6px;
        }

        .journey-subtitle {
            color: #64748b;
            font-size: 15px;
            margin-bottom: 20px;
        }

        .journey-row {
            display: flex;
            align-items: flex-start;
            gap: 16px;
            padding: 16px 0;
            border-bottom: 1px solid #f1f5f9;
        }

        .journey-row:last-child {
            border-bottom: 0;
        }

        .journey-number {
            min-width: 54px;
            height: 54px;
            border-radius: 16px;
            background: linear-gradient(135deg, #0ea5e9, #22c55e);
            color: white;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 900;
            font-size: 15px;
            box-shadow: 0 6px 14px rgba(14, 165, 233, 0.22);
        }

        .journey-step-title {
            font-size: 18px;
            font-weight: 900;
            color: #0f172a;
            margin-bottom: 4px;
        }

        .journey-step-desc {
            font-size: 15px;
            color: #475569;
            line-height: 1.5;
        }

        .pipeline {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 14px;
            margin: 20px 0;
        }

        .pipeline-card {
            border-radius: 18px;
            padding: 18px;
            text-align: center;
        }

        .pipeline-title {
            color: #0f172a;
            font-weight: 800;
            font-size: 15px;
            margin-bottom: 4px;
        }

        .pipeline-text {
            color: #64748b;
            font-size: 13px;
            line-height: 1.45;
        }

        .soft-note {
            background: #f8fafc;
            border-left: 5px solid #38bdf8;
            padding: 16px 18px;
            border-radius: 14px;
            color: #334155;
            margin: 16px 0;
        }

        .soft-panel {
            border-radius: 22px;
            padding: 22px;
            margin: 18px 0;
        }

        .status-card {
            border-radius: 22px;
            padding: 26px;
            margin: 18px 0;
        }

        .status-title {
            font-size: 28px;
            font-weight: 900;
            margin-bottom: 8px;
            color: #0f172a;
        }

        .status-text {
            color: #334155;
            font-size: 16px;
            line-height: 1.6;
        }

        .status-alert {
            background: linear-gradient(135deg, #fff7ed, #ffedd5);
            border: 1px solid #fed7aa;
        }

        .status-ready {
            background: linear-gradient(135deg, #ecfdf5, #dcfce7);
            border: 1px solid #bbf7d0;
        }

        .credit-card {
            border-radius: 20px;
            padding: 22px;
            margin-top: 20px;
        }

        .credit-title {
            font-size: 20px;
            font-weight: 900;
            color: #0f172a;
            margin-bottom: 10px;
        }

        .credit-text {
            font-size: 16px;
            color: #334155;
            line-height: 1.7;
        }

        @media (max-width: 900px) {
            .card-grid, .label-grid, .pipeline {
                grid-template-columns: 1fr;
            }

            .hero-title {
                font-size: 28px;
            }
        }
        </style>
        """
    )


load_custom_css()


# =========================
# Data helpers
# =========================
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-zA-Z\s']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def format_label(label):
    label = str(label).strip().lower()
    mapping = {
        "anxiety": "Anxiety",
        "bipolar": "Bipolar",
        "stress": "Stress",
        "personality disorder": "Personality Disorder",
        "normal": "Normal",
        "depression": "Depression",
        "suicidal": "Suicidal",
    }
    return mapping.get(label, str(label).title())


@st.cache_resource
def load_text_classifier():
    if not MODEL_DIR.exists():
        return None, None

    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    model.eval()
    return tokenizer, model


def classify_text(text):
    tokenizer, model = load_text_classifier()

    if tokenizer is None or model is None:
        return None

    import torch

    encoded = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512,
    )

    with torch.no_grad():
        outputs = model(**encoded)
        scores = torch.softmax(outputs.logits, dim=1)[0]

    id2label = getattr(model.config, "id2label", {})
    rows = []
    for class_index, score in enumerate(scores.tolist()):
        raw_label = id2label.get(class_index, id2label.get(str(class_index), class_index))
        rows.append(
            {
                "Class": format_label(raw_label),
                "Probability": float(score),
            }
        )

    probabilities = pd.DataFrame(rows).sort_values("Probability", ascending=False)
    prediction = str(probabilities.iloc[0]["Class"])
    confidence = float(probabilities.iloc[0]["Probability"])
    return prediction, confidence, probabilities


def load_log():
    if LOG_FILE.exists():
        log_df = pd.read_csv(LOG_FILE)
        for col in LOG_COLUMNS:
            if col not in log_df.columns:
                log_df[col] = None
        return log_df[LOG_COLUMNS]
    return pd.DataFrame(columns=LOG_COLUMNS)


def save_log(input_text, prediction, confidence):
    log_df = load_log()
    new_row = pd.DataFrame(
        [
            {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "user_id": st.session_state.user_id,
                "age": st.session_state.age,
                "gender": st.session_state.gender,
                "input_text": input_text,
                "word_count": len(input_text.split()),
                "prediction": prediction,
                "confidence": confidence,
            }
        ]
    )
    pd.concat([log_df, new_row], ignore_index=True).to_csv(LOG_FILE, index=False)


def format_percent(value):
    if value is None or pd.isna(value):
        return "Pending"
    return f"{float(value) * 100:.1f}%"


def reset_screening():
    for key, value in DEFAULT_STATE.items():
        if key != "page_index":
            st.session_state[key] = value
    st.session_state.page_index = 0
    st.rerun()


# =========================
# UI components
# =========================
def hero_banner():
    render_html(
        """
        <div class="hero-card">
            <div class="mini-badge">Text Screening Support</div>
            <div class="hero-title">🧠 Mental Health Text Screening</div>
            <div class="hero-subtitle">
                Share a short written statement and the app will prepare it for classification
                into one of seven possible categories: anxiety, bipolar, stress, personality
                disorder, normal, depression, or suicidal.
            </div>
        </div>
        """
    )


def feature_cards():
    render_html(
        """
        <div class="card-grid">
            <div class="info-card">
                <div class="info-card-title">✍️ Write Your Text</div>
                <div class="info-card-text">
                    Type a sentence or paragraph in your own words. Avoid sharing private details
                    such as phone numbers, addresses, or passwords.
                </div>
            </div>
            <div class="info-card">
                <div class="info-card-title">🔎 Submit for Screening</div>
                <div class="info-card-text">
                    The app cleans the text and sends it to the classification function once
                    your trained model is connected.
                </div>
            </div>
            <div class="info-card">
                <div class="info-card-title">📊 View Result</div>
                <div class="info-card-text">
                    The result page shows the predicted category and confidence score in a
                    user-friendly format.
                </div>
            </div>
        </div>
        """
    )


def label_cards():
    cards = "".join(f'<div class="label-card">{label}</div>' for label in TARGET_LABELS)
    render_html(f'<div class="label-grid">{cards}</div>')


def step_flow():
    render_html(
        """
        <div class="journey-card">
            <div class="journey-title">Screening Journey</div>
            <div class="journey-subtitle">
                Follow these steps to complete one text screening session.
            </div>
            <div class="journey-row">
                <div class="journey-number">01</div>
                <div>
                    <div class="journey-step-title">👤 Enter basic details</div>
                    <div class="journey-step-desc">
                        Add a user ID or name so the screening record can be organized in the app.
                    </div>
                </div>
            </div>
            <div class="journey-row">
                <div class="journey-number">02</div>
                <div>
                    <div class="journey-step-title">💬 Write a text sample</div>
                    <div class="journey-step-desc">
                        Type the statement you want the classifier to screen. Keep it focused on
                        one situation or feeling for clearer output.
                    </div>
                </div>
            </div>
            <div class="journey-row">
                <div class="journey-number">03</div>
                <div>
                    <div class="journey-step-title">🔎 Run classification</div>
                    <div class="journey-step-desc">
                        Click the classify button to process the text and generate the predicted class.
                    </div>
                </div>
            </div>
            <div class="journey-row">
                <div class="journey-number">04</div>
                <div>
                    <div class="journey-step-title">🫶 Read the result responsibly</div>
                    <div class="journey-step-desc">
                        Use the prediction as a support signal only. It is not a diagnosis or
                        replacement for professional care.
                    </div>
                </div>
            </div>
        </div>
        """
    )


def pipeline_visual():
    render_html(
        """
        <div class="pipeline">
            <div class="pipeline-card">
                <div class="pipeline-title">1. User Text</div>
                <div class="pipeline-text">A written statement is entered into the app.</div>
            </div>
            <div class="pipeline-card">
                <div class="pipeline-title">2. Cleaning</div>
                <div class="pipeline-text">Noise such as URLs, symbols, and extra spaces is reduced.</div>
            </div>
            <div class="pipeline-card">
                <div class="pipeline-title">3. Features</div>
                <div class="pipeline-text">The trained pipeline will transform text into model-ready features.</div>
            </div>
            <div class="pipeline-card">
                <div class="pipeline-title">4. Class Output</div>
                <div class="pipeline-text">The classifier returns one of the seven target labels.</div>
            </div>
        </div>
        """
    )


def render_progress():
    current = st.session_state.page_index + 1
    total = len(PAGES)
    st.progress(current / total)
    st.caption(f"Step {current} of {total}: {PAGES[st.session_state.page_index]}")


def can_go_next():
    current_page = PAGES[st.session_state.page_index]

    if current_page == "Personal Details":
        return str(st.session_state.user_id).strip() != ""

    if current_page == "Text Screening":
        return st.session_state.prediction_ready

    return True


def render_navigation_buttons():
    st.divider()
    back_col, spacer_col, next_col = st.columns([1, 4, 1])

    with back_col:
        if st.session_state.page_index > 0:
            if st.button("Back", use_container_width=True):
                st.session_state.page_index -= 1
                st.rerun()

    with spacer_col:
        render_progress()

    with next_col:
        if st.session_state.page_index < len(PAGES) - 1:
            if st.button("Next", use_container_width=True, disabled=not can_go_next()):
                st.session_state.page_index += 1
                st.rerun()


def render_placeholder_result():
    render_html(
        """
        <div class="status-card status-alert">
            <div class="status-title">Classification is not available yet</div>
            <div class="status-text">
                The user interface is ready, but the trained model has not been connected yet.
                Once the model file is added, this page will show the predicted category,
                confidence score, and class probabilities.
            </div>
        </div>
        """
    )


with st.sidebar:
    st.title("Navigation")
    selected_page = st.radio(
        "Go to",
        PAGES,
        index=st.session_state.page_index,
        label_visibility="collapsed",
    )
    st.session_state.page_index = PAGES.index(selected_page)
    st.divider()
    st.caption(DISCLAIMER)


# =========================
# Page rendering
# =========================
current_page = PAGES[st.session_state.page_index]


if current_page == "Home":
    hero_banner()
    feature_cards()

    st.subheader("🌿 Possible Screening Categories")
    label_cards()

    step_flow()

    st.info(
        "This screening result should be read as a machine-learning support output, "
        "not as medical advice or a clinical diagnosis."
    )


elif current_page == "Personal Details":
    st.header("👤 Personal Details")
    st.write(
        "Enter basic information for your screening record. These details are kept "
        "separate from the text classification input."
    )

    col1, col2 = st.columns(2)
    with col1:
        st.session_state.user_id = st.text_input(
            "User ID or Name",
            value=st.session_state.user_id,
            placeholder="Example: Student01",
        )
        st.session_state.age = st.number_input(
            "Age",
            min_value=10,
            max_value=100,
            value=int(st.session_state.age),
        )

    with col2:
        st.session_state.gender = st.selectbox(
            "Gender",
            ["Prefer not to say", "Female", "Male", "Other"],
            index=["Prefer not to say", "Female", "Male", "Other"].index(
                st.session_state.gender
            ),
        )

    render_html(
        """
        <div class="soft-note">
            The classifier is designed around the text you submit. Avoid entering
            sensitive personal information in the text box unless it is necessary.
        </div>
        """
    )

    if st.button("Save Details", use_container_width=True):
        if str(st.session_state.user_id).strip() == "":
            st.error("Please enter a User ID or Name before continuing.")
        else:
            st.success("Details saved. Click Next to continue to text screening.")


elif current_page == "Text Screening":
    st.header("💬 Text Screening")
    st.write(
        "Enter the text you want to screen. A short paragraph usually works better "
        "than a single word."
    )

    st.session_state.input_text = st.text_area(
        "Text input",
        value=st.session_state.input_text,
        height=220,
        placeholder=(
            "Example: I feel overwhelmed lately and I cannot stop worrying "
            "about small things."
        ),
    )

    cleaned = clean_text(st.session_state.input_text)
    st.session_state.cleaned_text = cleaned

    col1, col2, col3 = st.columns(3)
    col1.metric("Characters", len(st.session_state.input_text))
    col2.metric("Words", len(st.session_state.input_text.split()))
    col3.metric("Target Classes", len(TARGET_LABELS))

    with st.expander("View cleaned text preview"):
        st.write(cleaned if cleaned else "No cleaned text yet.")

    if st.button("Classify Text", type="primary", use_container_width=True):
        if not st.session_state.input_text.strip():
            st.error("Please enter text before classification.")
        else:
            try:
                result = classify_text(cleaned)
            except Exception as exc:
                st.session_state.prediction_ready = False
                st.error("The model could not generate a prediction.")
                st.code(str(exc))
                result = None

            if result is None:
                st.session_state.prediction_ready = False
                render_placeholder_result()
                st.info(
                    "Please check that the model folder exists at "
                    "`best_model/best_model` and that all deployment dependencies are installed."
                )
            else:
                prediction, confidence, probabilities = result
                st.session_state.prediction = prediction
                st.session_state.confidence = confidence
                st.session_state.probabilities = probabilities
                st.session_state.prediction_ready = True
                save_log(st.session_state.input_text, prediction, confidence)
                st.success("Prediction is ready. Click Next to view the result.")


elif current_page == "Result":
    st.header("📊 Screening Result")

    if not st.session_state.prediction_ready:
        render_placeholder_result()
        st.subheader("🧾 Expected Result Format")
        st.dataframe(
            pd.DataFrame(
                {
                    "Output": ["Prediction", "Confidence", "Probabilities"],
                    "Example": [
                        "Anxiety",
                        "0.87",
                        "DataFrame with Class and Probability columns",
                    ],
                }
            ),
            hide_index=True,
            use_container_width=True,
        )
        label_cards()
    else:
        render_html(
            f"""
            <div class="status-card status-ready">
                <div class="status-title">Predicted Class: {st.session_state.prediction}</div>
                <div class="status-text">
                    The model selected this class from the seven target categories.
                    This result is a machine-learning output only, not a diagnosis.
                </div>
            </div>
            """
        )

        col1, col2 = st.columns(2)
        col1.metric("Model Confidence", format_percent(st.session_state.confidence))
        col2.metric("Word Count", len(st.session_state.input_text.split()))

        if st.session_state.probabilities is not None:
            st.subheader("Class Probability Distribution")
            st.dataframe(
                st.session_state.probabilities,
                hide_index=True,
                use_container_width=True,
            )
            st.bar_chart(
                st.session_state.probabilities.set_index("Class")[["Probability"]]
            )

        if st.button("Start New Screening", use_container_width=True):
            reset_screening()


elif current_page == "History":
    st.header("🕒 Screening History")

    log_df = load_log()

    if log_df.empty:
        st.info(
            "No completed screening records yet. Your recent results will appear here "
            "after classification is enabled and a screening is completed."
        )
    else:
        log_df["confidence"] = pd.to_numeric(log_df["confidence"], errors="coerce")

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Screenings", len(log_df))
        col2.metric("Average Confidence", format_percent(log_df["confidence"].mean()))
        col3.metric("Most Recent Class", str(log_df.tail(1)["prediction"].iloc[0]))

        st.subheader("📈 Prediction Distribution")
        distribution = (
            log_df["prediction"]
            .value_counts()
            .reindex(TARGET_LABELS, fill_value=0)
            .rename_axis("Class")
            .reset_index(name="Count")
        )
        st.bar_chart(distribution.set_index("Class"))

        st.subheader("🧾 Recent Screening Records")
        st.dataframe(log_df.tail(10), use_container_width=True)


elif current_page == "About":
    st.header("ℹ️ About This Screening Tool")
    pipeline_visual()

    st.write(
        """
        This app supports text-based mental-health category screening. A user enters
        a written statement, the app cleans the text, and the connected model returns
        one predicted category with confidence information.
        """
    )

    model_info = pd.DataFrame(
        {
            "Component": [
                "Input",
                "Preprocessing",
                "Feature Extraction",
                "Classifier",
                "Output Labels",
                "Status",
            ],
            "Detail": [
                "User-entered text",
                "Lowercase, URL removal, symbol cleanup, whitespace normalization",
                "BERT tokenizer from best_model/best_model",
                "BertForSequenceClassification loaded from model.safetensors",
                ", ".join(TARGET_LABELS),
                "Model connected" if MODEL_DIR.exists() else "Model folder not found",
            ],
        }
    )
    st.dataframe(model_info, use_container_width=True, hide_index=True)

    st.subheader("🛠️ Developer Note")
    st.write(
        "The app loads the tokenizer and sequence-classification model from "
        "`best_model/best_model` using `transformers` and `torch`."
    )
    st.code(
        """
prediction = "Anxiety"
confidence = 0.87
probabilities = pd.DataFrame({
    "Class": TARGET_LABELS,
    "Probability": [0.87, 0.02, 0.04, 0.01, 0.03, 0.02, 0.01],
})
return prediction, confidence, probabilities
        """.strip(),
        language="python",
    )

    st.info(DISCLAIMER)


elif current_page == "Finish":
    st.header("✅ Finish")

    render_html(
        """
        <div class="status-card status-ready">
            <div class="status-title">Screening session complete</div>
            <div class="status-text">
                Thank you for using the text screening tool. Please remember that this
                app provides a machine-learning category prediction only and cannot
                confirm or rule out any mental-health condition.
            </div>
        </div>
        """
    )

    render_html(
        """
        <div class="credit-card">
            <div class="credit-title">WQF7007 Group Assignment:</div>
            <div class="credit-text">
                Text-Based Mental Health Disorder Classification Using Natural Language Processing<br>
                prepared by Group 14
            </div>
        </div>
        """
    )


render_navigation_buttons()
