
# ============================================================
# SENTIMENT INSIGHT AI
# Complete Professional Streamlit Application
# SQLite Persistent Prediction History
# ============================================================

import sys
from pathlib import Path

# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ============================================================
# LIBRARIES
# ============================================================

import streamlit as st
import pandas as pd
import plotly.express as px

from sentiment_tool import predict_sentiment
from statistics_tool import analyze_text
from dataset_tool import analyze_dataset
from agent import chat_with_agent

from database_tool import (
    initialize_database,
    save_prediction,
    get_all_predictions,
    get_prediction_count,
    delete_all_predictions
)

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Sentiment Insight AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# DATABASE INITIALIZATION
# ============================================================

initialize_database()

# ============================================================
# SESSION STATE
# ============================================================

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

.stApp {
    background: #0b1020;
    color: #e5e7eb;
}

.block-container {
    max-width: 1450px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

section[data-testid="stSidebar"] {
    background: #10182c;
    border-right: 1px solid #26334d;
}

section[data-testid="stSidebar"] * {
    color: #e5e7eb;
}

h1, h2, h3, h4 {
    color: #f8fafc !important;
}

p, label, li {
    color: #cbd5e1;
}

.hero {
    background: linear-gradient(
        120deg,
        #312e81 0%,
        #4338ca 45%,
        #2563eb 100%
    );
    padding: 36px;
    border-radius: 20px;
    margin-bottom: 28px;
    box-shadow: 0 12px 35px rgba(37, 99, 235, 0.18);
}

.hero h1 {
    color: white !important;
    font-size: 36px;
    font-weight: 800;
    margin-bottom: 10px;
}

.hero p {
    color: #e0e7ff !important;
    font-size: 16px;
    margin-bottom: 0;
}

div[data-testid="stMetric"] {
    background: #151f35;
    border: 1px solid #293957;
    border-radius: 15px;
    padding: 20px;
}

div[data-testid="stMetricLabel"] {
    color: #94a3b8 !important;
}

div[data-testid="stMetricValue"] {
    color: #f8fafc !important;
}

.feature-card {
    background: #151f35;
    border: 1px solid #293957;
    padding: 23px;
    border-radius: 16px;
    min-height: 155px;
    margin-bottom: 12px;
}

.feature-card h3 {
    color: #c4b5fd !important;
    font-size: 20px;
}

.feature-card p {
    color: #cbd5e1 !important;
}

.stButton > button {
    border-radius: 10px;
    border: 1px solid #4f46e5;
    font-weight: 600;
}

.stTextArea textarea,
.stTextInput input {
    background: #111b30 !important;
    color: white !important;
    border: 1px solid #334155 !important;
    border-radius: 10px !important;
}

.footer {
    text-align: center;
    color: #64748b;
    padding: 25px;
    border-top: 1px solid #26334d;
    margin-top: 35px;
}

</style>
""", unsafe_allow_html=True)

# ============================================================
# PLOTLY THEME
# ============================================================

PLOT_LAYOUT = {
    "paper_bgcolor": "#151f35",
    "plot_bgcolor": "#151f35",
    "font": {"color": "#e5e7eb"},
    "margin": {"l": 30, "r": 30, "t": 55, "b": 30},
    "legend": {"font": {"color": "#cbd5e1"}}
}

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def show_hero(title, description):

    st.markdown(
        f"""
        <div class="hero">
            <h1>{title}</h1>
            <p>{description}</p>
        </div>
        """,
        unsafe_allow_html=True
    )


def show_card(title, description):

    st.markdown(
        f"""
        <div class="feature-card">
            <h3>{title}</h3>
            <p>{description}</p>
        </div>
        """,
        unsafe_allow_html=True
    )


def get_dataset_details():

    result = analyze_dataset()

    if not isinstance(result, dict):
        raise ValueError(
            "Dataset tool must return a dictionary."
        )

    total = result.get(
        "total_records",
        result.get("total", result.get("total_rows", 0))
    )

    dominant = result.get(
        "dominant_sentiment",
        result.get("dominant", "Not available")
    )

    counts = result.get(
        "sentiment_counts",
        result.get("sentiment_distribution", {})
    )

    percentages = result.get(
        "sentiment_percentages",
        result.get("percentages", {})
    )

    if not counts and isinstance(
        result.get("distribution"), dict
    ):
        counts = result["distribution"]

    if not isinstance(counts, dict):
        counts = {}

    if not isinstance(percentages, dict):
        percentages = {}

    if not percentages and counts and total:

        percentages = {
            key: (value / total) * 100
            for key, value in counts.items()
            if isinstance(value, (int, float))
        }

    return {
        "total": total,
        "dominant": dominant,
        "counts": counts,
        "percentages": percentages
    }


def get_training_loss_image():

    candidates = [
        PROJECT_ROOT / "reports" / "qwen_training_loss.png",
        PROJECT_ROOT / "qwen_training_loss.png",
        PROJECT_ROOT / "models" / "qwen_training_loss.png",
        PROJECT_ROOT.parent / "reports" / "qwen_training_loss.png"
    ]

    for path in candidates:

        if path.exists() and path.stat().st_size > 0:
            return path

    return None


def get_qwen_adapter_path():

    candidates = [
        PROJECT_ROOT / "models" / "qwen2.5-1.5b-sentiment-lora",
        PROJECT_ROOT / "models" / "qwen-lora",
        PROJECT_ROOT / "qwen2.5-1.5b-sentiment-lora"
    ]

    for path in candidates:

        if (path / "adapter_config.json").exists():
            return path

    return None


# ============================================================
# SIDEBAR NAVIGATION
# ============================================================

st.sidebar.markdown("# 🧠 Sentiment AI")

st.sidebar.caption(
    "Intelligent Text Analysis Platform"
)

st.sidebar.divider()

page = st.sidebar.radio(
    "NAVIGATION",
    [
        "Dashboard",
        "Sentiment Analyzer",
        "Text Analytics",
        "AI Assistant",
        "LLM Fine-Tuning",
        "Model Evaluation",
        "Prediction History"
    ]
)

st.sidebar.divider()

st.sidebar.success("AI Platform Connected")

st.sidebar.markdown("""
### MODEL INFORMATION

**Classification:** DistilBERT + LoRA

**Comparison:** BERT + LoRA

**Fine-tuned LLM:** Qwen 2.5 1.5B

**Agent:** LangChain

**Interface:** Streamlit

**Database:** SQLite
""")

st.sidebar.divider()

st.sidebar.caption(
    "Deep Learning | NLP | LLM | AI Agents"
)

# ============================================================
# PAGE 1: DASHBOARD
# ============================================================

if page == "Dashboard":

    show_hero(
        "Sentiment Insight AI",
        "An intelligent NLP platform for sentiment classification, "
        "LLM fine-tuning, text analytics and AI-powered insights."
    )

    st.subheader("Project Overview")

    st.write("""
    Sentiment Insight AI is an end-to-end NLP application
    designed to identify opinions and emotions expressed in text.

    The project combines transformer-based sentiment classification,
    LoRA fine-tuning, QLoRA-based LLM adaptation and a LangChain
    agent with multiple analysis tools.
    """)

    try:

        data = get_dataset_details()

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Dataset Records",
            f"{int(data['total']):,}"
            if data["total"] else "N/A"
        )

        col2.metric(
            "Dominant Sentiment",
            str(data["dominant"])
        )

        col3.metric(
            "Total Predictions",
            get_prediction_count()
        )

        st.divider()

        st.subheader("Dataset Sentiment Distribution")

        if data["counts"]:

            chart_data = pd.DataFrame([
                {
                    "Sentiment": str(key),
                    "Count": value
                }
                for key, value in data["counts"].items()
                if isinstance(value, (int, float))
            ])

            if not chart_data.empty:

                fig = px.pie(
                    chart_data,
                    names="Sentiment",
                    values="Count",
                    hole=0.55,
                    title="TweetEval Dataset Distribution",
                    color="Sentiment",
                    color_discrete_map={
                        "Negative": "#ef4444",
                        "Neutral": "#f59e0b",
                        "Positive": "#22c55e"
                    }
                )

                fig.update_layout(**PLOT_LAYOUT)

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

            else:
                st.info("Sentiment counts are unavailable.")

        else:
            st.info("Dataset distribution is unavailable.")

    except Exception as e:

        st.error(f"Dashboard loading error: {e}")

    st.divider()

    st.subheader("Project Capabilities")

    col1, col2, col3 = st.columns(3)

    with col1:
        show_card(
            "😊 Sentiment Analysis",
            "Classify text into Positive, Negative and Neutral sentiments."
        )

    with col2:
        show_card(
            "📊 Text Analytics",
            "Extract word counts, character counts and text statistics."
        )

    with col3:
        show_card(
            "🤖 AI Assistant",
            "Use a LangChain agent connected to three analysis tools."
        )

    col4, col5, col6 = st.columns(3)

    with col4:
        show_card(
            "🧠 LLM Fine-Tuning",
            "Explore BERT, DistilBERT LoRA and Qwen QLoRA training."
        )

    with col5:
        show_card(
            "📈 Model Evaluation",
            "Compare model performance using classification metrics."
        )

    with col6:
        show_card(
            "🗂️ Prediction History",
            "Review and export permanently saved predictions."
        )


# ============================================================
# PAGE 2: SENTIMENT ANALYZER
# ============================================================

elif page == "Sentiment Analyzer":

    show_hero(
        "Sentiment Analyzer",
        "Discover the emotional meaning of your text using a fine-tuned model."
    )

    st.subheader("Enter Your Text")

    user_text = st.text_area(
        "Text Input",
        placeholder="I love this project. It is amazing!",
        height=160
    )

    if st.button(
        "🔍 Analyze Sentiment",
        type="primary",
        use_container_width=True
    ):

        if not user_text.strip():

            st.warning("Please enter some text first.")

        else:

            try:

                with st.spinner("AI is analyzing your text..."):

                    result = predict_sentiment(user_text)

                sentiment = result["sentiment"]

                confidence = float(result["confidence"])

                probabilities = result["probabilities"]

                # Save prediction to permanent SQLite database
                save_prediction(
                    text=user_text,
                    sentiment=sentiment,
                    confidence=confidence
                )

                st.session_state.last_prediction = result

                st.success("Analysis completed successfully!")

                col1, col2 = st.columns(2)

                col1.metric(
                    "Predicted Sentiment",
                    sentiment
                )

                col2.metric(
                    "Confidence Score",
                    f"{confidence:.2f}%"
                )

                st.progress(
                    max(0.0, min(confidence / 100, 1.0))
                )

                st.divider()

                st.subheader("Prediction Probabilities")

                chart_data = pd.DataFrame({
                    "Sentiment": list(probabilities.keys()),
                    "Probability (%)": [
                        float(value)
                        for value in probabilities.values()
                    ]
                })

                fig = px.bar(
                    chart_data,
                    x="Sentiment",
                    y="Probability (%)",
                    color="Sentiment",
                    text_auto=".2f",
                    color_discrete_map={
                        "Negative": "#ef4444",
                        "Neutral": "#f59e0b",
                        "Positive": "#22c55e"
                    }
                )

                fig.update_layout(
                    **PLOT_LAYOUT,
                    yaxis_range=[0, 100]
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

                with st.expander("View Original Text"):
                    st.write(user_text)

                st.success(
                    "Prediction permanently saved in SQLite database."
                )

            except Exception as e:

                st.error(
                    f"Sentiment analysis failed: {e}"
                )


# ============================================================
# PAGE 3: TEXT ANALYTICS
# ============================================================

elif page == "Text Analytics":

    show_hero(
        "Text Analytics",
        "Explore the structure and statistics of your text."
    )

    user_text = st.text_area(
        "Enter Text",
        placeholder="Enter your paragraph here...",
        height=160
    )

    if st.button(
        "📊 Analyze Text",
        type="primary",
        use_container_width=True
    ):

        if not user_text.strip():

            st.warning("Please enter some text.")

        else:

            try:

                with st.spinner(
                    "Calculating text statistics..."
                ):

                    result = analyze_text(user_text)

                st.success("Text analysis completed!")

                st.subheader("Text Statistics")

                col1, col2, col3 = st.columns(3)

                col1.metric(
                    "Word Count",
                    result["word_count"]
                )

                col2.metric(
                    "Character Count",
                    result["character_count"]
                )

                col3.metric(
                    "Sentence Count",
                    result["sentence_count"]
                )

                col4, col5 = st.columns(2)

                col4.metric(
                    "Average Word Length",
                    result["average_word_length"]
                )

                col5.metric(
                    "Common Words Listed",
                    len(result["most_common_words"])
                )

                st.divider()

                st.subheader("Most Common Words")

                common_words = result["most_common_words"]

                if common_words:

                    words_df = pd.DataFrame(
                        common_words,
                        columns=["Word", "Frequency"]
                    )

                    fig = px.bar(
                        words_df,
                        x="Word",
                        y="Frequency",
                        text_auto=True,
                        title="Word Frequency"
                    )

                    fig.update_layout(**PLOT_LAYOUT)

                    st.plotly_chart(
                        fig,
                        use_container_width=True
                    )

                    st.dataframe(
                        words_df,
                        use_container_width=True,
                        hide_index=True
                    )

                else:

                    st.info("No common words found.")

            except Exception as e:

                st.error(
                    f"Text analysis failed: {e}"
                )


# ============================================================
# PAGE 4: AI ASSISTANT
# ============================================================

elif page == "AI Assistant":

    show_hero(
        "AI Assistant",
        "Interact with your LangChain-powered Sentiment Insight Agent."
    )

    st.caption(
        "Ask questions about sentiment, text statistics or dataset insights."
    )

    st.markdown("""
    **Available Agent Tools**

    - Sentiment Analysis Tool
    - Text Statistics Tool
    - Dataset Analysis Tool
    """)

    for message in st.session_state.chat_history:

        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    question = st.chat_input(
        "Ask your AI assistant..."
    )

    if question:

        st.session_state.chat_history.append({
            "role": "user",
            "content": question
        })

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):

            with st.spinner(
                "AI assistant is thinking..."
            ):

                try:

                    answer = chat_with_agent(question)

                    st.markdown(answer)

                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": str(answer)
                    })

                except Exception as e:

                    st.error(
                        f"Agent error: {e}. "
                        "Please check whether Ollama is running."
                    )

    if st.button("🗑️ Clear Chat"):

        st.session_state.chat_history = []

        st.rerun()


# ============================================================
# PAGE 5: LLM FINE-TUNING
# ============================================================

elif page == "LLM Fine-Tuning":

    show_hero(
        "LLM Fine-Tuning Studio",
        "Explore transformer models, parameter-efficient fine-tuning "
        "and QLoRA-based LLM adaptation."
    )

    st.subheader("Fine-Tuning Project Overview")

    st.write("""
    Fine-tuning adapts a pretrained model to a specific task.

    This project explores LoRA for transformer-based sentiment
    classifiers and QLoRA for adapting the Qwen 2.5 1.5B Instruct model.
    """)

    col1, col2, col3 = st.columns(3)

    col1.metric("Fine-Tuned Model Families", "3")

    col2.metric("Qwen Training Steps", "750")

    col3.metric("Qwen Training Samples", "6,000")

    st.divider()

    st.subheader("Models Used in This Project")

    model_tab1, model_tab2, model_tab3 = st.tabs([
        "DistilBERT + LoRA",
        "BERT + LoRA",
        "Qwen 2.5 + QLoRA"
    ])

    with model_tab1:

        st.markdown("### DistilBERT + LoRA")

        st.write("""
        DistilBERT is a smaller transformer model derived from BERT.

        LoRA was used to adapt selected model parameters for
        three-class sentiment classification.
        """)

        col1, col2 = st.columns(2)

        col1.metric("Test Accuracy", "65.39%")

        col2.metric("Macro F1-Score", "65.37%")

        st.markdown("""
        **Training approach**

        - Pretrained model: distilbert-base-uncased
        - Fine-tuning technique: LoRA
        - Task: Three-class sentiment classification
        - Classes: Negative, Neutral and Positive
        """)

    with model_tab2:

        st.markdown("### BERT + LoRA")

        st.write("""
        BERT is a bidirectional transformer encoder.

        It was fine-tuned using LoRA and compared with DistilBERT
        on the sentiment classification task.
        """)

        col1, col2 = st.columns(2)

        col1.metric("Test Accuracy", "65.82%")

        col2.metric("Macro F1-Score", "66.02%")

        st.markdown("""
        **Training approach**

        - Architecture: BERT
        - Fine-tuning technique: LoRA
        - Task: Three-class sentiment classification
        - Comparison: DistilBERT + LoRA
        """)

    with model_tab3:

        st.markdown("### Qwen 2.5 1.5B Instruct + QLoRA")

        st.write("""
        Qwen 2.5 1.5B Instruct was adapted for sentiment-related
        instruction tasks using 4-bit QLoRA fine-tuning.

        Training was performed in Google Colab using a Tesla T4 GPU.
        """)

        col1, col2, col3 = st.columns(3)

        col1.metric("Training Samples", "6,000")

        col2.metric("Optimizer Steps", "750")

        col3.metric("LoRA Rank", "8")

        st.markdown("#### QLoRA Training Configuration")

        qwen_config = pd.DataFrame({
            "Parameter": [
                "Base Model",
                "Fine-Tuning Method",
                "Quantization",
                "Quantization Type",
                "Compute Type",
                "GPU",
                "LoRA Rank",
                "LoRA Alpha",
                "Target Modules",
                "LoRA Dropout",
                "Learning Rate",
                "Maximum Sequence Length",
                "Gradient Accumulation",
                "Training Samples",
                "Optimizer Steps"
            ],
            "Configuration": [
                "Qwen/Qwen2.5-1.5B-Instruct",
                "QLoRA",
                "4-bit",
                "NF4",
                "FP16",
                "Tesla T4",
                "8",
                "16",
                "q_proj, k_proj, v_proj, o_proj",
                "0.05",
                "2e-4",
                "192",
                "8",
                "6,000",
                "750"
            ]
        })

        st.dataframe(
            qwen_config,
            use_container_width=True,
            hide_index=True
        )

        st.info("""
        Qwen fine-tuning was completed and its LoRA adapter
        and checkpoints were saved.

        Final Qwen test evaluation is pending.
        """)

    st.divider()

    st.subheader("Qwen Training Loss Visualization")

    loss_image = get_training_loss_image()

    if loss_image:

        st.image(
            str(loss_image),
            caption=(
                "Qwen training loss during recorded continuation "
                "steps 501–750"
            ),
            use_container_width=True
        )

        st.caption(
            "This graph represents recorded continuation history "
            "for steps 501–750, not the complete 750-step history."
        )

    else:

        st.warning(
            "Training loss image was not found in the reports folder."
        )

    st.divider()

    st.subheader("Saved Qwen Adapter")

    adapter_path = get_qwen_adapter_path()

    if adapter_path:

        st.success(
            "Qwen LoRA adapter detected in the project folder."
        )

        st.code(str(adapter_path))

        adapter_files = [
            file.name
            for file in adapter_path.iterdir()
            if file.is_file()
        ]

        st.write("Available adapter files:")

        st.write(", ".join(adapter_files))

    else:

        st.warning(
            "Qwen adapter was not detected in the local project folder."
        )

    st.divider()

    st.subheader("Fine-Tuning Workflow")

    workflow = [
        "1. Load pretrained transformer or LLM",
        "2. Prepare and balance sentiment training data",
        "3. Configure LoRA or QLoRA parameters",
        "4. Fine-tune the model on training examples",
        "5. Save adapter weights and checkpoints",
        "6. Evaluate the model on validation and test data"
    ]

    for step in workflow:
        st.markdown(f"- {step}")


# ============================================================
# PAGE 6: MODEL EVALUATION
# ============================================================

elif page == "Model Evaluation":

    show_hero(
        "Model Evaluation",
        "Compare transformer model performance using classification metrics."
    )

    st.subheader("Model Performance Comparison")

    st.write("""
    BERT and DistilBERT were fine-tuned using LoRA
    for three-class sentiment classification.
    """)

    model_data = pd.DataFrame({
        "Model": [
            "DistilBERT + LoRA",
            "BERT + LoRA"
        ],
        "Accuracy": [65.39, 65.82],
        "Precision": [67.92, 68.51],
        "Recall": [65.39, 65.82],
        "Macro F1": [65.37, 66.02],
        "Weighted F1": [64.68, 65.07]
    })

    col1, col2 = st.columns(2)

    col1.metric(
        "DistilBERT Accuracy",
        "65.39%"
    )

    col2.metric(
        "BERT Accuracy",
        "65.82%",
        delta="+0.43 percentage points"
    )

    st.divider()

    st.subheader("Detailed Evaluation Metrics")

    st.dataframe(
        model_data.set_index("Model").style.format("{:.2f}%"),
        use_container_width=True
    )

    st.subheader("Accuracy Comparison")

    accuracy_chart = px.bar(
        model_data,
        x="Model",
        y="Accuracy",
        color="Model",
        text="Accuracy",
        title="Accuracy of Both Models"
    )

    accuracy_chart.update_traces(
        texttemplate="%{text:.2f}%",
        textposition="outside"
    )

    accuracy_chart.update_layout(
        **PLOT_LAYOUT,
        yaxis_title="Accuracy (%)",
        yaxis_range=[0, 80],
        showlegend=False
    )

    st.plotly_chart(
        accuracy_chart,
        use_container_width=True
    )

    st.subheader("Performance Metrics Comparison")

    metric_data = model_data.melt(
        id_vars="Model",
        var_name="Metric",
        value_name="Score"
    )

    metric_chart = px.bar(
        metric_data,
        x="Metric",
        y="Score",
        color="Model",
        barmode="group",
        text_auto=".2f",
        title="Classification Metrics"
    )

    metric_chart.update_layout(
        **PLOT_LAYOUT,
        yaxis_title="Score (%)",
        yaxis_range=[0, 85]
    )

    st.plotly_chart(
        metric_chart,
        use_container_width=True
    )

    st.divider()

    st.subheader("Confusion Matrix Comparison")

    labels = [
        "Negative",
        "Neutral",
        "Positive"
    ]

    distil_matrix = [
        [3383, 413, 176],
        [2068, 3005, 864],
        [190, 541, 1644]
    ]

    bert_matrix = [
        [3396, 420, 156],
        [2098, 2992, 847],
        [182, 494, 1699]
    ]

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("**DistilBERT + LoRA**")

        fig_distil = px.imshow(
            distil_matrix,
            x=labels,
            y=labels,
            text_auto=True,
            color_continuous_scale="Blues",
            labels={
                "x": "Predicted Label",
                "y": "Actual Label",
                "color": "Samples"
            }
        )

        fig_distil.update_layout(**PLOT_LAYOUT)

        st.plotly_chart(
            fig_distil,
            use_container_width=True
        )

    with col2:

        st.markdown("**BERT + LoRA**")

        fig_bert = px.imshow(
            bert_matrix,
            x=labels,
            y=labels,
            text_auto=True,
            color_continuous_scale="Greens",
            labels={
                "x": "Predicted Label",
                "y": "Actual Label",
                "color": "Samples"
            }
        )

        fig_bert.update_layout(**PLOT_LAYOUT)

        st.plotly_chart(
            fig_bert,
            use_container_width=True
        )

    st.divider()

    st.subheader("Performance Analysis")

    st.markdown("""
    **Observations**

    1. BERT achieved 65.82% accuracy, while DistilBERT achieved 65.39%.

    2. BERT obtained a macro F1-score of 66.02%, compared
       with 65.37% for DistilBERT.

    3. BERT performed slightly better on the recorded overall metrics.

    4. DistilBERT has a smaller and lighter architecture,
       which can be useful when computational resources are limited.

    5. Both models show differences in classification performance
       across Negative, Neutral and Positive classes.
    """)

    st.info("""
    Evaluation note: These are previously recorded test-set results.
    Since the test set was used during earlier model comparisons,
    treat this as an exploratory comparison. For an unbiased final
    evaluation, select the model using validation results and
    evaluate once on a fresh untouched test set.
    """)

    st.divider()

    st.subheader("Qwen 2.5 QLoRA Evaluation Status")

    st.warning(
        "Qwen fine-tuning is complete, but final test evaluation "
        "has not yet been completed. Accuracy, precision, recall "
        "and F1-score are therefore not reported here."
    )


# ============================================================
# PAGE 7: PREDICTION HISTORY - SQLITE
# ============================================================

elif page == "Prediction History":

    show_hero(
        "Prediction History",
        "View, analyze and download your permanently saved predictions."
    )

    history_df = get_all_predictions()

    st.subheader("Prediction Summary")

    total = len(history_df)

    if total:

        positive = int(
            (history_df["Sentiment"] == "Positive").sum()
        )

        negative = int(
            (history_df["Sentiment"] == "Negative").sum()
        )

        neutral = int(
            (history_df["Sentiment"] == "Neutral").sum()
        )

    else:

        positive = 0
        negative = 0
        neutral = 0

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Predictions", total)
    col2.metric("Positive", positive)
    col3.metric("Negative", negative)
    col4.metric("Neutral", neutral)

    if total:

        st.divider()

        st.subheader("Saved Prediction Records")

        search_text = st.text_input(
            "Search prediction text",
            placeholder="Search your previous predictions..."
        )

        filtered_df = history_df.copy()

        if search_text.strip():

            filtered_df = filtered_df[
                filtered_df["Input Text"].str.contains(
                    search_text,
                    case=False,
                    na=False
                )
            ]

        st.dataframe(
            filtered_df,
            use_container_width=True,
            hide_index=True
        )

        st.caption(
            f"Showing {len(filtered_df)} of {total} saved records."
        )

        st.divider()

        st.subheader("Sentiment Distribution")

        sentiment_counts = (
            history_df["Sentiment"]
            .value_counts()
            .reset_index()
        )

        sentiment_counts.columns = [
            "Sentiment",
            "Count"
        ]

        fig = px.pie(
            sentiment_counts,
            names="Sentiment",
            values="Count",
            hole=0.5,
            color="Sentiment",
            color_discrete_map={
                "Positive": "#22c55e",
                "Negative": "#ef4444",
                "Neutral": "#f59e0b"
            }
        )

        fig.update_layout(**PLOT_LAYOUT)

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        csv_data = history_df.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            label="📥 Download Prediction History CSV",
            data=csv_data,
            file_name="sentiment_prediction_history.csv",
            mime="text/csv",
            use_container_width=True
        )

        st.divider()

        if st.button(
            "🗑️ Clear All Prediction Records",
            type="secondary"
        ):

            delete_all_predictions()

            st.success(
                "All prediction records deleted successfully."
            )

            st.rerun()

    else:

        st.info(
            "No predictions saved yet. "
            "Visit Sentiment Analyzer to create your first prediction."
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.markdown("""
<div class="footer">
    <b>🧠 Sentiment Insight AI</b><br>
    Deep Learning | NLP | LoRA | QLoRA | LangChain | Streamlit | SQLite
    <br><br>
    Intelligent Text Analysis Platform
</div>
""", unsafe_allow_html=True)
