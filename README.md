# Sentiment Insight AI

## 🚀 Live Demo

[Click here to open Sentiment Insight AI](https://sentiment-intelligence-ai-gdkn8irnmhwemuwjkodhje.streamlit.app/)

---
An AI-powered sentiment analysis system that classifies text into Positive, Negative, and Neutral categories using fine-tuned transformer models, LoRA, QLoRA, and an intelligent AI assistant.

## 1. Project Overview

Sentiment Insight AI is designed to analyze textual data and identify opinions and emotions expressed in user input. It combines Deep Learning, Natural Language Processing (NLP), Large Language Models (LLMs), and LangChain-based AI agents in an interactive Streamlit application.

## 2. Objectives

- Develop a sentiment analysis system using transformer architectures.
- Fine-tune pretrained language models for sentiment classification.
- Compare BERT and DistilBERT model performance.
- Fine-tune Qwen 2.5 1.5B using QLoRA.
- Build an intelligent AI assistant using LangChain.
- Evaluate model performance using classification metrics.

## 3. Key Features

- Sentiment Analyzer – classifies text into Positive, Negative, and Neutral.
- Text Analytics – analyzes word count, character count, sentence count, and frequent words.
- AI Assistant – uses a LangChain agent with three tools.
- Dataset Analysis – displays sentiment distribution and dataset statistics.
- Model Evaluation – displays accuracy, precision, recall, F1-score, and confusion matrices.
- Fine-Tuning Information – displays Qwen training configuration and training-loss graph.
- Prediction History – stores predictions using SQLite.
- Interactive Dashboard – provides a visual interface using Streamlit.

## 4. Technologies Used

| Category | Technologies |
|---|---|
| Language | Python |
| Interface | Streamlit |
| Deep Learning | PyTorch |
| NLP | Hugging Face Transformers |
| Fine-Tuning | PEFT, LoRA, QLoRA |
| AI Agent | LangChain |
| Local LLM | Ollama, Qwen 2.5 1.5B |
| Data Processing | Pandas, NumPy |
| Visualization | Plotly, Matplotlib |
| Database | SQLite |
| Version Control | Git, GitHub |

## 5. Dataset

The project uses the TweetEval sentiment dataset.

The three sentiment classes are:
- Negative
- Neutral
- Positive

Training dataset distribution:

| Sentiment | Records | Percentage |
|---|---:|---:|
| Negative | 7,093 | 15.55% |
| Neutral | 20,673 | 45.32% |
| Positive | 17,849 | 39.13% |
| Total | 45,615 | 100% |

The dataset contains training, validation, and testing files.

## 6. Model Development and Fine-Tuning

### DistilBERT with LoRA

- Base model: distilbert-base-uncased
- Test accuracy: 65.39%
- Weighted F1-score: 64.68%

### BERT with LoRA

- Fine-tuning approach: LoRA
- Test accuracy: 65.82%
- Weighted F1-score: 65.07%

### Qwen 2.5 1.5B with QLoRA

- Base model: Qwen/Qwen2.5-1.5B-Instruct
- Fine-tuning method: QLoRA, 4-bit NF4
- Training platform: Google Colab with NVIDIA T4 GPU
- Training examples: 6,000 balanced examples
- LoRA rank: 8
- LoRA alpha: 16
- Maximum sequence length: 192
- Optimizer steps: 750
- Final recorded training loss: 0.1027

Note: Qwen fine-tuning is completed and the adapter is saved. Independent test-set evaluation is pending.

## 7. Model Evaluation

| Model | Test Accuracy | Weighted F1 |
|---|---:|---:|
| DistilBERT + LoRA | 65.39% | 64.68% |
| BERT + LoRA | 65.82% | 65.07% |

Evaluation metrics include:
- Accuracy
- Precision
- Recall
- Weighted F1-score
- Macro F1-score
- Confusion Matrix

The reported comparison is exploratory. Final model selection should use validation data, followed by evaluation on an untouched test set.

## 8. AI Agent Tools

The LangChain AI assistant includes three tools:

1. Sentiment Analysis Tool
2. Text Statistics Tool
3. Dataset Analysis Tool

The agent uses a local Ollama model to process user requests and select appropriate tools.

## 9. Project Structure

```text
Sentiment-Intelligence-AI/
│
├── app.py/
│   └── streamlit_app.py
├── dataset/
│   ├── train.csv
│   ├── validation.csv
│   └── test.csv
├── models/
│   ├── sentiment-lora/
│   └── qwen2.5-1.5b-sentiment-lora/
├── reports/
│   ├── figures/
│   └── qwen_training_loss.png
├── agent.py
├── sentiment_tool.py
├── statistics_tool.py
├── dataset_tool.py
├── database_tool.py
├── predict.py
├── train.py
├── train_model.py
├── requirements.txt
├── README.md
└── .gitignore
```

## 10. Installation and Setup

### Clone the repository

```bash
git clone https://github.com/palak-surana/Sentiment-Intelligence-AI.git
cd Sentiment-Intelligence-AI
```

### Create a virtual environment

```bash
python -m venv .venv
```

Activate on Windows:

```powershell
.venv\Scripts\activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Install and start Ollama

Install Ollama and download the model:

```bash
ollama pull qwen2.5:1.5b
```

### Run the Streamlit application

```bash
streamlit run app.py/streamlit_app.py
```

The application will open in your browser.

## 11. Reports and Visualizations

The reports directory contains:
- Classification reports
- Confusion matrices
- Model evaluation results
- Sentiment distribution charts
- Text length distribution charts
- Qwen training-loss visualization

## 12. Future Scope

- Complete independent Qwen test-set evaluation.
- Integrate the fine-tuned Qwen adapter into the inference pipeline.
- Improve sentiment classification performance.
- Add multilingual sentiment analysis.
- Explore real-time social media sentiment monitoring.
- Deploy the application on a cloud hosting platform.
- Improve fairness, robustness, and explainability.

## 13. Project Repository

GitHub: https://github.com/palak-surana/Sentiment-Intelligence-AI

## Author

**Palak Surana**  
B.Tech – Data Science  
MIT Academy of Engineering (MITAOE), Pune

---

*Sentiment Insight AI – Combining Deep Learning, NLP, and Intelligent Agents.*