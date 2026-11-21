
import json

from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_ollama import ChatOllama

from sentiment_tool import predict_sentiment
from statistics_tool import analyze_text
from dataset_tool import analyze_dataset


# TOOL 1: Sentiment Prediction
@tool
def sentiment_analysis(text: str) -> str:
    """Predict whether text has Positive, Negative, or Neutral sentiment."""
    result = predict_sentiment(text)
    return json.dumps(result, ensure_ascii=False, default=str)


# TOOL 2: Text Statistics
@tool
def text_statistics(text: str) -> str:
    """Calculate word count, character count, sentence count,
    average word length, and common words in the given text."""
    result = analyze_text(text)
    return json.dumps(result, ensure_ascii=False, default=str)


# TOOL 3: Dataset Analysis
@tool
def dataset_analysis() -> str:
    """Analyze the training dataset and return sentiment distribution,
    percentages, total records, and dominant sentiment."""
    result = analyze_dataset()
    return json.dumps(result, ensure_ascii=False, default=str)


# Initialize Ollama LLM
llm = ChatOllama(
    model="qwen2.5:1.5b",
    temperature=0
)


# Register all tools with the agent
tools = [
    sentiment_analysis,
    text_statistics,
    dataset_analysis
]


# Create LangChain Agent
agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt="""
    You are Sentiment Insight AI, an intelligent sentiment analysis assistant.

    You have three tools:
    1. sentiment_analysis: Predicts sentiment from text.
    2. text_statistics: Calculates text statistics.
    3. dataset_analysis: Analyzes the sentiment dataset.

    Instructions:
    - Select the appropriate tool based on the user's question.
    - Use the tools when their results are required.
    - Never invent tool results.
    - Explain the results in simple and clear language.
    - If the user asks about sentiment, classify it as Positive,
      Negative, or Neutral.
    - If the user asks about dataset distribution, use dataset_analysis.
    """
)


# Chat with the agent
def chat_with_agent(question):
    response = agent.invoke({
        "messages": [
            {"role": "user", "content": question}
        ]
    })

    for message in reversed(response["messages"]):
        if getattr(message, "type", "") == "ai" and message.content:
            return message.content

    return "I could not generate a response."


if __name__ == "__main__":
    print("\n======================================")
    print("       SENTIMENT INSIGHT AI AGENT")
    print("======================================")
    print("Type 'exit' to stop.\n")

    while True:
        question = input("You: ").strip()

        if question.lower() in ["exit", "quit"]:
            print("Agent: Goodbye!")
            break

        if not question:
            print("Please enter a question.")
            continue

        try:
            answer = chat_with_agent(question)
            print("\nAgent:", answer, "\n")

        except Exception as error:
            print("\nError:", error)
            print("Please check that Ollama is running.\n")
