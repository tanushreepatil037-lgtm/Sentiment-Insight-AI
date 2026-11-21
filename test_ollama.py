from langchain_ollama import ChatOllama

llm = ChatOllama(
    model="qwen2.5:1.5b",
    temperature=0
)

response = llm.invoke(
    "What is sentiment analysis? Explain in two sentences."
)

print("\nOllama Response:")
print(response.content)
