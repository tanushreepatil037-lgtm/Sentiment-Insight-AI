
import re
from collections import Counter


def analyze_text(text):

    if not isinstance(text, str) or not text.strip():
        raise ValueError("Please enter valid text.")

    # Word count
    words = re.findall(r"\b[\w']+\b", text)

    # Character count
    characters = len(text)

    # Sentence count
    sentences = [
        s for s in re.split(r"[.!?]+", text)
        if s.strip()
    ]

    # Average word length
    average_word_length = (
        sum(len(word) for word in words) / len(words)
        if words else 0
    )

    # Most common words
    common_words = Counter(
        word.lower() for word in words
        if len(word) > 2
    ).most_common(5)

    return {
        "word_count": len(words),
        "character_count": characters,
        "sentence_count": len(sentences),
        "average_word_length": round(average_word_length, 2),
        "most_common_words": common_words
    }


if __name__ == "__main__":

    sample = "I love this AI project! It is amazing and very useful."

    result = analyze_text(sample)

    print("\nTEXT STATISTICS")
    print("--------------------------")

    for key, value in result.items():
        print(f"{key}: {value}")
