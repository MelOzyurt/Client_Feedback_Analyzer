from textblob import TextBlob
import pandas as pd

def analyze_sentiment(text):
    """
    Analyze sentiment of a single text string.
    Returns: polarity score (-1 to 1) and sentiment label.
    """
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity

    if polarity > 0.1:
        label = "Positive"
    elif polarity < -0.1:
        label = "Negative"
    else:
        label = "Neutral"

    return polarity, label


def batch_sentiment_analysis(sentences):
    """
    Analyze sentiment for a list of sentences.
    Returns DataFrame with polarity and label.
    """
    results = []
    for sentence in sentences:
        polarity, label = analyze_sentiment(sentence)
        results.append({
            "text": sentence,
            "polarity": round(polarity, 3),
            "sentiment": label
        })
    return pd.DataFrame(results)


def summarize_sentiments(df_result):
    """
    Summarizes sentiment analysis results.
    """
    sentiment_counts = df_result['sentiment'].value_counts().to_dict()
    avg_polarity = df_result['polarity'].mean()

    summary = {
        "total": len(df_result),
        "positive": sentiment_counts.get("Positive", 0),
        "neutral": sentiment_counts.get("Neutral", 0),
        "negative": sentiment_counts.get("Negative", 0),
        "average_polarity": round(avg_polarity, 3)
    }

    return summary
