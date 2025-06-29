import re
import string
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from textblob import TextBlob
import openai

# --- Global OpenAI client ---
openai_client = None

def set_openai_client(client):
    global openai_client
    openai_client = client


# --- Feedback Processor ---

def clean_text(text):
    text = text.lower()
    text = re.sub(r"\n+", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def split_into_sentences(text):
    sentences = re.split(r"[.!?]+", text)
    return [s.strip() for s in sentences if len(s.strip()) > 10]


def find_duplicates(sentences, threshold=0.9):
    tfidf = TfidfVectorizer().fit_transform(sentences)
    cosine_sim = cosine_similarity(tfidf)
    duplicates = []
    for i in range(len(sentences)):
        for j in range(i + 1, len(sentences)):
            if cosine_sim[i, j] > threshold:
                duplicates.append((sentences[i], sentences[j], cosine_sim[i, j]))
    return duplicates


def cluster_similar_sentences(sentences, threshold=0.6):
    tfidf = TfidfVectorizer().fit_transform(sentences)
    cosine_sim = cosine_similarity(tfidf)
    clusters = []
    used = set()
    for i in range(len(sentences)):
        if i in used:
            continue
        cluster = [sentences[i]]
        used.add(i)
        for j in range(i + 1, len(sentences)):
            if cosine_sim[i, j] > threshold and j not in used:
                cluster.append(sentences[j])
                used.add(j)
        if len(cluster) > 1:
            clusters.append(cluster)
    return clusters


# --- Sentiment Analysis ---

def analyze_sentiment(text):
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


# --- SWOT Analysis ---

def generate_swot_from_text(text):
    if not openai_client:
        raise ValueError("OpenAI client not set. Call set_openai_client(client) first.")

    prompt = (
        "You are a business analyst. Based on the following customer feedback text, "
        "generate a concise SWOT analysis. Focus on recurring themes and insights:\n\n"
        f"{text}\n\n"
        "Respond in this structured format:\n"
        "Strengths:\n- ...\nWeaknesses:\n- ...\nOpportunities:\n- ...\nThreats:\n- ..."
    )
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4-1106-preview",
            messages=[
                {"role": "system", "content": "You are an expert SWOT analyst."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5,
            max_tokens=600
        )
        output = response.choices[0].message.content
        return parse_swot_response(output)

    except Exception as e:
        return {"error": str(e)}


def parse_swot_response(text):
    sections = {"Strengths": [], "Weaknesses": [], "Opportunities": [], "Threats": []}
    current = None
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("Strengths"):
            current = "Strengths"
        elif line.startswith("Weaknesses"):
            current = "Weaknesses"
        elif line.startswith("Opportunities"):
            current = "Opportunities"
        elif line.startswith("Threats"):
            current = "Threats"
        elif line.startswith("-") and current:
            sections[current].append(line[1:].strip())
    return sections


# Optional: High-level summary pipeline (if needed)
def summarize_feedback(text):
    cleaned = clean_text(text)
    sentences = split_into_sentences(cleaned)
    duplicates = find_duplicates(sentences)
    clusters = cluster_similar_sentences(sentences)

    summary = {
        "total_sentences": len(sentences),
        "num_duplicates": len(duplicates),
        "num_clusters": len(clusters),
        "sample_clusters": clusters[:3],
    }
    return summary
