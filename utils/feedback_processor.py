import re
import string
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def clean_text(text):
    """
    Basic text cleaning: lower, remove punctuation, special chars.
    """
    text = text.lower()
    text = re.sub(r"\n+", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def split_into_sentences(text):
    """
    Split large text into sentences using punctuation.
    """
    sentences = re.split(r"[.!?]+", text)
    return [s.strip() for s in sentences if len(s.strip()) > 10]


def find_duplicates(sentences, threshold=0.9):
    """
    Use TF-IDF and cosine similarity to find repeated/near-identical sentences.
    Returns a list of (i, j, score) for similar pairs.
    """
    tfidf = TfidfVectorizer().fit_transform(sentences)
    cosine_sim = cosine_similarity(tfidf)

    duplicates = []
    for i in range(len(sentences)):
        for j in range(i + 1, len(sentences)):
            if cosine_sim[i, j] > threshold:
                duplicates.append((sentences[i], sentences[j], cosine_sim[i, j]))
    return duplicates


def cluster_similar_sentences(sentences, threshold=0.6):
    """
    Cluster similar sentences together using cosine similarity.
    Very rough topic grouping.
    """
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


def summarize_feedback(text):
    """
    High-level pipeline to clean, split, and detect patterns from raw feedback.
    Returns summary dictionary.
    """
    cleaned = clean_text(text)
    sentences = split_into_sentences(cleaned)
    duplicates = find_duplicates(sentences)
    clusters = cluster_similar_sentences(sentences)

    summary = {
        "total_sentences": len(sentences),
        "num_duplicates": len(duplicates),
        "num_clusters": len(clusters),
        "sample_clusters": clusters[:3],  # only show top 3
    }
    return summary
