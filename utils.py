import re
from textblob import TextBlob

def preprocess_reviews(text):
    # Basic text cleaning
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r"[^a-zA-Z0-9\s]", '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def get_sentiment_summary(text, client):
    # Quick sentiment analysis with TextBlob
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity

    if polarity > 0.2:
        sentiment = "Positive"
    elif polarity < -0.2:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"

    return {
        "Sentiment": sentiment,
        "Polarity": round(polarity, 3),
        "Subjectivity": round(subjectivity, 3),
        "Length": len(text.split())
    }

def generate_swot_analysis(text, client):
    # SWOT analysis using GPT
    prompt = f"""
    Based on the customer feedback below, create a SWOT analysis.

    Feedback:
    \"\"\"{text}\"\"\"

    Respond in the following format:
    Strengths:
    -
    Weaknesses:
    -
    Opportunities:
    -
    Threats:
    -
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an expert market analyst."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )
        result = response.choices[0].message.content
        sections = result.split("\n\n")
        swot = {}
        for section in sections:
            if ":" in section:
                title, content = section.split(":", 1)
                swot[title.strip()] = content.strip()
        return swot
    except Exception as e:
        return {"Error": f"SWOT generation failed: {str(e)}"}
