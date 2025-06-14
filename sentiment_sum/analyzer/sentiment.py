from transformers import pipeline
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Load once
distilbert_pipeline = pipeline("sentiment-analysis")
vader = SentimentIntensityAnalyzer()

def analyze_sentiment(comments):
    vader_scores = {"pos": 0, "neu": 0, "neg": 0}
    distilbert_scores = {"POSITIVE": 0, "NEGATIVE": 0}
    
    for comment in comments:
        if not comment.strip():
            continue

        vs = vader.polarity_scores(comment)
        if vs["compound"] >= 0.05:
            vader_scores["pos"] += 1
        elif vs["compound"] <= -0.05:
            vader_scores["neg"] += 1
        else:
            vader_scores["neu"] += 1

    total = sum(vader_scores.values())
    if total == 0: total = 1  # avoid division by zero

    return {
        "positive": round((vader_scores["pos"] / total) * 100, 2),
        "neutral": round((vader_scores["neu"] / total) * 100, 2),
        "negative": round((vader_scores["neg"] / total) * 100, 2),
    }