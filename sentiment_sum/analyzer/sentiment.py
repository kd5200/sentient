from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
from collections import defaultdict
from .theme_analysis import generate_theme_analysis

# Load models once
distilbert_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
vader = SentimentIntensityAnalyzer()

def preprocess_comments(comments):
    """Preprocess comments by removing duplicates, spam, and grouping similar ones."""
    if not comments:
        return []
    
    # Remove empty comments and strip whitespace
    comments = [c.strip() for c in comments if c.strip()]
    
    # Remove duplicates
    comments = list(set(comments))
    
    # Basic spam detection (very short comments, repeated characters, etc.)
    spam_patterns = [
        r'^.{0,3}$',  # Very short comments
        r'(.)\1{4,}',  # Repeated characters
        r'[A-Z]{10,}',  # All caps
        r'[!]{3,}',    # Multiple exclamation marks
    ]
    
    filtered_comments = []
    for comment in comments:
        if not any(re.search(pattern, comment) for pattern in spam_patterns):
            filtered_comments.append(comment)
    
    # Group similar comments using TF-IDF and cosine similarity
    if len(filtered_comments) > 1:
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(filtered_comments)
        similarity_matrix = cosine_similarity(tfidf_matrix)
        
        # Group similar comments (similarity threshold: 0.7)
        groups = defaultdict(list)
        used_indices = set()
        
        for i in range(len(filtered_comments)):
            if i in used_indices:
                continue
                
            current_group = [i]
            used_indices.add(i)
            
            for j in range(i + 1, len(filtered_comments)):
                if j not in used_indices and similarity_matrix[i, j] > 0.7:
                    current_group.append(j)
                    used_indices.add(j)
            
            # Use the longest comment as representative
            group_comments = [filtered_comments[idx] for idx in current_group]
            representative = max(group_comments, key=len)
            groups[representative].extend(group_comments)
        
        # Return unique comments, with similar ones grouped
        return list(groups.keys())
    
    return filtered_comments

def chunk_comments(comments, chunk_size=30):
    """Split comments into chunks for efficient processing."""
    chunks = []
    current_chunk = []
    current_length = 0
    
    for comment in comments:
        comment_length = len(comment.split())
        if current_length + comment_length > chunk_size and current_chunk:
            chunks.append(" ".join(current_chunk))
            current_chunk = [comment]
            current_length = comment_length
        else:
            current_chunk.append(comment)
            current_length += comment_length
    
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks

def generate_summary(text, max_length=130, min_length=30):
    """Generate a summary for a given text."""
    try:
        # Calculate appropriate max_length based on input length
        input_length = len(text.split())
        if input_length < 50:
            # For short texts, use shorter summary
            max_length = min(input_length, 20)
            min_length = min(input_length // 2, 10)
        elif input_length < 200:
            # For medium texts
            max_length = min(input_length // 2, 100)
            min_length = min(input_length // 4, 30)
        else:
            # For long texts
            max_length = min(input_length // 3, 150)
            min_length = min(input_length // 6, 50)

        summary_result = summarizer(
            text,
            max_length=max_length,
            min_length=min_length,
            do_sample=False,
            truncation=True
        )
        return summary_result[0]['summary_text']
    except Exception as e:
        print(f"Summarization error: {e}")
        return "Summary generation failed"

def aggregate_summaries(summaries):
    """Aggregate multiple summaries into final bullet points."""
    if not summaries:
        return []
    
    # Combine all summaries
    combined_text = " ".join(summaries)
    
    # Calculate appropriate lengths based on combined text
    input_length = len(combined_text.split())
    max_length = min(input_length // 2, 150)
    min_length = min(input_length // 4, 50)
    
    # Generate final summary
    try:
        final_summary = summarizer(
            combined_text,
            max_length=max_length,
            min_length=min_length,
            do_sample=False,
            truncation=True
        )[0]['summary_text']
        
        # Split into bullet points and clean up
        bullet_points = [
            point.strip() 
            for point in final_summary.split('.') 
            if point.strip() and len(point.strip()) > 10  # Filter out very short points
        ]
        return bullet_points[:5]  # Return top 5 bullet points
    except Exception as e:
        print(f"Final summarization error: {e}")
        return summaries

def analyze_sentiment(comments):
    """Main sentiment analysis function with preprocessing and summarization."""
    if not comments:
        return {
            "positive": 0,
            "neutral": 0,
            "negative": 0,
            "summary": [],
            "detailed_sentiment": [],
            "avg_vader_score": 0,
            "avg_distilbert_score": 0,
            "theme_analysis": {
                "themes": [],
                "explanation": "No comments to analyze."
            }
        }

    # Preprocess comments
    processed_comments = preprocess_comments(comments)
    
    # Chunk comments for efficient processing
    chunks = chunk_comments(processed_comments)
    
    # Generate summaries for each chunk
    chunk_summaries = [generate_summary(chunk) for chunk in chunks]
    
    # Aggregate summaries into final bullet points
    final_summary = aggregate_summaries(chunk_summaries)

    # Analyze sentiment using both models
    vader_scores = {"pos": 0, "neu": 0, "neg": 0}
    distilbert_scores = {"POSITIVE": 0, "NEGATIVE": 0}
    detailed_sentiment = []
    
    # For calculating averages
    total_vader_score = 0
    total_distilbert_score = 0
    valid_comments = 0

    for comment in processed_comments:
        if not comment.strip():
            continue

        # VADER analysis
        vs = vader.polarity_scores(comment)
        if vs["compound"] >= 0.05:
            vader_scores["pos"] += 1
        elif vs["compound"] <= -0.05:
            vader_scores["neg"] += 1
        else:
            vader_scores["neu"] += 1

        # DistilBERT analysis
        try:
            distilbert_result = distilbert_pipeline(comment)[0]
            distilbert_scores[distilbert_result['label']] += 1
            
            # Store detailed sentiment for each comment
            detailed_sentiment.append({
                'text': comment,
                'vader_score': vs['compound'],
                'distilbert_label': distilbert_result['label'],
                'distilbert_score': distilbert_result['score']
            })
            
            # Update averages
            total_vader_score += vs['compound']
            total_distilbert_score += distilbert_result['score']
            valid_comments += 1
            
        except Exception as e:
            print(f"DistilBERT analysis error: {e}")

    total = sum(vader_scores.values())
    if total == 0: total = 1  # avoid division by zero

    # Calculate weighted sentiment using both models
    vader_positive = (vader_scores["pos"] / total) * 100
    distilbert_positive = (distilbert_scores["POSITIVE"] / total) * 100
    
    # Combine scores with more weight to DistilBERT
    final_positive = (vader_positive * 0.3 + distilbert_positive * 0.7)
    final_negative = 100 - final_positive

    # Calculate averages
    avg_vader_score = total_vader_score / valid_comments if valid_comments > 0 else 0
    avg_distilbert_score = total_distilbert_score / valid_comments if valid_comments > 0 else 0

    # Prepare sentiment data for theme analysis
    sentiment_data = {
        "positive": round(final_positive, 2),
        "neutral": round((vader_scores["neu"] / total) * 100, 2),
        "negative": round(final_negative, 2),
        "avg_vader_score": round(avg_vader_score, 2),
        "avg_distilbert_score": round(avg_distilbert_score, 2)
    }

    # Generate theme analysis
    theme_analysis = generate_theme_analysis(
        comments=processed_comments,
        sentiment_data=sentiment_data,
        detailed_sentiment=detailed_sentiment
    )

    return {
        "positive": round(final_positive, 2),
        "neutral": round((vader_scores["neu"] / total) * 100, 2),
        "negative": round(final_negative, 2),
        "summary": final_summary,
        "detailed_sentiment": detailed_sentiment,
        "avg_vader_score": round(avg_vader_score, 2),
        "avg_distilbert_score": round(avg_distilbert_score, 2),
        "theme_analysis": theme_analysis
    }