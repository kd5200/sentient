from typing import List, Dict, Any
import openai
from django.conf import settings
import json

# Configure OpenAI client
openai.api_key = settings.OPENAI_API_KEY

def generate_theme_analysis(
    comments: List[str],
    sentiment_data: Dict[str, Any],
    detailed_sentiment: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Generate theme analysis and explanations using LLM.
    
    Args:
        comments: List of comment texts
        sentiment_data: Overall sentiment statistics
        detailed_sentiment: Detailed sentiment analysis for each comment
    
    Returns:
        Dict containing theme analysis and explanations
    """
    try:
        # Prepare the prompt with sentiment context
        sentiment_context = {
            "positive_percent": sentiment_data["positive"],
            "neutral_percent": sentiment_data["neutral"],
            "negative_percent": sentiment_data["negative"],
            "total_comments": len(comments),
            "avg_vader_score": sentiment_data["avg_vader_score"],
            "avg_distilbert_score": sentiment_data["avg_distilbert_score"]
        }

        # Group comments by sentiment for better context
        positive_comments = [
            item["text"] for item in detailed_sentiment 
            if item["vader_score"] > 0.05 and item["distilbert_label"] == "POSITIVE"
        ]
        negative_comments = [
            item["text"] for item in detailed_sentiment 
            if item["vader_score"] < -0.05 and item["distilbert_label"] == "NEGATIVE"
        ]
        neutral_comments = [
            item["text"] for item in detailed_sentiment 
            if -0.05 <= item["vader_score"] <= 0.05
        ]

        # Construct the prompt
        prompt = f"""Analyze these user comments and provide insights about their themes and sentiment patterns.

Context:
- Total comments: {sentiment_context['total_comments']}
- Sentiment distribution: {sentiment_context['positive_percent']}% positive, {sentiment_context['neutral_percent']}% neutral, {sentiment_context['negative_percent']}% negative
- Average sentiment scores: VADER={sentiment_context['avg_vader_score']}, DistilBERT confidence={sentiment_context['avg_distilbert_score']}

Sample comments by sentiment:
Positive comments:
{json.dumps(positive_comments[:3], indent=2)}

Negative comments:
{json.dumps(negative_comments[:3], indent=2)}

Neutral comments:
{json.dumps(neutral_comments[:3], indent=2)}

Please provide:
1. 3-5 bullet points summarizing the main themes and topics discussed
2. An explanation of why the comments were classified as they were, considering:
   - Emotional tone and language patterns
   - Sarcasm or irony if present
   - Common phrases or expressions
   - Any notable sentiment patterns

Format the response as JSON with these keys:
{{
    "themes": ["theme1", "theme2", ...],
    "explanation": "detailed explanation text"
}}
"""

        # Call the LLM
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that analyzes text sentiment and themes."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=500
        )

        # Parse and return the response
        analysis = json.loads(response.choices[0].message.content)
        return {
            "themes": analysis["themes"],
            "explanation": analysis["explanation"]
        }

    except Exception as e:
        print(f"Theme analysis error: {e}")
        return {
            "themes": ["Error generating themes"],
            "explanation": "Unable to generate theme analysis at this time."
        } 