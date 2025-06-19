from typing import List, Dict, Any
from openai import OpenAI
from django.conf import settings
import json
import logging
from dotenv import load_dotenv
import os

load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

def _group_comments_by_sentiment(detailed_sentiment: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    """Group comments by their sentiment category."""
    return {
        "positive": [item["text"] for item in detailed_sentiment 
                    if item["vader_score"] > 0.05 and item["distilbert_label"] == "POSITIVE"],
        "negative": [item["text"] for item in detailed_sentiment 
                    if item["vader_score"] < -0.05 and item["distilbert_label"] == "NEGATIVE"],
        "neutral": [item["text"] for item in detailed_sentiment 
                   if -0.05 <= item["vader_score"] <= 0.05]
    }

def _extract_themes_from_content(content: str) -> List[str]:
    """Extract themes from content using various formats."""
    themes = []
    try:
        # Split content into lines and process each line
        lines = content.split("\n")
        in_themes_section = False
        
        for line in lines:
            line = line.strip()
            if "Summary of Main Themes" in line or "Main Themes" in line:
                in_themes_section = True
                continue
            elif in_themes_section and ("Explanation" in line or "Classification" in line):
                break
                
            if in_themes_section:
                # Remove markdown formatting
                line = line.replace("**", "").replace("*", "")
                if line.startswith(("1.", "2.", "3.", "4.", "5.")):
                    themes.append(line[3:].strip())
                elif line and not line.startswith(("###", "---")):
                    themes.append(line)
                    
        logger.info(f"Extracted themes: {themes}")
        return themes
    except Exception as e:
        logger.error(f"Error extracting themes: {str(e)}")
        return []

def _extract_explanation_from_content(content: str) -> str:
    """Extract explanation from content."""
    try:
        # Split content into sections
        sections = content.split("###")
        
        # Look for explanation section with more flexible matching
        for section in sections:
            if any(keyword in section for keyword in ["Explanation", "Classification", "Sentiment Analysis"]):
                # Clean up the explanation text
                lines = section.split("\n")
                explanation_lines = []
                in_explanation = False
                
                for line in lines:
                    line = line.strip()
                    # Start collecting explanation after finding the header
                    if any(keyword in line for keyword in ["Explanation", "Classification", "Sentiment Analysis"]):
                        in_explanation = True
                        continue
                    # Stop if we hit another section header
                    elif line.startswith("###") or line.startswith("---"):
                        break
                    elif in_explanation and line:
                        # Remove markdown formatting
                        line = line.replace("**", "").replace("*", "").strip()
                        if line and not line.startswith(("###", "---", "##")):
                            explanation_lines.append(line)
                        
                if explanation_lines:
                    return " ".join(explanation_lines)
        
        # If no structured explanation found, try to extract from the end of the content
        lines = content.split("\n")
        explanation_lines = []
        found_explanation = False
        
        for line in reversed(lines):  # Start from the end
            line = line.strip()
            if any(keyword in line for keyword in ["Explanation", "Classification", "Sentiment Analysis"]):
                found_explanation = True
                break
            elif found_explanation and line:
                explanation_lines.insert(0, line)
        
        if explanation_lines:
            return " ".join(explanation_lines)
                
        return "No detailed explanation available"
    except Exception as e:
        logger.error(f"Error extracting explanation: {str(e)}")
        return "Error extracting explanation"

def _get_response_content(response) -> str:
    """Get content from response using various methods."""
    try:
        # Log the response structure for debugging
        logger.info(f"Response type: {type(response)}")
        logger.info(f"Response attributes: {dir(response)}")
        
        # Try to get content from the response object
        if hasattr(response, 'output'):
            logger.info("Found output attribute")
            # Handle list of ResponseOutputMessage objects
            if isinstance(response.output, list) and response.output:
                first_message = response.output[0]
                if hasattr(first_message, 'content') and first_message.content:
                    first_content = first_message.content[0]
                    if hasattr(first_content, 'text'):
                        return first_content.text
            return str(response.output)
        elif hasattr(response, 'text'):
            logger.info("Found text attribute")
            return response.text
        elif hasattr(response, 'content'):
            logger.info("Found content attribute")
            return response.content
        elif hasattr(response, 'choices') and response.choices:
            logger.info("Found choices attribute")
            return response.choices[0].text
        elif hasattr(response, 'messages') and response.messages:
            logger.info("Found messages attribute")
            return response.messages[-1].content
        elif hasattr(response, 'response'):
            logger.info("Found response attribute")
            return response.response
            
        # If all else fails, convert the entire response to string
        logger.info("No specific content found, converting response to string")
        return str(response)
    except Exception as e:
        logger.error(f"Error getting response content: {str(e)}")
        return str(response)

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
        logger.info("Starting theme analysis...")
        logger.info(f"API Key configured: {'Yes' if client else 'No'}")
        
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
        grouped_comments = _group_comments_by_sentiment(detailed_sentiment)

        logger.info(f"Processing {len(comments)} comments")
        logger.info(f"Positive comments: {len(grouped_comments['positive'])}")
        logger.info(f"Negative comments: {len(grouped_comments['negative'])}")
        logger.info(f"Neutral comments: {len(grouped_comments['neutral'])}")

        # Format comments for analysis
        comments_text = "\n".join([f"{i+1},\"{comment}\"" for i, comment in enumerate(comments)])

        logger.info("Calling OpenAI API...")
        response = client.responses.create(
            model="gpt-4o-mini",
            input=[
                {
                    "role": "user",
                    "content": [{
                        "type": "input_text",
                        "text": "1. 3-5 bullet points summarizing the main themes and topics discussed\n2. An explanation of why the comments were classified as they were, considering:\n   - Emotional tone and language patterns\n   - Sarcasm or irony if present\n   - Common phrases or expressions\n   - Any notable sentiment patterns"
                    }]
                },
                {
                    "role": "user",
                    "content": [{
                        "type": "input_text",
                        "text": f"id,text\n{comments_text}"
                    }]
                }
            ],
            text={"format": {"type": "text"}},
            temperature=0.7,
            max_output_tokens=2048,
            top_p=1,
            store=True
        )
        logger.info("Received response from OpenAI API")

        content = _get_response_content(response)
        logger.info(f"Raw response content: {content}")
        
        # Log the content structure for debugging
        logger.info(f"Content length: {len(content)}")
        logger.info(f"Content sections (###): {len(content.split('###'))}")
        logger.info(f"Content lines: {len(content.split(chr(10)))}")
        
        # Try to parse the content as JSON first
        try:
            content_dict = json.loads(content)
            if isinstance(content_dict, dict):
                logger.info("Successfully parsed content as JSON")
                return {
                    "themes": content_dict.get("themes", ["No specific themes identified"]),
                    "explanation": content_dict.get("explanation", "No detailed explanation available")
                }
        except json.JSONDecodeError:
            logger.info("Content is not JSON, proceeding with text parsing")
        
        themes = _extract_themes_from_content(content)
        explanation = _extract_explanation_from_content(content)
        
        logger.info(f"Extracted themes: {themes}")
        logger.info(f"Extracted explanation: {explanation}")
        logger.info(f"Explanation length: {len(explanation)}")
        
        return {
            "themes": themes if themes else ["No specific themes identified"],
            "explanation": explanation if explanation else "No detailed explanation available"
        }

    except Exception as e:
        logger.error(f"Theme analysis error: {str(e)}")
        return {
            "themes": ["Error generating themes"],
            "explanation": f"Unable to generate theme analysis at this time. Error: {str(e)}"
        } 