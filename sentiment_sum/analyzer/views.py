from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from .sentiment import analyze_sentiment
from .models import UploadedFile
from .file_processor import FileProcessor
import logging
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework import status

logger = logging.getLogger(__name__)

# Create your views here.
def home(request):
    sentiment_data = None
    file_processor = FileProcessor()
    
    if request.method == 'POST':
        if 'file' in request.FILES:
            uploaded_file = request.FILES['file']
            
            try:
                # Validate file
                if uploaded_file.size > 10 * 1024 * 1024:  # 10MB limit
                    return render(request, 'analyzer/home.html', {
                        'error': 'File too large. Please upload files smaller than 10MB.'
                    })
                
                # Save the file
                fs = FileSystemStorage()
                filename = fs.save(uploaded_file.name, uploaded_file)
                file_path = fs.path(filename)
                
                # Validate file format and content
                try:
                    file_info = file_processor.validate_file(file_path)
                    if not file_info['is_supported']:
                        return render(request, 'analyzer/home.html', {
                            'error': f'Unsupported file format: {file_info["file_extension"]}. Please upload .txt, .csv, or .tsv files.'
                        })
                except Exception as e:
                    return render(request, 'analyzer/home.html', {
                        'error': f'File validation error: {str(e)}'
                    })
                
                # Create database entry
                UploadedFile.objects.create(
                    file=filename,
                    file_type=file_info['file_extension'][1:]  # Remove the dot
                )
                
                # Process the file using the new FileProcessor
                try:
                    # Get processing options from form (if any)
                    comment_column = request.POST.get('comment_column')
                    has_header = request.POST.get('has_header', 'true').lower() == 'true'
                    delimiter = request.POST.get('delimiter', ',')
                    
                    # Convert comment_column to int if provided
                    if comment_column:
                        try:
                            comment_column = int(comment_column)
                        except ValueError:
                            comment_column = None
                    
                    comments = file_processor.process_file(
                        file_path,
                        comment_column=comment_column,
                        has_header=has_header,
                        delimiter=delimiter
                    )
                    
                    if not comments:
                        return render(request, 'analyzer/home.html', {
                            'error': 'No valid comments found in the file. Please check the file format and content.'
                        })
                    
                    logger.info(f"Successfully processed {len(comments)} comments from file")
                    sentiment_data = analyze_sentiment(comments)
                    
                except Exception as e:
                    logger.error(f"Error processing file: {e}")
                    return render(request, 'analyzer/home.html', {
                        'error': f'Error processing file: {str(e)}'
                    })
                
                finally:
                    # Clean up the file
                    try:
                        fs.delete(filename)
                    except Exception as e:
                        logger.warning(f"Error deleting temporary file: {e}")
                
            except Exception as e:
                logger.error(f"File upload error: {e}")
                return render(request, 'analyzer/home.html', {
                    'error': f'File upload error: {str(e)}'
                })
            
        elif 'comments' in request.POST:
            comments_text = request.POST.get('comments', '').strip()
            if not comments_text:
                return render(request, 'analyzer/home.html', {
                    'error': 'Please enter some comments to analyze.'
                })
            
            # Split comments by newlines and filter empty lines
            comments = [line.strip() for line in comments_text.split('\n') if line.strip()]
            
            if not comments:
                return render(request, 'analyzer/home.html', {
                    'error': 'No valid comments found. Please check your input.'
                })
            
            sentiment_data = analyze_sentiment(comments)

    return render(request, 'analyzer/home.html', {'sentiment_data': sentiment_data})

class SentimentAnalysisAPI(APIView):
    parser_classes = (MultiPartParser, FormParser)

    def post(self, request, format=None):
        file = request.FILES.get('file')
        comments = request.data.get('comments')
        # ... (reuse your file processing logic here)
        # Return JSON response
        return Response(sentiment_data, status=status.HTTP_200_OK)

