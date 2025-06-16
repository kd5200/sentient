from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from .sentiment import analyze_sentiment
from .models import UploadedFile
import csv
import os

# Create your views here.
def home(request):
    sentiment_data = None
    if request.method == 'POST':
        if 'file' in request.FILES:
            uploaded_file = request.FILES['file']
            file_type = os.path.splitext(uploaded_file.name)[1].lower()
            
            if file_type not in ['.txt', '.csv']:
                return render(request, 'analyzer/home.html', {
                    'error': 'Please upload only .txt or .csv files'
                })
            
            # Save the file
            fs = FileSystemStorage()
            filename = fs.save(uploaded_file.name, uploaded_file)
            
            # Create database entry
            UploadedFile.objects.create(
                file=filename,
                file_type=file_type[1:]  # Remove the dot
            )
            
            # Process the file
            file_path = fs.path(filename)
            comments = []
            
            if file_type == '.txt':
                with open(file_path, 'r', encoding='utf-8') as f:
                    comments = [line.strip() for line in f if line.strip()]
            else:  # CSV
                with open(file_path, 'r', encoding='utf-8') as f:
                    reader = csv.reader(f)
                    # Assuming the first column contains the comments
                    comments = [row[0].strip() for row in reader if row and row[0].strip()]
            
            sentiment_data = analyze_sentiment(comments)
            
            # Clean up the file
            fs.delete(filename)
            
        elif 'comments' in request.POST:
            comments = request.POST.get('comments')
            sentiment_data = analyze_sentiment(comments.split('\n'))

    return render(request, 'analyzer/home.html', {'sentiment_data': sentiment_data})

