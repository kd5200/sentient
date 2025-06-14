from django.shortcuts import render
from .sentiment import analyze_sentiment

# Create your views here.
def home(request):
    sentiment_data = None
    if request.method == 'POST':
        comments = request.POST.get('comments')
        sentiment_data = analyze_sentiment(comments.split('\n'))

    return render(request, 'analyzer/home.html', {'sentiment_data': sentiment_data})

