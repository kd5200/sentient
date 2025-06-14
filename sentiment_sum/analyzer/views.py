from django.shortcuts import render
from .sentiment_sum import analyze_sentiment  # This will be our custom logic

# Create your views here.
def home(request):
    sentiment_data = None
    if request.method == 'POST':
        comments = request.POST.get('comments')
        sentiment_data = analyze_sentiment(comments.split('\n'))

    return render(request, 'home.html', {'sentiment_data': sentiment_data})

