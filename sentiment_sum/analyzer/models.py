from django.db import models

# Create your models here.

class UploadedFile(models.Model):
    file = models.FileField(upload_to='uploads/')
    uploaded_at = models.DateTimeField(auto_now_add=True)
    file_type = models.CharField(max_length=10)  # 'txt' or 'csv'
    
    def __str__(self):
        return f"{self.file.name} ({self.file_type})"
