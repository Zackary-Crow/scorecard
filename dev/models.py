from django.db import models

class Photo(models.Model):
    data = models.ImageField(upload_to='upload/') 
    # list = models
# Create your models here.
