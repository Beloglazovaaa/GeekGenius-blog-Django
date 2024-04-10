from django.db import models
from django.contrib.auth.models import User
from ckeditor_uploader.fields import RichTextUploadingField
from django.utils import timezone

class DataModel(models.Model):
    feature1 = models.FloatField()
    feature2 = models.FloatField()
    target = models.FloatField()

class Post(models.Model):
    h1 = models.CharField(max_length=200)
    title = models.CharField(max_length=200)
    url = models.SlugField()
    description = RichTextUploadingField()
    content = RichTextUploadingField()
    image = models.ImageField()
    created_at = models.DateTimeField()
    author = models.ForeignKey(User, on_delete=models.CASCADE)
    tag = models.CharField(max_length=200)