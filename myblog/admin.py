from django.contrib import admin
from .models import Post, DataModel


class PostAdmin(admin.ModelAdmin):
    prepopulated_fields = {'url': ['title']}


admin.site.register([DataModel, Post])
