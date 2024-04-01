from django.shortcuts import render, get_object_or_404
from django.views import View
from .models import Post
from django.http import Http404

class MainView(View):
    def get(self, request, *args, **kwargs):
        posts = Post.objects.all()
        return render(request, 'myblog/index.html', context={
            'posts': posts
        })


class PostDetailView(View):
    def get(self, request, slug, *args, **kwargs):
        post = get_object_or_404(Post, url=slug)
        return render(request, 'myblog/post_detail.html', context={
            'post': post
        })
