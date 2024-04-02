from django.shortcuts import render, get_object_or_404
from django.views import View
from .models import Post
from django.http import Http404
from django.core.paginator import Paginator

class MainView(View):
    def get(self, request, *args, **kwargs):
        posts = Post.objects.all().order_by('-created_at')
        paginator = Paginator(posts, 3)

        page_number = request.GET.get('page')
        page_obj = paginator.get_page(page_number)

        return render(request, 'myblog/index.html', context={
            'page_obj': page_obj
        })


class PostDetailView(View):
    def get(self, request, slug, *args, **kwargs):
        post = get_object_or_404(Post, url=slug)
        return render(request, 'myblog/post_detail.html', context={
            'post': post
        })

