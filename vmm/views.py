import time
from pathlib import Path

from django.shortcuts import render
from django.views import View
from distance_computation import compute
from siftgen import generate_descriptors


class Index(View):
    template = 'index.html'

    def get(self, request):
        return render(request, self.template)

    def post(self, request):
        # Measure time
        start = time.time()
        handle_uploaded_file(request.FILES['query_img'])
        similarity = request.POST.get('similarity')
        desc_num = request.POST.get('range')
        if desc_num == 1000:
            desc_num = 0

        generate_descriptors(desc_num)
        results = compute('static/images/uploaded.jpg', desc_num=desc_num, method=0, similarity=similarity)

        # End time
        end = time.time()
        return render(request, self.template, {'results': results, 'paths': 'images/uploaded.jpg',
                                               'time': str(round(end - start, 2))})


def handle_uploaded_file(f):
    file_type = f.name.split('.')[-1]
    with open('static/images/uploaded.' + file_type, 'wb+') as destination:
        for chunk in f.chunks():
            destination.write(chunk)
