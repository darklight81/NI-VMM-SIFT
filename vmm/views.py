from django.shortcuts import render
from django.views import View
from distance_computation import compute

class Index(View):
    template = 'index.html'

    def get(self, request):
        return render(request, self.template)

    def post(self, request):
        handle_uploaded_file(request.FILES['query_img'])
        results = compute('static/images/uploaded.jpg', desc_num=1000, method=0)
        return render(request, self.template, {'results': results})


def handle_uploaded_file(f):
    file_type = f.name.split('.')[-1]
    with open('static/images/uploaded.' + file_type, 'wb+') as destination:
        for chunk in f.chunks():
            destination.write(chunk)
