from django.shortcuts import render
from django.views import View
from distance_computation import compute_distance

class Index(View):
    template = 'index.html'

    def get(self, request):
        return render(request, self.template)

    def post(self, request):
        handle_uploaded_file(request.FILES['query_img'])

        return render(request, self.template)


def handle_uploaded_file(f):
    file_type = f.name.split('.')[-1]
    with open('images/uploaded.' + file_type, 'wb+') as destination:
        for chunk in f.chunks():
            destination.write(chunk)
