from django.shortcuts import render
from django.views.generic import TemplateView
from django.http import HttpResponse

from dev.forms import PhotoForm
from dev.models import Photo

# Create your views here.
class HomeView(TemplateView):
    template_name = "index.html"
    def get(self, request):
        context = {'form':PhotoForm}
        return render(request,'index.html', context)
    
def api(request):
    if(request.method == 'POST'):
        form = PhotoForm(request.POST or None, request.FILES or None)
        if form.is_valid():
            form.save()
    return render(request, 'partials/form.html', {'form':PhotoForm()})
    
