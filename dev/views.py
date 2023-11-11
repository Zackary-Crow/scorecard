from django.shortcuts import render
from django.views.generic import TemplateView
from django import http

from dev.forms import PhotoForm
from dev.models import Photo
from PIL import Image
from backend import integration as ocr
import numpy as np

# Create your views here.
class HomeView(TemplateView):
    template_name = "index.html"
    def get(self, request):
        context = {'form':PhotoForm}
        return render(request,'index.html', context)
    
def api(request):
    if(request.method == 'POST'):
        form = PhotoForm(request.POST or None, request.FILES or None)
        temp = request.FILES['data']
        img = np.array(Image.open(temp).convert('RGB'))
        results = ocr.fullProcess(img)
        print(results)
        # if form.is_valid():
        #     form.save()
        
        return render(request, 'display.html', {'riders':results})

    if(request.method == 'GET'):
        return render(request, 'partials/camera.html')
    
