from django.shortcuts import render
from django.views.generic import TemplateView
from django import http

# from dev.forms import PhotoForm
# from dev.models import Photo
from PIL import Image
from backend import integration as ocr
import numpy as np

# Create your views here.
class HomeView(TemplateView):
    def get(self, request):
        # context = {'form':PhotoForm}
        # return render(request,'index.html', context)
        return render(request,'index.html')

def api(request):
    if(request.method == 'POST'):
        # form = PhotoForm(request.POST or None, request.FILES or None)
        temp = []
        files = request.FILES.getlist('files')
        for file in files:
            temp.append(file)
        img = np.array(Image.open(temp[0]).convert('RGB'))
        try:
            results = ocr.fullProcess(img)
            print(results)
            return render(request, 'display.html', {'riders':results})
        except:
            print("Error on form")
            return render(request, 'partials/toast.html', {'message':"There has been an error with your form"})
        # if form.is_valid():
        #     form.save()
        
        

    if(request.method == 'GET'):
        return render(request, 'partials/camera.html')
    
