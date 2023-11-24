from django.shortcuts import render
from django.views.generic import TemplateView
from django import http
import json
import cv2
import base64

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
        debugList = []
        inputBox = list(request.POST.dict())
        print(inputBox)
        files = request.FILES.getlist('files')
        scorecards = []
        for i,file in enumerate(files):
            # temp.append(file)
            img = np.array(Image.open(file).convert('RGB'))
            try:
                if ("blueink" + str(i)) in inputBox:
                    print("done blue")
                results, debug = ocr.fullProcess(img)
                if ("debug") in inputBox:
                    currDebug = []
                    for d in debug:
                        _, buffer = cv2.imencode('.png', d)
                        currDebug.append(base64.b64encode(buffer).decode('utf-8'))
                    debugList.append(currDebug)
                    
                scorecards.append(results)
            except Exception as e:
                print(f"Error on form {i}")
                print(e)
                return render(request, 'partials/toast.html', {'message':"There has been an error with your form"})
        # if form.is_valid():
        #     form.save()
        print(scorecards)

        return render(request, 'display.html', {'scorecards':scorecards, 'debug':debugList})
        
        

    if(request.method == 'GET'):
        return render(request, 'partials/camera.html')
    
