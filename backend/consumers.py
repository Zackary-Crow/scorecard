from channels.generic.websocket import WebsocketConsumer
import json
from django.template.loader import get_template
import base64
import cv2
# from .scanForm import docFind, warpDocument
from . import integration as ocr
# import integration as ocr

class CameraConsumer(WebsocketConsumer):
    def connect(self):
        self.accept()
        self.send(text_data=json.dumps({
            'type':'connection_established',
            'message':'poggers, connected'}))
    def receive(self, text_data):
        data = json.loads(text_data)['data'].split(',')[1]
        if(data == ""):
            return
        if(data != None):
            eval, c, img = ocr.docFind(data)
            if(eval):
                img, corners = ocr.straightenImage(img, c)
                img, maxHeight = ocr.centerImage(img, corners)
                _, im_arr = cv2.imencode('.png', img)  # im_arr: image in Numpy one-dim array format.
                im_bytes = im_arr.tobytes()
                im_b64 = str(base64.b64encode(im_bytes)).split("'")[1]
                html = get_template("partials/display.html").render(
                    context = {
                        'base64': im_b64
                    }
                )
                self.send(text_data=html)
                print("document found")
        print("no document found")
    
    def disconnect(self, close_code):
        pass