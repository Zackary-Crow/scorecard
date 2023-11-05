from channels.generic.websocket import WebsocketConsumer
import json
from django.template.loader import get_template
from .scanForm import scanData , docFind, warpDocument

class CameraConsumer(WebsocketConsumer):
    def connect(self):
        self.accept()
        self.send(text_data=json.dumps({
            'type':'connection_established',
            'message':'poggers, connected'}))
    def receive(self, text_data= None):
        data = json.loads(text_data)['test'].split(',')[1]
        if(data != None):
            if(docFind(data)):
                warpImg = warpDocument(data)
                html = get_template("partials/display.html").render(
                    context = {
                        'base64': warpImg
                    }
                )
                self.send(text_data=html)
                print("document found")
        print("no document found")
    
    def disconnect(self, close_code):
        pass