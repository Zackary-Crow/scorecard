import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

from . import numeralRecognition

def passPixelArray(arr):
    # load image

    model = numeralRecognition.CNN()
    model.load_state_dict(torch.load("src/neuralNetwork/scorecardCNN.pt",map_location=torch.device('cpu')))

    transform = transforms.ToTensor()

    i = 0
    for group in arr:
        i+=1
        print(f"group {i}")
        for item in group:
            # Transform
            input = transform(np.array(item))

            #torch.set_printoptions(edgeitems=14)
            # print(input)
            
            # unsqueeze batch dimension, in case you are dealing with a single image
            input = input.unsqueeze(0)
            # print(input.size())

            model.eval()
            with torch.no_grad():
                output = model(input)

            print(output)

            pred = torch.argmax(output, 1)
            print(pred)
