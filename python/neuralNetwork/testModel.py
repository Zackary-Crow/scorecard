import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from PIL import Image

import numeralRecognition

# load image
img = Image.open("python/generated/img1.png")
img = img.convert("L")

model = numeralRecognition.CNN()
model.load_state_dict(torch.load("src/neuralNetwork/scorecardCNN.pt"))

transform = transforms.ToTensor()

# Transform
input = transform(img)

#torch.set_printoptions(edgeitems=14)
#print(input)

# unsqueeze batch dimension, in case you are dealing with a single image
input = input.unsqueeze(0)
print(input.size())

model.eval()
with torch.no_grad():
    output = model(input)

print(output)

pred = torch.argmax(output, 1)
print(pred)
