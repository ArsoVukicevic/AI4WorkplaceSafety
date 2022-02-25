
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.optim import lr_scheduler
import torchvision
import numpy as np
import time
import os
import argparse
import cv2
import sys

import PIL
from PIL import Image, ImageDraw

def napraviMobileNet_V2(ModelPath="ModelRuke\\MobileNetV2_acc=[0.95].pth", DebugMode=False, n_class=2, device=torch.device('cuda:0')):
    # Generisi model
    model_conv = torchvision.models.mobilenet_v2(pretrained='imagenet')
    if DebugMode:
        for name, params in model_conv.named_children():
            print(name)
    print("[Building Squeezenet]")
    num_ftrs                 = 1280 #model_conv.classifier.in_features
    model_conv.classifier[1] = nn.Linear(num_ftrs, n_class)
    # Ucitaj pretreniran model
    model_conv = torch.load(ModelPath)
    model_conv.to(device)
    return model_conv


# Ova klasa sluzi kao kontejner za ANN klasifikatore - koji se koriste za klasifikaciju regija da li se nose rukavice, slemovi itd.
# Napravljena je da bi bilo lakse raditi sa vecim br. klasifikatora za razlicite regije tela
class DeepLearningClassificator:
    ANN         = None # Objekat gde se cuva ucitani model
    Path        = None # Putanja do .pth treniranog modela
    Annotations = None # Anotacije po klasama - npr. ['Slem', 'Kacket', 'Gologlav']
    Arhitektura = None

    # Konstruktor - setuje se ANN koja se koristi za dati task
    # INPUTS
        # Path        - putanja gde je sacuvan model
        # Annotations - Anotacije po klasama - npr. ['Slem', 'Kacket', 'Gologlav']
        # Arhitektura - Naziv arhitekture, npr. 'MobileNetV2'
    def __init__(self,  Path = None,  Arhitektura = None,  Annotations = None,   device = None):
        if Arhitektura == None:
            self.ANN         = None
            self.Path        = None
            self.Annotations = None
            self.device      = None
            self.Arhitektura = None
        if Arhitektura == 'MobileNetV2':
            self.ANN         = napraviMobileNet_V2(Path,  DebugMode = False,  n_class = len(Annotations),  device = device)
            self.Path        = Path
            self.Annotations = Annotations
            self.device      = device
            self.Arhitektura = Arhitektura
        # Ovde dodaci sledecu arhitekturu

    # Fja vrsi klasifikaciju slike koja je prosledjena kao argument
    # INPUTS
        # image- slika koja se klasifikuje, putanja ili numpy.ndarray ucitane/kropovane
    # OUTPUTS
        # pred_class - predvidjena klasa (npr. 0, 1, 2, itd.)
        # pred_label - anotacija predvijdene kase (nor. slem, kapa, itd.)
    def inference(self, image):
        # Ako je path - ucitaj img
        if type(image) == str:
            image = PIL.Image.open(image)
        if type(image) ==  np.ndarray: 
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = PIL.Image.fromarray(image)
        # Izvrsi transformacije - pripremi za inference
        preprocess = transforms.Compose([
                        transforms.Resize(310),
                        transforms.CenterCrop(299),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                        ])
        image  = preprocess(image)
        image  = image.to(self.device)
        image  = image.unsqueeze(0) # create a mini-batch as expected by the model
        # Forwarduj PIL img to PyTorch
        outputs        = self.ANN(image)
        _, preds       = torch.max(outputs.data, 1)
        pred_class     = torch.Tensor.cpu(preds).detach().numpy()
        pred_class     = np.int(pred_class)
        pred_label     = self.Annotations[pred_class]
        return pred_class, pred_label # Vrati ID klase i labelu