'''
    BoundingBox omogucava kropovanje  regija, njihovo arhirivanje, i iscrtavanje nazad na org sliku
'''

import numpy as np
import time
import os
import cv2

class BoundingBox ():
    img    = None # Crop regija
    Center = None # Centar crop regije
    Width  = None # Sirina
    Height = None # Visina
    Annotations = None # Lista anotacija koje potencijalno treba ispisati pored BBox-a, npr. rezultat klasifikacije

    # Konstruktor
    def __init__(self,  img = None, Center = None,  Width = None,  Height = None, Annotations = list()) :
        imgShape = img.shape
        if imgShape[0]==0   or imgShape[1]==0:
            return None
        self.img = img.copy()
        self.Center      = Center.copy()
        self.Width       = int(Width)
        self.Height      = int(Height)
        self.Annotations = Annotations.copy()

    # Iscrtavanje BBox-a na vecoj-originalnoj slici sa koje je cropovan bbox (prilikom detekcije, anotiranja, itd.)
    # Fja ima opciju da iscrtava dinamicne anotacije - kao fju FPS
    # INPUTS
        # image - velika/originalna slika
    # OUTPUTS
        # image sa iscrtanim regijama    
    def iscrtajBoundingBoxNaSlici(self,  image):
        # definisanje globalne varijable: global FrameID
        if 'FrameID' in globals():
            do =1 # uradi animaciju koja je fja-broja frejmova
        # Blue color in BGR 
        color = (0, 0, 255) 
        # Line thickness of 2 px 
        thickness = 1
        # Using cv2.rectangle() method 
        # Draw a rectangle with blue line borders of thickness of 2 px 
        start_point = self.Center - (self.Width, self.Height)
        start_point = tuple(np.int(a)  for a in start_point[::-1])
        end_point   = self.Center + (self.Width, self.Height)
        end_point   = tuple(np.int(a)  for a in end_point[::-1])
        image       = cv2.rectangle(image, start_point, end_point, color, thickness) 
        return image
    
    def blurujBBox(self, image=None):
        if type(image) == type(None):
            image = self.img
        
        # Bluruj celu kocku 
        x_min, y_min = self.Center + (-self.Width, -self.Height)
        x_min = np.int(x_min)
        y_min = np.int(y_min)

        x_max, y_max = self.Center + (+self.Width, +self.Height)
        x_max = np.int(x_max)
        y_max = np.int(y_max)

        pom = cv2.medianBlur(image[x_min:x_max, y_min:y_max,:], 11)
        image[x_min:x_max, y_min:y_max,:] = pom

        # Bluruj jache sredinu
        x_min, y_min = self.Center + (-self.Width * 0.5, -self.Height * 0.5)
        x_min = np.int(x_min)
        y_min = np.int(y_min)

        x_max, y_max = self.Center + (+self.Width * 0.5, +self.Height * 0.5)
        x_max = np.int(x_max)
        y_max = np.int(y_max)

        pom = cv2.medianBlur(image[x_min:x_max, y_min:y_max,:], 17)
        image[x_min:x_max, y_min:y_max,:] = pom

        return image
    
    def ispisiLabeleBoxaNaSlici(self, image, headWidthHeight=0):   
        position = self.Center.copy()     
        position = self.Center + (-self.Width + 10, self.Height + 2)
        position = (np.int(position[1]), np.int(position[0]))
        brLabela = len(self.Annotations)

        for id in range(brLabela):
            labela         = self.Annotations[id]
            
            font           = cv2.FONT_HERSHEY_SIMPLEX
            if headWidthHeight > 0:
                font_scale     = 6* 0.3 * 2 * 150 / headWidthHeight
            else:
                font_scale     = 0.3 * 2 *62
            if labela[0:2] == 'No' :
                font_color     = (0, 0,   255, 255)
            else:
                font_color     = (0, 255, 0, 255)
            font_thickness = 1
            text_color_bg  = (0, 0, 0)

            font_scale = font_scale * 5

            text_size, _   = cv2.getTextSize(labela, font, font_scale, font_thickness)
            text_w, text_h = text_size
            
            position       = (position[0], position[1] + text_h + 4)
            x     , y      = position
#            cv2.rectangle(image, position, (x + text_w + 1, y - text_h + 1), text_color_bg, -1)

#            cv2.putText(
#                image, #numpy array on which text is written
#                labela, #text
#                position, #position at which writing has to start
#                font, #font family
#                font_scale, #font size
#                font_color, #font color
#                1) #font stroke
            a = 1

        return image

    def KlasifikujBBox(self, Klasifikator = None):
        cv2.imwrite('tmp.jpg', self.img)
        pred_class, pred_label = Klasifikator.inference('tmp.jpg')#self.img.copy())
        pred_class, pred_label = Klasifikator.inference(self.img.copy())

        self.Annotations.append(str(pred_label))

    def getLabels(self):
        return self.Annotations

    def getImage(self):
        return self.img.copy()
