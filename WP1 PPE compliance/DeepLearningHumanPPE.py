
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

from DeepLearningPoseEstimator import DeepLearningPoseEstimator
from DeepLearningClassificator import DeepLearningClassificator

'''
def nacrtajBBoxOkoTacke(img, center, width, height, outline_color='red'):
    draw = ImageDraw.Draw(img)
    draw.rectangle(((center[1]-width, center[0]-height), (center[1]+width, center[0]+height)), outline = outline_color, width=3)
    return img

# Fja crta rectangle oko tacke 
#INPUTS
    #img   - slika na kojoj se crta (PIL format)
    #tacka - tacka oko koje se crta bbox
    #w     - sirina bboxa
    #h     - visina bboxa 
def nacrtajBBoxOkoSake(img, wrist, elbow, outline_color='red', width=-1, height=-1):
    vPravcaPodlaktice      = wrist[0:2]-elbow[0:2]
    vPravcaPodlaktice_norm = np.linalg.norm(vPravcaPodlaktice)
    #vPravcaPodlaktice_unit = vPravcaPodlaktice / vPravcaPodlaktice_norm  
    if width<0:  
        width  = vPravcaPodlaktice_norm * 0.4
        height = vPravcaPodlaktice_norm * 0.4
    center = wrist[0:2] + vPravcaPodlaktice * 0.3
    img = nacrtajBBoxOkoTacke(img, center, width, height, outline_color)
    return img

def nacrtajBBoxOkoGlave(img, left_ear, right_ear, outline_color='red'):
    rastojanje_left_right_ear = np.linalg.norm(left_ear[0:2]-right_ear[0:2])  
    width                     = rastojanje_left_right_ear * 1.0
    height                    = rastojanje_left_right_ear  * 1.0
    center                    = (left_ear[0:2]+right_ear[0:2]) * 0.5
    img = nacrtajBBoxOkoTacke(img, center, width, height, outline_color)
    return img, width, height 

'''

class HumanPPE():
    # Deep Learning modeli za proveru PPE   
    DL_PoseEstimator         = [] 
    
    device            = None

    DL_ClasifikatoriPPE     = { 'glava' : list(),
                                'trup'  : list(),
                                'ruke'  : list(),
                                'noge'  : list(),
                                'telo'  : list(),
                                }   

    DictBBoxVsClassifikator = { 'glava'      : 'glava',
                                'trup'       : 'trup',
                                'ruka_leva'  : 'ruke',
                                'ruka_desna' : 'ruke',
                                'noga_leva'  : 'noge',
                                'noga_desna' : 'noge',
                                'celo_telo'  : 'telo',
                                } 

    # Konstruktor
    def __init__(self,  DL_PoseEstimator = [os.path.join('03 Trenirani_DeepLearning_modeli','0 ModelPose','pose_hrnet_w32_384x288.pth'), 
                                        'SimpleHRNet', 
                                            {
                                                        'nose'          :  0,
                                                        'left_eye'      :  1,
                                                        'right_eye'     :  2,
                                                        'left_ear'      :  3,
                                                        'right_ear'     :  4,
                                                        'left_shoulder' :  5,
                                                        'right_shoulder':  6,
                                                        'left_elbow'    :  7,
                                                        'right_elbow'   :  8,
                                                        'left_wrist'    :  9,
                                                        'right_wrist'   : 10,
                                                        'left_hip'      : 11,
                                                        'right_hip'     : 12,
                                                        'left_knee'     : 13,
                                                        'right_knee'    : 14,
                                                        'left_ankle'    : 15,
                                                        'right_ankle'   : 16,
                                            }
                                        ],
                        DL_ClasifikatorGlava = [ 
                                            [os.path.join('03 Trenirani_DeepLearning_modeli','1 Glava'    ,'Hamlet_model[MobileNetV2]_acc[0.9862068965517241]_epochs[10]_batch[4]_.pth'), 
                                            'MobileNetV2', 
                                            ['No Hamlet',    'Hamlet'      ]
                                            ],
                                            [os.path.join('03 Trenirani_DeepLearning_modeli','1 Glava'    ,'CovidMask_model[MobileNetV2]_acc[0.9839080459770115]_epochs[20]_batch[8]_.pth'), 
                                            'MobileNetV2', 
                                            ['No Covid mask',  'Covid mask'  ]
                                            ]
                                        ],
                        DL_ClasifikatorTrup = [
                                            [os.path.join('03 Trenirani_DeepLearning_modeli','2 Trup'    ,'YellowVests_model[MobileNetV2]_acc[0.9345794392523364]_epochs[10]_batch[16]_.pth'), 
                                            'MobileNetV2', 
                                            ['No Yellow Vest', 'Yellow Vest' ]
                                            ]
                                        ],
                        DL_ClasifikatorShake = [
                                            [os.path.join('03 Trenirani_DeepLearning_modeli','3 Shake'   ,'Gloves_model[MobileNetV2]_acc[0.9319196428571429]_epochs[10]_batch[16]_.pth'), 
                                            'MobileNetV2', 
                                            ['No Gloves', 'Covid gloves', 'Industry gloves']
                                            ]
                                        ],
                        DL_ClasifikatorStopala   = list(),
                        DL_ClasifikatorCeloTelo  = list(),
                        device = torch.device('cuda:0')
                ):     
        self.device = device
       
        # Pose
        self.DL_PoseEstimator = DeepLearningPoseEstimator(Path = DL_PoseEstimator[0], Annotations = DL_PoseEstimator[2],  Arhitektura = DL_PoseEstimator[1],  device = device)
    
        # Glava  
        for iModelGlava in DL_ClasifikatorGlava:
            tmp = DeepLearningClassificator(Path = iModelGlava[0],  Arhitektura = iModelGlava[1],  Annotations = iModelGlava[2],  device = device)
            self.DL_ClasifikatoriPPE['glava'].append(tmp)
        
        # Trup
        for iDL_ModelTrup in DL_ClasifikatorTrup:
            tmp = DeepLearningClassificator(Path = iDL_ModelTrup[0], Arhitektura = iDL_ModelTrup[1], Annotations = iDL_ModelTrup[2], device = device)
            self.DL_ClasifikatoriPPE['trup'].append(tmp)

        # Shake
        for iDL_ModelShake in DL_ClasifikatorShake:
            tmp = DeepLearningClassificator(Path = iDL_ModelShake[0], Arhitektura = iDL_ModelShake[1], Annotations = iDL_ModelShake[2], device = device)
            self.DL_ClasifikatoriPPE['ruke'].append(tmp)

        # Noge
        for iDL_ModelNoge in DL_ClasifikatorStopala:
            tmp = DeepLearningClassificator(Path = iDL_ModelNoge[0], Arhitektura = iDL_ModelNoge[1], Annotations = iDL_ModelNoge[2], device = device)
            self.DL_ClasifikatoriPPE['noge'].append(tmp)

        # Celo Telo
        for iDL_ModelTelo in DL_ClasifikatorCeloTelo:
            tmp = DeepLearningClassificator(Path = iDL_ModelTelo[0], Arhitektura = iDL_ModelTelo[1], Annotations = iDL_ModelTelo[2], device = device)
            self.DL_ClasifikatoriPPE['telo'].append(tmp)
   
    def resetCollections(self):
        self.DL_PoseEstimator.resetCollections()


    def inference(self, image=None, DebugMode=False, BlurFace=False):
        # Odradi pose estimaciju i uradi cropovanje regiona oko delova tela
        self.DL_PoseEstimator.inference(image)
        
        for deo_tela in self.DL_PoseEstimator.rezPoses_crop_regions: # Uzmi naziv tela id dict 
            for bbox in self.DL_PoseEstimator.rezPoses_crop_regions[deo_tela]: # na osnovu deo_tela uzmi BBoX za odgovarajuci deo tela
                if bbox != None:
                    for iKlasifikator in self.DL_ClasifikatoriPPE[ self.DictBBoxVsClassifikator[deo_tela] ]: # selektuje klasifikator napravljen za deo_tela
                        bbox.KlasifikujBBox(iKlasifikator) # klasifikuj bbox

        image = self.DL_PoseEstimator.iscrtajBoundingBoxovePrekoImg() # iscrtaj anotacije preo bbox-ova

        if BlurFace:
           for bbox in self.DL_PoseEstimator.rezPoses_crop_regions['glava']:
               image = bbox.blurujBBox(image)        
        
        return image


    def inferenceVideo(self, input_video=None, output_video=None, DebugMode=False):
        cap        = cv2.VideoCapture(input_video)
        fps        = int(cap.get(cv2.CAP_PROP_FPS))
        n_frames   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_size = (height, width)

        #0x00000021
        try:
            out = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*'MP4V'), fps, frame_size)

            while(cap.isOpened()):
                ret, img_cv2 = cap.read()                
                img_cv2 = cv2.rotate(img_cv2, cv2.ROTATE_90_CLOCKWISE)
                if ret == False:
                    break
                self.DL_PoseEstimator.resetCollections()
                img_cv2_annotated = self.inference(img_cv2)
                out.write(img_cv2_annotated)
        except:
            print("Pukao inferenceVideo !!!")    
        cap.release()
        out.release()

    def inferenceLiveUSB(self):
        cap = cv2.VideoCapture(0)

        while(True):
            self.DL_PoseEstimator.resetCollections()
            # Capture frame-by-frame
            ret, frame = cap.read()

            if type(frame) != type(None):
                img_cv2_annotated = self.inference(frame)
            cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
            cv2.setWindowProperty('frame', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            # Display the resulting frame
            cv2.imshow('frame',img_cv2_annotated)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # When everything done, release the capture
        cap.release()
        cv2.destroyAllWindows()
