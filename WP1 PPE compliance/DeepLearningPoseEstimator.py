'''
Klasa enkapsulira funkcionalnosti Pose-estimatora; Koji se moze pozivati u okviru dr. projekata/klasa.
Trenutno podrzani pose-estimatori su:
    1 - HRNet _w32_384x288 

'''

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

from DeepLearningBoundingBox import BoundingBox # Klasa za rad sa BBoX-ovima

# Folder gde se nalazi definicija HRNet za pose estimation
pathPoseEstimation = os.path.join('F:','_ARSO_DeepLearningCode','_Deep High-Resolution Representation Learning networks','Multi-person Human Pose Estimation with HRNet in PyTorch')
sys.path.append(os.path.abspath(pathPoseEstimation))
sys.path.append(os.path.join(pathPoseEstimation, 'models', 'detectors', 'yolo'))
# Importovanje klasa iz HRNet foldera
from SimpleHRNet import SimpleHRNet
from misc.visualization import draw_points, draw_skeleton, draw_points_and_skeleton, joints_dict, check_video_rotation
from misc.utils import find_person_id_associations

# Fja za pravljenje HRNet arhitekture
def napraviSimpleHRNet(ModelPath=os.path.join('ModelPose','pose_hrnet_w32_384x288.pth=[0.95].pth'), DebugMode=False, device=torch.device('cuda:0')):  
    print("[Building SimpleHRNet]")
    # Parametri 
    c                     = 32
    nof_joints            = 17
    model_name            = 'HRNet'
    resolution            = (384, 288)
    interpolation         = cv2.INTER_CUBIC
    multiperson           = True
    return_bounding_boxes = False
    max_batch_size        = 32
    yolo_model_def        = os.path.join(pathPoseEstimation,'models','detectors','yolo','config','yolov3.cfg')
    yolo_class_path       = os.path.join(pathPoseEstimation,'models','detectors','yolo','data','coco.names')
    yolo_weights_path     = os.path.join(pathPoseEstimation,'models','detectors','yolo','weights','yolov3.weights')
    # Poziv fje za pravljenje HRNet na osnovu parametara
    model = SimpleHRNet(c, nof_joints, ModelPath, model_name, resolution, interpolation, multiperson,
                        return_bounding_boxes, max_batch_size, yolo_model_def, yolo_class_path, yolo_weights_path, device)
    return model

class DeepLearningPoseEstimator():
    ANN                 = None # Objekat gde se cuva ucitani model
    Path                = None # Putanja do .pth treniranog modela
    Annotations         = None # Anotacije po klasama - npr. {'nose':0, 'lef_eye':1, ... itd.}
    Arhitektura         = None # string-naziv arhitekture. Podrzane su { HRNet, }
    rezPoses_pts        = None # Detektovane keypoints
    rezPoses_person_ids = None # Id ljudi na slikama
    rezImage            = None
    rezPoses_pts        = None
    
    # Konstruktor - setuje se ANN koja se koristi za pose estimation
    # INPUTS
    # Path        - putanja gde je sacuvan model
    # Annotations - Anotacije po klasama - npr. ['Slem', 'Kacket', 'GolaGlava', itd.]
    # Arhitektura - Naziv arhitekture, npr. 'MobileNetV2'
    def __init__(self,  Path = None, Arhitektura = 'SimpleHRNet',  Annotations = None,  device = None):
        self.Arhitektura = Arhitektura
        if Arhitektura == 'SimpleHRNet':
            self.ANN         = napraviSimpleHRNet(Path,  device = device)
            self.Path        = Path                
            self.device      = device
            self.Annotations = {
                                'nose'          :  0,
                                'left_eye'      :  1,
                                'right_eye'     :  2,
                                'left_ear'      :  3,
                                'right_ear'     :  4,
                                'left_shoulder' :  5,
                                'right_shoulder':  6,
                                'left_elbow'    :  7,
                                'right_elbow'   :  8,
                                'left_wrist'    :  9, # rucni zglob
                                'right_wrist'   : 10,
                                'left_hip'      : 11, # kuk
                                'right_hip'     : 12,
                                'left_knee'     : 13,
                                'right_knee'    : 14,
                                'left_ankle'    : 15,
                                'right_ankle'   : 16,
                                }
            self.rezPoses_pts        = None
            self.rezPoses_person_ids = None
            self.rezPoses_crop_regions = {
                                          'glava'      : list(),
                                          'trup'       : list(),
                                          'ruka_leva'  : list(),
                                          'ruka_desna' : list(),
                                          'noga_leva'  : list(),
                                          'noga_desna' : list(),
                                          'celo_telo'  : list(),  
                                        }
    # Pobrisi sve sto je detektovano na trenutnoj slici - i spremi se za sledeci inference
    def resetCollections(self):
        self.rezPoses_pts        = None
        self.rezPoses_person_ids = None
        self.rezPoses_crop_regions = {
                                       'glava'      : list(),
                                       'trup'       : list(),
                                       'ruka_leva'  : list(),
                                       'ruka_desna' : list(),
                                       'noga_leva'  : list(),
                                       'noga_desna' : list(),
                                       'celo_telo'  : list(),  
                                      }          

    def inference(self, img, DebugMode=False):
        # Ako je input OpenCV slika - prebaci u Image.PIL
        if type(img) == np.ndarray: # input je opencv ndray -BGR - prebaci u PIL RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = PIL.Image.fromarray(img)
        # Ako je input filepath - ucitaj sliku
        if type(img) == str:
            img = PIL.Image.open(img)

        if self.Arhitektura == 'SimpleHRNet':            
            pts                      = self.ANN.predict(np.asarray(img)[:,:,::-1].copy())
            self.rezImage            = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR).copy() # PIL to CV2
            self.rezPoses_pts        = pts.copy()
            self.rezPoses_person_ids = np.arange(len(pts), dtype=np.int32) # Br ljudi na slici  
            self.KropujRegijuGlave()
            self.KropujRegijuTrupa()
            self.KropujRegijuLeveRuke()
            self.KropujRegijuDesneRuke()
            self.KropujRegijuLeveNoge()
            self.KropujRegijuDesneNoge()
            self.KropujRegijuCelogTela()

            #self.iscrtajSekletonePrekoImg(image=None, DebugMode=DebugMode)
            #self.iscrtajBoundingBoxovePrekoImg(image=None, DebugMode=DebugMode)
            return pts
        # Default value
        return None

    # Vraca landmark tacku iz pts na osnovu njenog naziva iz self.Annotations
    # INPUTS
        # id_person - id coveka
        # keypoint  - deo tela
    def getKeypointZaIdCoveka(self, id_person, keypoint = 'left_ear'):
        pt = self.rezPoses_pts[id_person]
        return pt[self.Annotations[keypoint]]
    
    # Proverava da li je algoritam nasao sve delove tela - tj. da li su vidljivi svi delovi tela na slici
    def daLiJeNasaoSveLandmarkZaCoveka(self, id_person):
        pt = self.rezPoses_pts[id_person]
        if len(pt) == len(self.Annotations):
            return True
        else:
            return False

    # Fja za kropovanje regije oko glave
    def KropujRegijuGlave(self, DebugMode=False):
        img = self.rezImage.copy()
        id  = 0 # Brojach
        for iPerson in self.rezPoses_person_ids:
            try:
                left_ear                  = self.getKeypointZaIdCoveka(iPerson, 'left_ear' )
                right_ear                 = self.getKeypointZaIdCoveka(iPerson, 'right_ear')
                rastojanje_left_right_ear = np.linalg.norm(left_ear[0:2]-right_ear[0:2])
                
                pom1  = self.getKeypointZaIdCoveka(iPerson, 'left_elbow')
                pom2  = self.getKeypointZaIdCoveka(iPerson, 'right_elbow')
                pom2  = np.linalg.norm(right_ear[0:2]-pom2[0:2]) * 0.5
                pom1  = np.linalg.norm(left_ear[0:2]-pom1[0:2]) * 0.5
                
                pom   = 0.5 * (pom1 + pom2)
                
                width        = np.int(pom  * 0.85)
                height       = np.int(pom  * 0.85)
                center       = np.round( (left_ear[0:2]+right_ear[0:2]) * 0.5 )
                
                gornji_ugao  = center - (width,height)
                donji_ugao   = center + (width,height)
                imgCropGlava = img[ np.int(gornji_ugao[0]) : np.int(donji_ugao[0]), np.int(gornji_ugao[1]):np.int(donji_ugao[1]), :]
                
                bbox = BoundingBox(img=imgCropGlava, Center=center, Width=width, Height=height) # Narpavi BBox od cropovane regije
                if min(imgCropGlava.shape) > 2:
                    self.rezPoses_crop_regions['glava'].append(bbox) # Ubaci ga u kontejner gde se cuvaju kropovane regije glave

                if DebugMode: # ako je debug mode - sacuvaj sliku
                    id = id + 1
                    cv2.imwrite('ars' + str(id)  + '.jpg', imgCropGlava) 

            except:
                print("Exception u fji DeepLearningPoseEstimator.KropujRegijuGlave() !!!")
                self.rezPoses_crop_regions['glava'].append(None)
    
    # Fja za kropovanje regije oko trupa
    def KropujRegijuTrupa(self, DebugMode=False):
        img = self.rezImage.copy()
        id  = 0

        for iPerson in self.rezPoses_person_ids:
            try:
                
                left_hip       = self.getKeypointZaIdCoveka(iPerson, 'left_hip'      )
                right_hip      = self.getKeypointZaIdCoveka(iPerson, 'right_hip'     )
                left_shoulder  = self.getKeypointZaIdCoveka(iPerson, 'left_shoulder' )
                right_shoulder = self.getKeypointZaIdCoveka(iPerson, 'right_shoulder')

                rastojanje_hip_shoulder   = (np.linalg.norm(left_hip[0:2]-left_shoulder[0:2]) + np.linalg.norm(right_hip[0:2]-right_shoulder[0:2])) * 0.5
                
                width             = np.int(rastojanje_hip_shoulder  * 0.7)
                height            = np.int(rastojanje_hip_shoulder  * 0.55)
                center            = np.round((left_hip[0:2]+right_hip[0:2]+left_shoulder[0:2]+right_shoulder[0:2])*0.25)
                
                gornji_ugao       = center - (width,height)
                donji_ugao        = center + (width,height)
                
                imgCrop           = img[ np.int(gornji_ugao[0]) : np.int(donji_ugao[0]), np.int(gornji_ugao[1]):np.int(donji_ugao[1]), :]

                if DebugMode:
                    id = id + 1
                    cv2.imwrite('ars_Trup' + str(id) + '.jpg', imgCrop) 

                bbox = BoundingBox(img=imgCrop, Center=center, Width=width, Height=height)
                if min(imgCrop.shape) > 2:
                    self.rezPoses_crop_regions['trup'].append(bbox) 
            except:
                print("Exception u fji DeepLearningPoseEstimator.KropujRegijuTrupa() !!!")
                self.rezPoses_crop_regions['trup'].append(None)
    
    # Kropuj regiju oko leve ruke
    def KropujRegijuLeveRuke(self, DebugMode=False):
        img = self.rezImage.copy()
        id = 0
        for iPerson in self.rezPoses_person_ids:
            try:
                width_glave            = self.rezPoses_crop_regions['glava'][iPerson].Width
                width                  = width_glave  * 0.5
                height                 = width_glave * 0.5
                left_wrist             = self.getKeypointZaIdCoveka(iPerson, 'left_wrist')
                left_elbow             = self.getKeypointZaIdCoveka(iPerson, 'left_elbow')
                vPravcaPodlaktice      = left_wrist[0:2]-left_elbow[0:2]                
                center                 = left_wrist[0:2] + vPravcaPodlaktice * 0.4
                gornji_ugao            = center - (width,height)
                donji_ugao             = center + (width,height)
                 
                imgCrop                = img[ np.int(gornji_ugao[0]) : np.int(donji_ugao[0]), np.int(gornji_ugao[1]):np.int(donji_ugao[1]), :]
                if DebugMode:
                    id = id + 1
                    cv2.imwrite('ars_LevaRuka' + str(id) + '.jpg', imgCrop) 
                
                bbox = BoundingBox(img=imgCrop, Center=center, Width=width, Height=height)
                if min(imgCrop.shape) > 2:
                    self.rezPoses_crop_regions['ruka_leva'].append(bbox) 
            except:
                print("Exception u fji DeepLearningPoseEstimator.KropujRegijuGlave() !!!")
                self.rezPoses_crop_regions['ruka_leva'].append(None)

    # Krop desna ruka
    def KropujRegijuDesneRuke(self, DebugMode=False):
        img = self.rezImage.copy()
        id = 0
        for iPerson in self.rezPoses_person_ids:
            try:
                width_glave            = self.rezPoses_crop_regions['glava'][iPerson].Width
                width                  = width_glave  * 0.7
                height                 = width_glave * 0.7
                right_wrist            = self.getKeypointZaIdCoveka(iPerson, 'right_wrist')
                right_elbow            = self.getKeypointZaIdCoveka(iPerson, 'right_elbow')
                vPravcaPodlaktice      = right_wrist[0:2]-right_elbow[0:2]                
                center                 = right_wrist[0:2] + vPravcaPodlaktice * 0.4
                gornji_ugao            = center - (width,height)
                donji_ugao             = center + (width,height)
                 
                imgCrop                = img[ np.int(gornji_ugao[0]) : np.int(donji_ugao[0]), np.int(gornji_ugao[1]):np.int(donji_ugao[1]), :]
                if DebugMode:
                    id = id + 1
                    cv2.imwrite('ars_DesnaRuka' + str(id) + '.jpg', imgCrop) 
                
                bbox = BoundingBox(img=imgCrop, Center=center, Width=width, Height=height)
                if min(imgCrop.shape) > 2:
                    self.rezPoses_crop_regions['ruka_desna'].append(bbox) 
            except:
                print("Exception u fji DeepLearningPoseEstimator.KropujRegijuGlave() !!!")
                self.rezPoses_crop_regions['ruka_desna'].append(None)               

    # Krop desna ruka
    def KropujRegijuDesneNoge(self, DebugMode=False):
        img = self.rezImage.copy()
        id = 0
        for iPerson in self.rezPoses_person_ids:
            try:
                width_glave            = self.rezPoses_crop_regions['glava'][iPerson].Width
                width                  = width_glave * 0.7
                height                 = width_glave * 0.7

                right_ankle            = self.getKeypointZaIdCoveka(iPerson, 'right_ankle')
                  
                center                 = right_ankle[0:2] + (height*0.6, 0) 
                
                gornji_ugao            = center - (width,height) 
                donji_ugao             = center + (width,height) 
                 
                imgCrop                = img[ np.int(gornji_ugao[0]) : np.int(donji_ugao[0]), np.int(gornji_ugao[1]):np.int(donji_ugao[1]), :]
                if DebugMode:
                    id = id + 1
                    cv2.imwrite('ars_DesnaRuka' + str(id) + '.jpg', imgCrop) 
                
                bbox = BoundingBox(img=imgCrop, Center=center, Width=width, Height=height)
                if min(imgCrop.shape) > 2:
                    self.rezPoses_crop_regions['noga_desna'].append(bbox) 
            except:
                print("Exception u fji DeepLearningPoseEstimator.KropujRegijuDesneNoge() !!!")
                self.rezPoses_crop_regions['noga_desna'].append(None)

    # Krop desna ruka
    def KropujRegijuLeveNoge(self, DebugMode=False):
        img = self.rezImage.copy()
        id = 0
        for iPerson in self.rezPoses_person_ids:
            try:
                width_glave            = self.rezPoses_crop_regions['glava'][iPerson].Width
                width                  = width_glave * 0.8
                height                 = width_glave * 0.8

                left_ankle            = self.getKeypointZaIdCoveka(iPerson, 'left_ankle')
                  
                center                 = left_ankle[0:2] + (height*0.6, 0) 

                gornji_ugao            = center - (width,height) 
                donji_ugao             = center + (width,height) 
                 
                imgCrop                = img[ np.int(gornji_ugao[0]) : np.int(donji_ugao[0]), np.int(gornji_ugao[1]):np.int(donji_ugao[1]), :]
                if DebugMode:
                    id = id + 1
                    cv2.imwrite('ars_DesnaRuka' + str(id) + '.jpg', imgCrop) 
                
                bbox = BoundingBox(img=imgCrop, Center=center, Width=width, Height=height)
                if min(imgCrop.shape) > 2:
                    self.rezPoses_crop_regions['noga_leva'].append(bbox) 
            except:
                print("Exception u fji DeepLearningPoseEstimator.KropujRegijuLeveNoge() !!!")
                self.rezPoses_crop_regions['noga_leva'].append(None)

    # Krop celo telo
    def KropujRegijuCelogTela(self, DebugMode=False):
        img = self.rezImage.copy()
        id = 0
        for iPerson in self.rezPoses_person_ids:
            try:
                keyPoints = self.rezPoses_pts[iPerson]

                gornji_ugao            = (np.min(keyPoints[:,0]), np.min(keyPoints[:,1])) 
                donji_ugao             = (np.max(keyPoints[:,0]), np.max(keyPoints[:,1])) 

                center = np.add(gornji_ugao, donji_ugao)/2
                width  = 1.2*(donji_ugao[0] - gornji_ugao[0])/2
                height = 1.5*(donji_ugao[1] - gornji_ugao[1])/2
                 
                imgCrop                = img[ np.int(gornji_ugao[0]) : np.int(donji_ugao[0]), np.int(gornji_ugao[1]):np.int(donji_ugao[1]), :]
                if DebugMode:
                    id = id + 1
                    cv2.imwrite('ars_DesnaRuka' + str(id) + '.jpg', imgCrop) 
                
                bbox = BoundingBox(img=imgCrop, Center=center, Width=width, Height=height)
                if min(imgCrop.shape) > 2:
                    self.rezPoses_crop_regions['celo_telo'].append(bbox) 
            except:
                print("Exception u fji DeepLearningPoseEstimator.KropujRegijuCelogTela() !!!")
                self.rezPoses_crop_regions['noga_leva'].append(None)


    # Iscrtaj keypoints preko ulazne slike
    def iscrtajSekletonePrekoImg(self, image=None, DebugMode=False):
        if image == None:
            image = self.rezImage.copy()

        pts        = self.rezPoses_pts
        person_ids = self.rezPoses_person_ids

        for i, (pt, pid) in enumerate(zip(pts, person_ids)):
            image = draw_points_and_skeleton(image, pt, joints_dict()['coco']['skeleton'], person_index=pid,
                                             points_color_palette='gist_rainbow', skeleton_color_palette='jet',
                                             points_palette_samples=10)
        if DebugMode:
            cv2.imwrite('ars_iscrtane_keypoints.jpg', image)
        
        return image

    # Iscrta sve kropovane BBox preko slike
    def iscrtajBoundingBoxovePrekoImg(self, image=None, DebugMode=False, ispisiLabeleBoxaNaSlici=True):
        if image == None:
            image = self.rezImage.copy()

        max_width = 0
        for deo_tela in self.rezPoses_crop_regions:
            for bbox in self.rezPoses_crop_regions[deo_tela]:
                try:
                    max_width = max(max_width,bbox.Width)
                except:
                    max_width = max_width

        for deo_tela in self.rezPoses_crop_regions:
            for bbox in self.rezPoses_crop_regions[deo_tela]:
                if bbox != None:
                    image = bbox.iscrtajBoundingBoxNaSlici(image)
                    if ispisiLabeleBoxaNaSlici:
                        image = bbox.ispisiLabeleBoxaNaSlici(image, max_width)

        if DebugMode:
            cv2.imwrite('ars_BBOX.jpg', image)
        
        return image