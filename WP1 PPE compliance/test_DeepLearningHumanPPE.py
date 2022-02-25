import os as os
from typing import List
import torch as torch
import cv2

from DeepLearningHumanPPE import HumanPPE

def pokupiSlikeUnutarFoldera(folderPath):
    rezPutanjeSlika = []
    for r,d,f in os.walk(folderPath):
     for files in f:
           if files[-3:].lower()=='jpg' or files[-4:].lower() =="jpeg" or files[-3:].lower() =="png":
                #print (os.path.join(r,files) )
                rezPutanjeSlika.append(os.path.join(r,files))
    return rezPutanjeSlika

def pokupiKlipoveUnutarFoldera(folderPath):
    rezPutanjeKlipova = []
    for r,d,f in os.walk(folderPath):
        for files in f:
            if files[-3:].lower()=='avi' or files[-3:].lower() =="mp4":
                rezPutanjeKlipova.append(os.path.join(r,files))
    return rezPutanjeKlipova

def obrisiSlikeIzFoldera(folderPath):
    imgs = pokupiSlikeUnutarFoldera(folderPath)
    for img in imgs:
        os.remove(img)

def obrisiFajloveIzFoldera(folderPath):
    for r,d,f in os.walk(folderPath):
        for files in f:
            os.remove(os.path.join(r,files))

imgpaths = ['test_img_1.jpg','test_img_2.jpg','test_img_3.jpg','test_img_4.jpg','test_img_5.jpg',
            'test_img_6.jpg','test_img_7.jpg','test_img_8.jpg','test_img_9.jpg','test_img_10.jpg','test_img_11.jpg','test_img_12.jpg'] 

imgpaths = pokupiSlikeUnutarFoldera( os.path.join('ars_tmp_input_folder', '6 Slike za rad') ) #'5 Celo Telo' # '6 Slike za rad'

DL_HumanPPE = HumanPPE( DL_PoseEstimator = [os.path.join('03_Trenirani_DeepLearning_modeli','0_ModelPose','pose_hrnet_w32_384x288.pth'), 
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
                                            [os.path.join('03_Trenirani_DeepLearning_modeli','1_Glava','Navlaka','model[MobileNetV2]_acc[0.9983429991714996]_epochs[10]_batch[40]_.pth'), 
                                            'MobileNetV2', 
                                            ['No Hair cap', 'Hair cap']
                                            ],
                                            [os.path.join('03_Trenirani_DeepLearning_modeli','1_Glava','Slem',   'model[MobileNetV2]_acc[0.9975144987572494]_epochs[20]_batch[40]_.pth'), 
                                            'MobileNetV2', 
                                            ['No Hardhat', 'Hardhat']
                                            ],

                                            
                                            [os.path.join('03_Trenirani_DeepLearning_modeli','1_Glava','SafetyGlasses',   'model[MobileNetV2]_acc[0.9727095516569201]_epochs[10]_batch[24]_.pth'), 
                                            'MobileNetV2', 
                                            ['No Safety Glasses', 'Safety Glasses']
                                            ],

                                            
                                            [os.path.join('03_Trenirani_DeepLearning_modeli','1_Glava','Vizir',   'model[MobileNetV2]_acc[1.0]_epochs[10]_batch[32]_.pth'), 
                                            'MobileNetV2', 
                                            ['No Vizier', 'Vizier']
                                            ],
                                            [os.path.join('03_Trenirani_DeepLearning_modeli','1_Glava','Zavarivanje',   'model[MobileNetV2]_acc[1.0]_epochs[20]_batch[24]_.pth'), 
                                            'MobileNetV2', 
                                            ['No Welding mask', 'Welding mask']
                                            ],

                                            
                                            [os.path.join('03_Trenirani_DeepLearning_modeli','1_Glava','MaskCovid',   'model[MobileNetV2]_acc[1.0]_epochs[20]_batch[24]_.pth'), 
                                            'MobileNetV2', 
                                            ['No Covid Mask', 'Covid Mask']
                                            ],
                                            [os.path.join('03_Trenirani_DeepLearning_modeli','1_Glava','MaskIndustry',   'model[MobileNetV2]_acc[1.0]_epochs[20]_batch[40]_.pth'), 
                                            'MobileNetV2', 
                                            ['No Industry Mask', 'Industry Mask']
                                            ],

                                            
                                            [os.path.join('03_Trenirani_DeepLearning_modeli','1_Glava','Slusalice',   'model[MobileNetV2]_acc[1.0]_epochs[10]_batch[32]_.pth'), 
                                            'MobileNetV2', 
                                            ['No Headphones', 'Headphones']
                                            ],
                                        ],
                        DL_ClasifikatorTrup = [ 
                                            [os.path.join('03_Trenirani_DeepLearning_modeli','2_Trup','SvetleceTrake','model[MobileNetV2]_acc[0.9477977161500816]_epochs[12]_batch[24]_.pth'), 
                                            'MobileNetV2', 
                                            ['No Lith tracks', 'Lith tracks']
                                            ],
                                            [os.path.join('03_Trenirani_DeepLearning_modeli','2_Trup','ZutiPrsluci',  'model[MobileNetV2]_acc[0.9853181076672104]_epochs[12]_batch[24]_.pth'), 
                                            'MobileNetV2', 
                                            ['No Yellow vests', 'Yellow vests']
                                            ],
                                        ],
                        DL_ClasifikatorShake = [ 
                                            [os.path.join('03_Trenirani_DeepLearning_modeli','3_Ruke','CovidRukavice','covid-model[MobileNetV2]_acc[1.0]_epochs[20]_batch[40]_.pth'), 
                                            'MobileNetV2', 
                                            ['No Covid gloves', 'Covid gloves']
                                            ],
                                            [os.path.join('03_Trenirani_DeepLearning_modeli','3_Ruke','RadnickeRukavice',  'radnicke-model[MobileNetV2]_acc[0.9905437352245863]_epochs[20]_batch[40]_.pth'), 
                                            'MobileNetV2', 
                                            ['No Industry gloves', 'Industry gloves']
                                            ],
                                        ],
                        DL_ClasifikatorStopala  = [ 
                                            [os.path.join('03_Trenirani_DeepLearning_modeli','4_Stopala','Navlake','model[MobileNetV2]_acc[0.972]_epochs[20]_batch[32]_.pth'), 
                                            'MobileNetV2', 
                                            ['No Feet covers', 'Feet covers']
                                            ],
                                            [os.path.join('03_Trenirani_DeepLearning_modeli','4_Stopala','DubokeCizme',  'model[MobileNetV2]_acc[0.908]_epochs[20]_batch[12]_.pth'), 
                                            'MobileNetV2', 
                                            ['No Boots', 'Boots']
                                            ],
                                            [os.path.join('03_Trenirani_DeepLearning_modeli','4_Stopala','Sandale',  'model[MobileNetV2]_acc[0.928]_epochs[10]_batch[32]_.pth'), 
                                            'MobileNetV2', 
                                            ['No Sandals', 'Sandals']
                                            ],
                                        ],
                        DL_ClasifikatorCeloTelo = [
                                            [os.path.join('03_Trenirani_DeepLearning_modeli','5_Celo telo','RadnoOdelo'   , 'model[MobileNetV2]_acc[0.9681467181467182]_epochs[20]_batch[40]_.pth'), 
                                            'MobileNetV2', 
                                            ['No Overal', 'Overal']
                                            ],
                                            [os.path.join('03_Trenirani_DeepLearning_modeli','5_Celo telo','ZastitnoOdelo', 'model[MobileNetV2]_acc[0.9884169884169884]_epochs[20]_batch[32]_.pth'), 
                                            'MobileNetV2', 
                                            ['No Protective cloth', 'Protective cloth']
                                            ],
                                        ],
                        device           = torch.device('cuda:0')
                )     

outputFolderPath = 'ars_tmp_output_rez'
obrisiFajloveIzFoldera(outputFolderPath)

# Live

if False:
    DL_HumanPPE.inferenceLiveUSB()

# Slike
if True:  
    for imgpath in imgpaths:
        DL_HumanPPE.resetCollections()
        image     = DL_HumanPPE.inference(image=imgpath, DebugMode=True)
        save_path = os.path.join(outputFolderPath, os.path.splitext(os.path.basename(imgpath))[0] + '.jpg')

        cv2.imwrite(save_path, image)

exit()

# Video file
videopaths = pokupiKlipoveUnutarFoldera( os.path.join('ars_tmp_input_folder', 'Video') )
obrisiFajloveIzFoldera(outputFolderPath)

if False:
    for videopath in videopaths:
        DL_HumanPPE.resetCollections()
        save_path = os.path.join(outputFolderPath, os.path.splitext(os.path.basename(videopath))[0] + '.mp4')
        DL_HumanPPE.inferenceVideo(input_video=videopath, output_video=save_path) 