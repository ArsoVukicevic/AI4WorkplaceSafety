# Testiranje pose estimatora

import os as os
import cv2
import torch as torch
from DeepLearningPoseEstimator import DeepLearningPoseEstimator

def pokupiObradiSlikeUnutarFoldera(folderPath):
    rezPutanjeSlika = []
    for r,d,f in os.walk(folderPath):
     for files in f:
           if files[-3:].lower()=='jpg' or files[-4:].lower() =="jpeg" or files[-3:].lower() =="png":
                #print (os.path.join(r,files) )
                rezPutanjeSlika.append(os.path.join(r,files))
    return rezPutanjeSlika

def obrisiSlikeIzFoldera(folderPath):
    imgs = pokupiObradiSlikeUnutarFoldera(folderPath)
    for img in imgs:
        os.remove(img)

imgpaths = ['test_img_1.jpg','test_img_2.jpg','test_img_3.jpg','test_img_4.jpg'] 
imgpaths = pokupiObradiSlikeUnutarFoldera( os.path.join('ars_tmp_input_folder', '1 Glava') )

DL_ModelPose   = [os.path.join('03 Trenirani_DeepLearning_modeli','0 ModelPose','pose_hrnet_w32_384x288.pth'), 
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
                    ]
device           = torch.device('cuda:0')

poseEstimator = DeepLearningPoseEstimator(Path = DL_ModelPose[0], Annotations = DL_ModelPose[2],  Arhitektura = DL_ModelPose[1],  device = device)

outputFolderPath = 'ars_tmp_output_rez'
obrisiSlikeIzFoldera(outputFolderPath)

for imgpath in imgpaths:
    poseEstimator.resetCollections()
    pts   = poseEstimator.inference(img=imgpath, DebugMode=False)
    image = poseEstimator.iscrtajSekletonePrekoImg()


    save_path = os.path.join(outputFolderPath,
                            os.path.splitext(os.path.basename(imgpath))[0] + '.jpg')

    cv2.imwrite(save_path, image)