# Skripta vrsi testiranje klasifikatora - slike su vec kropovane

import os
import cv2
import torch

from DeepLearningClassificator import DeepLearningClassificator

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

DL_ClasifikatorGlava = [ 
                        [os.path.join('03 Trenirani_DeepLearning_modeli','1 Glava'    ,'Hamlet_model[MobileNetV2]_acc[0.9862068965517241]_epochs[10]_batch[4]_.pth'), 
                                      'MobileNetV2', 
                                       ['No Hamlet',      'Hamlet'      ]
                        ],
                        [os.path.join('03 Trenirani_DeepLearning_modeli','1 Glava'    ,'CovidMask_model[MobileNetV2]_acc[0.9839080459770115]_epochs[20]_batch[8]_.pth'), 
                                       'MobileNetV2', 
                                       ['No Covid mask',  'Covid mask'  ]
                        ]
                       ]
DL_ClasifikatorTrup = [
                        [os.path.join('03 Trenirani_DeepLearning_modeli','2 Trup'    ,'YellowVests_model[MobileNetV2]_acc[0.9345794392523364]_epochs[10]_batch[16]_.pth'), 
                                       'MobileNetV2', 
                                       ['No Yellow Vest', 'Yellow Vest' ]
                        ]
                      ]
DL_ClasifikatorShake = [
                        [os.path.join('03 Trenirani_DeepLearning_modeli','3 Shake'   ,'Gloves_model[MobileNetV2]_acc[0.9319196428571429]_epochs[10]_batch[16]_.pth'), 
                                       'MobileNetV2', 
                                       ['No Gloves', 'Covid gloves', 'Industry gloves']
                        ]
                       ]

device = torch.device('cuda:0')

# DICT klasifikatora i delovi tela za koji su vezani
Klasifikatori = {
                  'Glava' : DL_ClasifikatorGlava,
                  'Trup'  : DL_ClasifikatorTrup,
                  'Ruke'  : DL_ClasifikatorShake,
                  'Noge'  : None,
                  'Telo'  : None,
                }

# Lista inpt foldera za razlicite slike
inputTestSlike = {
                  'Glava' : os.path.join('ars_tmp_input_folder', '1 Glava'    ),
                  'Trup'  : os.path.join('ars_tmp_input_folder', '2 Trup'     ),
                  'Ruke'  : os.path.join('ars_tmp_input_folder', '3 Ruke'     ),
                  'Noge'  : os.path.join('ars_tmp_input_folder', '4 Noge'     ),
                  'Telo'  : os.path.join('ars_tmp_input_folder', '5 Celo Telo'),
                 }


###############################################################################################################################
DeoTelaKojiSeTestira = 'Glava'     ###     JEDINI PARAMETAR KOJI SE SETUJE   ###                                           ####
###############################################################################################################################

inputFiles       = pokupiObradiSlikeUnutarFoldera(inputTestSlike[ DeoTelaKojiSeTestira ])
DL_Klasifikatori = Klasifikatori[DeoTelaKojiSeTestira]
outputFolderPath = 'ars_tmp_output_rez'
obrisiSlikeIzFoldera(outputFolderPath)

for iKlasifikator in DL_Klasifikatori:     
    iKlasifikator = DeepLearningClassificator(Path = iKlasifikator[0],  Arhitektura = iKlasifikator[1],  Annotations = iKlasifikator[2],  device = device)
    for iSlika in inputFiles:
        pred_class, pred_label = iKlasifikator.inference(iSlika)
        img = cv2.imread(iSlika)
        save_path = os.path.join(outputFolderPath,
                                 os.path.splitext(os.path.basename(iSlika))[0] +
                                  '_' + pred_label + '_' + '.jpg')
        cv2.imwrite( save_path, img)

        


                                        
                                        