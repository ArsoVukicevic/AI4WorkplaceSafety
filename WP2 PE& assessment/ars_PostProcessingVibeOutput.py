import joblib
import pickle
import zipp 
import vtk
from vtk import *
import numpy as np
import pyvista as pv
import os
import xlsxwriter


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Pogledati https://github.com/ReillyBova/Point-Cloud-Registration  MIT Licence
# https://docs.pyvista.org/examples/00-load/create-tri-surface.html#sphx-glr-examples-00-load-create-tri-surface-py
#INPUTS
    # Points      - np.array(brTacki, 3)     - izlaz iz VIBE algoritma
    # SIMPL_faces - np.array(brTrouglova, 3) - izlaz iz VIBE algoritma
    # filename    - naziv vtk fajla
def upisiVTK(Points, Faces=None, iscrtaj=False, filename="TriangleVerts.vtp"):
    brCvorova = len(Points[:,1])
    cvorovi   = np.ones((brCvorova,3))
    for i in range(brCvorova):
        cvorovi[i,:] = (Points[i,0], Points[i,2], -Points[i,1]) # Korekcija VIBE formata u XYZ
    if Faces is None:
        # Upisi samo cvorove
        surf  = pv.PolyData(cvorovi)
    else:
        brFejsova = len(Faces[:,1])
        faces     = np.ones((brFejsova,4))
        for i in range(brFejsova):
            faces[i,:] = (3, Faces[i,0], Faces[i,1], Faces[i,2])
        faces = np.hstack(faces).astype(np.int16)
        surf  = pv.PolyData(cvorovi, faces)
    surf.save(filename, binary=False)
    if iscrtaj==True:
        surf.plot(show_edges=True)

# Fja racuna ugao izmedju 2 vektora
    #Inputs
        #vector_1, vector_2 - numpy vektori x y z
        #NullOsa            - osa koju treba nulirati
def ugaoIzmedjuVektora(vector_1, vector_2, NullOsa=None):
    if NullOsa is not None:
        vector_1[NullOsa] = 0
        vector_2[NullOsa] = 0
    unit_vector_1 = vector_1 / np.linalg.norm(vector_1)
    unit_vector_2 = vector_2 / np.linalg.norm(vector_2)
    dot_product = np.dot(unit_vector_1, unit_vector_2)
    angle = np.arccos(dot_product)
    return np.rad2deg(angle)

    

"""  VIBE OUTPUT DICTIONARY
output_dict = {
        'pred_cam': pred_cam,
        'orig_cam': orig_cam,
        'verts': pred_verts,
        'pose': pred_pose,
        'betas': pred_betas,
        'joints3d': pred_joints3d,
        'joints2d': joints2d,
        'bboxes': bboxes,
        'frame_ids': frames,
        }
"""

# Podaci po frejmovima
resultsPath = "F:\\_ARSO_DeepLearningCode\\_BazaArhitektura\\PoseEstimation\\VIBE MaxPlank (3D shape od keypoints 2D iz Single view)\\output\\IMG_2510.mov"
resultsPath = "F:\\_ARSO_DeepLearningCode\\_BazaArhitektura\\PoseEstimation\\VIBE MaxPlank (3D shape od keypoints 2D iz Single view)\\output\\IMG_2606_DzoniNoga2.mov"
resultsPath = "F:\\_ARSO_DeepLearningCode\\_BazaArhitektura\\PoseEstimation\\VIBE MaxPlank (3D shape od keypoints 2D iz Single view)\\output\\\IMG_8304"
dict_pkl    = joblib.load(os.path.join(resultsPath,"vibe_output.pkl"))
# SMPL face id
SMPL_faces  = joblib.load(os.path.join(resultsPath,"vibe_faces.pkl"))


osoba1     = dict_pkl[1] #  0335
result = {}
for d, id in zip(dict_pkl,range(len(dict_pkl))):
    osoba1 = dict_pkl[d]
    if id == 0:
        result = osoba1    
    else:
        for pred_cam, orig_cam, verts, pose, betas, joints3d, bboxes, frame_ids in zip(osoba1['pred_cam'], osoba1['orig_cam'], osoba1['verts'], osoba1['pose'], osoba1['betas'], osoba1['joints3d'],  osoba1['bboxes'], osoba1['frame_ids'] ):
            result['pred_cam'] = np.append(result['pred_cam'], [pred_cam], axis=0)
            result['orig_cam'] = np.append(result['orig_cam'], [orig_cam], axis=0)
            result['verts'] = np.append(result['verts'], [verts], axis=0)
            result['pose'] = np.append(result['pose'], [pose], axis=0)
            result['betas'] = np.append(result['betas'], [betas], axis=0)
            result['joints3d'] = np.append(result['joints3d'], [joints3d], axis=0)
            result['bboxes'] = np.append(result['bboxes'], [bboxes], axis=0)
            result['frame_ids'] = np.append(result['frame_ids'], [frame_ids], axis=0)

osoba1 = result
brFrejmova = len(osoba1['bboxes'])

# Triangulacija point cloud 
# https://github.com/marcomusy/vtkplotter/blob/85e2e2ba9a6e40d001ac4145c2787aab8b246fe1/vtkplotter/analysis.py#L1212
# https://www.programcreek.com/python/example/11293/vtk.vtkPolyData 
# https://docs.pyvista.org/examples/index.html
# Posto se tracking radio preko bbox sada nema osoba1['joints2d']
resultsPath = os.path.join(resultsPath,"vtk") 
#if os.path.exists(resultsPath):
#    os.rmdir(resultsPath)
#os.mkdir(resultsPath) 
ugaoKicmaVertikala             = []
UgaoKicmaDonjiSrednjiGornjiDeo = [] 
ugaoTorzijaKukoviRamena        = []
visinaTezistaKukova            = []
visinaTezistaRamena            = []
#
ugaoDesno_ClanakKolenoKuk      = []
ugaoLevo_ClanakKolenoKuk       = []
ugaoDesno_PotkolenicaVertikala = []
ugaoLevo_PotkolenicaVertikala  = []
#
ugaoDesno_SakaLakatRame        = []
ugaoDesno_Podlaktica           = []
ugaoLevo_SakaLakatRame         = []
ugaoLevo_Podlaktica            = []
 

#
for pred_cam, orig_cam, verts, pose, betas, joints3d, bboxes, frame_ids in zip(osoba1['pred_cam'], osoba1['orig_cam'], osoba1['verts'], osoba1['pose'], osoba1['betas'], osoba1['joints3d'],  osoba1['bboxes'], osoba1['frame_ids'] ):
    upisiVTK(joints3d , Faces=None, iscrtaj=False, filename=os.path.join(resultsPath,"Joints3D_" + str(frame_ids).zfill(4) + ".vtk"    ))
    upisiVTK(verts    , SMPL_faces, iscrtaj=False, filename=os.path.join(resultsPath,"Pose3D_"   + str(frame_ids).zfill(4) + ".vtk"    ))

    # joints3d   - cvorovi kicme su 39 41 40  
    #            - desna ruka 31 32 33 // prvi je saka, drugi je lakat, treci rame
    #            - leva ruka  36 35 34    
    #            - desna noga 25 26 27 // claank, koleno, kuk
    #            - leva noga  30 29 28
    #            - desni i levi kuk 27 28
    #            - desno levo rame 33 34
    # Kicma
    ugaoKicmaVertikala.append(ugaoIzmedjuVektora([0,0,-1], (joints3d[40][np.array([0,2,1])]-joints3d[39][np.array([0,2,1])]))) # org format je: (Points[i,0], Points[i,2], -Points[i,1])
    UgaoKicmaDonjiSrednjiGornjiDeo.append(ugaoIzmedjuVektora(joints3d[40]-joints3d[41], joints3d[39]-joints3d[41]))
    ugaoTorzijaKukoviRamena.append(ugaoIzmedjuVektora(joints3d[28]-joints3d[27], joints3d[34]-joints3d[33], NullOsa=1)) #1=Zosa
    visinaTezistaKukova.append(-joints3d[39][1]) # U VIBE-u ose su zaokrenute 
    visinaTezistaRamena.append(-joints3d[40][1])
    # Ruke
    ugaoDesno_SakaLakatRame.append(ugaoIzmedjuVektora(joints3d[31]-joints3d[32], joints3d[33]-joints3d[32]))
    ugaoDesno_Podlaktica.append(ugaoIzmedjuVektora(joints3d[27]-joints3d[33], joints3d[32]-joints3d[33]))
    ugaoLevo_SakaLakatRame.append(ugaoIzmedjuVektora(joints3d[36]-joints3d[35], joints3d[34]-joints3d[35]))
    ugaoLevo_Podlaktica.append(ugaoIzmedjuVektora(joints3d[28]-joints3d[34], joints3d[35]-joints3d[34]))
    # Noge
    ugaoDesno_PotkolenicaVertikala.append(ugaoIzmedjuVektora([0,0,-1], (joints3d[26][np.array([0,2,1])]-joints3d[25][np.array([0,2,1])]))) 
    ugaoLevo_PotkolenicaVertikala.append(ugaoIzmedjuVektora([0,0,-1], (joints3d[29][np.array([0,2,1])]-joints3d[30][np.array([0,2,1])]))) 
    ugaoDesno_ClanakKolenoKuk.append(ugaoIzmedjuVektora(joints3d[25]-joints3d[26], joints3d[27]-joints3d[26]))
    ugaoLevo_ClanakKolenoKuk.append(ugaoIzmedjuVektora(joints3d[30]-joints3d[29], joints3d[28]-joints3d[29]))

"""

workbook = xlsxwriter.Workbook('Dijagrami.xlsx') 
worksheet = workbook.add_worksheet() 
  
# Start from the first cell. 
# Rows and columns are zero indexed. 
row    = 0
column = 0
  
content = ["Br Frejma", 
                "Kicma-Vertikala", "Kicma-Donji Srednji Gornji deo", "Kicma-Torzija", "Kicma-Visina donji deo", "Kicma-Visina godnji deo", 
                "Desno Saka-Lakat-Rame","Desno Podlaktica", "Levo Saka-Lakat-Rame", "Levo Podlaktica", 
                "Desno Potkolenica-Vertikala", "Levo Potkolenica-Vertikala", "Desno Clanak-Koleno-Kuk", "Levo Clanak-Koleno-Kuk"] 
for iKolone, item in zip(range(len(content)), content):
    worksheet.write(0, iKolone, item)  
# iterating through content list 
for iReda, u1, u2, u3, u4, u5, u6, u7, u8, u9, u10, u11, u12, u13 in zip(range(1, len(ugaoLevo_SakaLakatRame)), 
                                            ugaoKicmaVertikala, UgaoKicmaDonjiSrednjiGornjiDeo, ugaoTorzijaKukoviRamena, visinaTezistaKukova, visinaTezistaRamena,
                                            ugaoDesno_SakaLakatRame, ugaoDesno_Podlaktica, ugaoLevo_SakaLakatRame, ugaoLevo_Podlaktica,
                                            ugaoDesno_PotkolenicaVertikala, ugaoLevo_PotkolenicaVertikala, ugaoDesno_ClanakKolenoKuk, ugaoLevo_ClanakKolenoKuk): 
    # write operation perform 
    worksheet.write(iReda, 0, iReda)
    worksheet.write(iReda, 1,  u1 )
    worksheet.write(iReda, 2,  u2 )
    worksheet.write(iReda, 3,  u3 )
    worksheet.write(iReda, 4,  u4 )
    worksheet.write(iReda, 5,  u5 )
    worksheet.write(iReda, 6,  u6 )
    worksheet.write(iReda, 7,  u7 )
    worksheet.write(iReda, 8,  u8 )
    worksheet.write(iReda, 9,  u9 )
    worksheet.write(iReda, 10, u10)
    worksheet.write(iReda, 11, u11) 
    worksheet.write(iReda, 11, u12) 
    worksheet.write(iReda, 11, u13)       
workbook.close()
"""

import matplotlib.pyplot as plt
iReda = range(0, len(ugaoLevo_SakaLakatRame))
# Kicma
plt.subplot(511)
plt.ylabel('Ugao sa V osom')
plt.plot(np.array(iReda), np.array(ugaoKicmaVertikala))
plt.subplot(512)
plt.ylabel('Ugao kicmenog stuba')
plt.plot(np.array(iReda), np.array(UgaoKicmaDonjiSrednjiGornjiDeo))
plt.subplot(513)
plt.ylabel('Ugao torzija')
plt.plot(np.array(iReda), np.array(ugaoTorzijaKukoviRamena))
plt.subplot(514)
plt.ylabel('Visina kukova')
plt.plot(np.array(iReda), np.array(visinaTezistaKukova))
plt.subplot(515)
plt.ylabel('Visina ramena')
plt.plot(np.array(iReda), np.array(visinaTezistaRamena))
plt.show()
# Ruke
plt.subplot(411)
plt.ylabel('Ugao desni lakat')
plt.plot(np.array(iReda), np.array(ugaoDesno_SakaLakatRame))
plt.subplot(412)
plt.ylabel('Ugao desna podlaktica')
plt.plot(np.array(iReda), np.array(ugaoDesno_Podlaktica))
plt.subplot(413)
plt.ylabel('Ugao levi lakat')
plt.plot(np.array(iReda), np.array(ugaoLevo_SakaLakatRame))
plt.subplot(414)
plt.ylabel('Ugao desna podlaktica')
plt.plot(np.array(iReda), np.array(ugaoLevo_Podlaktica))
plt.show()
# Noge
plt.subplot(411)
plt.ylabel('Ugao leva potkolenica vertikala')
plt.plot(np.array(iReda), np.array(ugaoDesno_PotkolenicaVertikala))
plt.subplot(412)
plt.ylabel('Ugao desna potkolenica vertikala')
plt.plot(np.array(iReda), np.array(ugaoLevo_PotkolenicaVertikala))
plt.subplot(413)
plt.ylabel('Ugao desna noga calanak-koleno-kuk')
plt.plot(np.array(iReda), np.array(ugaoDesno_ClanakKolenoKuk))
plt.subplot(414)
plt.ylabel('Ugao leva noga calanak-koleno-kuk')
plt.plot(np.array(iReda), np.array(ugaoLevo_ClanakKolenoKuk))
plt.show()