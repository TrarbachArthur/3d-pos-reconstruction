import cv2
import numpy as np
from cv2 import aruco
import matplotlib.pyplot as plt
import sys
import matplotlib
matplotlib.use("TkAgg")

def process_aruco(file_name):
    parameters =  aruco.DetectorParameters()
    dictionary = aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    arucoDetector = aruco.ArucoDetector(dictionary, parameters)
    vid = cv2.VideoCapture(file_name)

    aruco_map = [] # Lista de posições do aruco por frame do vídeo

    while True:
        _, img = vid.read()
        if img is None:
            print("Empty Frame")
            break

        corners, ids, rejectedImgPoints = arucoDetector.detectMarkers(img)
        frame_markers = aruco.drawDetectedMarkers(img.copy(), corners, ids)
        cv2.imshow('output', frame_markers)
        
        if cv2.waitKey(1) == ord('q'):
            break
        
        if ids is not None and 0 in ids: # Se encontrou Aruco, checar se é o id 0
            idx = np.where(ids==0)[0][0]
            x,y = np.mean(corners[idx][0], axis=0).astype(int) # Calcula centro do Aruco
            aruco_map.append(np.array([[x,y,1]]))
        else:
            aruco_map.append(None) # None quando o aruco não é encontrado.

    # cv2.destroyAllWindows() # Fecha a janela, mas faz a proxima aparecer em lugar aleatorio da tela

    return aruco_map
