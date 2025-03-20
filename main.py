#
# Realizado por: 
# Arthur Trarbach e Juliana Priori
#

import numpy as np
import matplotlib.pyplot as plt
from parametros import camera_parameters
from detect_markers_video import process_aruco

np.set_printoptions(formatter={'float': lambda x: "{0:0.2f}".format(x)})

# Criando listas para armazenar [k0, k1, k2, k3] ...
K = []
R = []
T = []

# Carrega os parâmetros de cada câmera
for idx in range(4):
    k, r, t, _, _ = camera_parameters(f'./cam_data/{idx}.json')
    K.append(k)
    R.append(r)
    T.append(t)

print(K)
print(R)
print(T)

aruco_map = [] # [[pnts_0], [pnts_1], [pnts_2], [pnts_3]]
proj_Ms = [] # [proj_m0, proj_m1, proj_m2, proj_m3] -> matriz de projeção de cada câmera
for i in range(4):
    # Obtendo posição do aruco em cada câmera (por frame)
    # "./cam_data/camera-00.mp4"
    # "./cam_data/camera-01.mp4"
    # "./cam_data/camera-02.mp4"
    # "./cam_data/camera-03.mp4"
    aruco_map.append(process_aruco(f"./cam_data/camera-0{i}.mp4"))

    # Calculando matrizes de projeção
    PI = np.array([[1,0,0,0],
                    [0,1,0,0],
                    [0,0,1,0]])

    M = np.vstack((np.hstack((R[i], T[i])), np.array([[0, 0, 0, 1]])))

    proj_Ms.append(K[i] @ PI @ np.linalg.inv(M))

# Calculando coordenadas 3d
# Usando proj_m e aruco_map
# x3d = [x3d0, x3d1, ..., x3dn]
# y3d e z3d semelhantes a x3d
x3d = []
y3d = []
z3d = []

for frame in range(len(aruco_map[0])): # qtd de pontos detectados na camera 0 (qtd frames do video)
    pts_added = 0

    for camera in range(4):
        if aruco_map[camera][frame] is not None: # Se o aruco foi identificado no frame da câmera
            # Obtendo B
            # OBS: Primeira iteração da camera precisa de atenção especial

            if pts_added == 0: # Primeira iteração
                B = np.append(proj_Ms[camera], -aruco_map[camera][frame].T, axis=1)
            else:
                B = np.append(B, np.zeros((3*pts_added, 1)), axis=1) # Preenchendo 0's a direita
                # Calculando nova linha da matriz B
                new_line = np.concatenate((proj_Ms[camera], np.zeros((3, pts_added)), -aruco_map[camera][frame].T), axis=1)
                B = np.append(B, new_line, axis=0)
            
            pts_added += 1

    # SVD of B.
    U,S,Vt = np.linalg.svd(B)
    # Use the first 4 elements of the last column of matrix V as the estimation of M .
    point = Vt[-1,:][:4]
    point /= point[3] # Setting as homogenous coordinates

    x3d.append(point[0])
    y3d.append(point[1])
    z3d.append(point[2])

# Plotando pontos3d calculados.
fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x3d, y3d, z3d)
#ax.set_aspect('equal', adjustable='box') # "Normaliza" o grafico. Enquanto comentado, mantem oscilações em Z visiveis
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
ax.axes.set_zlim3d(bottom=0, top=1)

plt.show()