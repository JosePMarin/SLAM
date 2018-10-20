# pylint: disable=no-member

import cv2
import pygame
import numpy as np
from Extractor import Extractor

W=1920//2
H=1080//2
             

fe=Extractor()

def process_frame(img):
    
    img=cv2.resize(img,(W,H))
    kps, des, matches=fe.extract(img)

    for p in kps:

        #aproximacion numerica y mapeo de las coordenadas de los keypoints
        u, v=map(lambda x: int(round(x)),p.pt) 
        #Dibuja un circulo verde por cada keypoint
        cv2.circle(img,(u,v), color=(0,255,0), radius=3)
        

    
    cv2.imshow("image",img)
"""cv2.waitKey()
print(img.shape)"""

#Al ejecutar el .py, ejecuta la captura de video 
if __name__=="__main__":
    cap=cv2.VideoCapture("./videos/test.mp4")

#Se crea un bucle para ejecutar procesado de imagen mientras el video se reproduzca
while cap.isOpened():
    # Capture frame-by-frame
    ret, frame = cap.read()
    if ret == True:
        process_frame(frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    else:
        break
    
cap.release()
cv2.destroyAllWindows()        

