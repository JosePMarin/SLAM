# pylint: disable=no-member

import cv2
import pygame
import numpy as np

W=1920//2
H=1080//2





class FeatureExtractor():


    def __init__(self):
        self.orb = cv2.ORB_create(10000)

    #Funcion que extrae los keypoints y sus descriptors para cada imagen
    def extract(self,img):
        #features(diferencia de escalas en los pixels): detecta las zonas donde se localizaran los keypoints 
        feats=cv2.goodFeaturesToTrack(np.mean(img, axis=2).astype(np.uint8),3000, qualityLevel=0.01, minDistance=3)
        kps = [cv2.KeyPoint(x=f[0][0], y=f[0][1], _size=20) for f in feats] #Se extraen los keypoints de cada feature (recorridas por un for)
        des=self.orb.compute(img,kps) #Se analizan los keypoints de cada feature y se calcula su vector (descriptor)
       
        return kps,des 
               

fe=FeatureExtractor()


def process_frame(img):
    
    img=cv2.resize(img,(W,H))
    kps, des=fe.extract(img)

    for p in kps:
        u, v=map(lambda x: int(round(x)),p.pt) #aproximacion numerica y mapeo de las coordenadas de los keypoints
        
        cv2.circle(img,(u,v), color=(0,255,0), radius=3)#Dibuja un circulo verde por cada keypoint
        

    
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

