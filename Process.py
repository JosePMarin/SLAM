# pylint: disable=no-member
import time
import cv2
import numpy as np
from Extractor import Extractor



W=1920//2
H=1080//2
             



extrac=Extractor()

        
def process_frame(img):
        
        
        img=cv2.resize(img,(W,H))
        matches=extrac.extract(img)

        #print("%d matches" %(len(matches)))

        for pt1, pt2 in matches:

                #aproximacion numerica y mapeo de las coordenadas de los keypoints
                u1, v1=map(lambda x: int(round (x)), pt1) 
                u2, v2=map(lambda x: int(round (x)), pt2)

                #Dibuja un circulo verde por cada keypoint
                cv2.circle(img,(u1,v1), color=(0,255,0), radius=3)
                cv2.circle(img,(u2,v2), color=(0,0,255), radius=3)

                #Dibuja una linea entre matches
                cv2.line(img,(u1,v1), (u2, v2), color=(255,0,0))
        
        cv2.imshow('Image',img)
       

