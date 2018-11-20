# pylint: disable=no-member
import time
import cv2
import numpy as np
from Extractor import Extractor



W=1920//2
H=1080//2

F=1
             



extrac=Extractor(F,H,W)

class Process():

       
                     
        def process_frame(self,img):
                
                
                self.img=cv2.resize(img,(W,H))

                #matches es un array-2D con los puntos normalizados y filtrados
                matches=extrac.extract(self.img)

                print("%d matches" %(len(matches)))

                for pt1, pt2 in matches:

                        #Se desnormalizan las coordenadas para poder mostrarse
                        u1, v1=extrac.denormalize(pt1) 
                        u2, v2=extrac.denormalize(pt2)
                        
                        

                        #Dibuja un circulo verde por cada keypoint
                        cv2.circle(self.img,(u1,v1), color=(0,255,0), radius=3)
                        cv2.circle(self.img,(u2,v2), color=(0,0,255), radius=3)

                        #Dibuja una linea entre matches
                        cv2.line(self.img,(u1,v1), (u2, v2), color=(255,0,0))
                
                cv2.imshow('Image',self.img)
        

