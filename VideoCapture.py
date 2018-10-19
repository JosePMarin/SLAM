# pylint: disable=no-member

import cv2
import pygame

W=1920//2
H=1080//2





class FeatureExtractor():
    GX=16//2 #1920/2=960 -> 960/16=60
    GY=12//2 #1080/2=540 -> 540/12=45

    def __init__(self):
        self.orb = cv2.ORB_create(10000)

    def extract(self,img):
        #Hacemos cuadrillas con la imagen, con seccion sy*sx
        sy=img.shape[0]//self.GY
        sx=img.shape[1]//self.GX
        akp=[]
        #Bucle para recorrer toda la cuadrilla
        for ry in range(0,img.shape[0],sy):
            for rx in range(0,img.shape[1],sx):
                img_chunk=img[ry:ry+sy, rx:rx+sx] #Se hace un chunk que se va incrementando diagonalmente conforme avanza el bucle
                
                kp = self.orb.detect(img_chunk, None) 
                
                for p in kp:
                    
                    p.pt=(p.pt[0] + rx, p.pt[1] + ry) 
                    akp.append(p)  
        return akp                 

fe=FeatureExtractor()


def process_frame(img):
    
    img=cv2.resize(img,(W,H))
    
    kp=fe.extract(img)

    for p in kp:
        u, v=map(lambda x: int(round(x)),p.pt)
        cv2.circle(img,(u,v), color=(0,255,0), radius=3)
        

    
    cv2.imshow("image",img)
"""cv2.waitKey()
print(img.shape)"""


if __name__=="__main__":
    cap=cv2.VideoCapture("./videos/test.mp4")

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

