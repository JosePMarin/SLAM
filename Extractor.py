# pylint: disable=no-member
import numpy as np
import cv2

from skimage.measure import ransac
from skimage.transform import FundamentalMatrixTransform

class Extractor():


    def __init__(self):
        self.orb = cv2.ORB_create(100)
        #Brute-Force matcher: matches entre los descriptors(vectores) de los keypoints de cada feature, usando la distancia de hamming
        self.bf=cv2.BFMatcher(cv2.NORM_HAMMING)

        self.last= None
        
        

    #Funcion que extrae los keypoints y sus descriptors para cada imagen
    def extract(self,img):
        
        #features(diferencia de escalas en los pixels): DETECTA las zonas donde se localizaran los keypoints 
        feats=cv2.goodFeaturesToTrack(np.mean(img, axis=2).astype(np.uint8),3000, qualityLevel=0.01, minDistance=3)
        
        #EXTRAE los keypoints de cada feature (recorridas por un for)
                    
        kps = [cv2.KeyPoint(x=f[0][0], y=f[0][1], _size=20) for f in feats] 
        
        #Se analizan los keypoints de cada feature y se calcula su vector (descriptor)
        kps, des=self.orb.compute(img,kps)

        #Se hacen los MATCHES entre los descriptors de los features
        ret  = []
        if self.last is not None:
            
            matches=self.bf.knnMatch(des, self.last["des"],k=2)
           
            for m,n in matches:
                if m.distance < 0.75*n.distance:
                                      
                    kp1=kps[m.queryIdx]
                    kp2=self.last["kps"][m.trainIdx]
                    ret.append((kp1,kp2))
        
           
        #Filter
    

        

        self.last={"kps":kps, "des":des}
        return ret