# pylint: disable=no-member
import numpy as np
import cv2

class Extractor():


    def __init__(self):
        self.orb = cv2.ORB_create(10000)
        self.last= None
        #Brute-Force matcher: matches entre los descriptors(vectores) de los keypoints de cada feature
        self.bf=cv2.BFMatcher()

    #Funcion que extrae los keypoints y sus descriptors para cada imagen
    def extract(self,img):
        
        #features(diferencia de escalas en los pixels): detecta las zonas donde se localizaran los keypoints 
        feats=cv2.goodFeaturesToTrack(np.mean(img, axis=2).astype(np.uint8),3000, qualityLevel=0.01, minDistance=3)
        
        #Se extraen los keypoints de cada feature (recorridas por un for)
        kps = [cv2.KeyPoint(x=f[0][0], y=f[0][1], _size=20) for f in feats] 
        
         #Se analizan los keypoints de cada feature y se calcula su vector (descriptor)
        kps, des=self.orb.compute(img,kps)

        
        matches=None
        if self.last is not None:
            matches=self.bf.match(des, self.last["des"])
            
        
        self.last={"kps":kps, "des":des}
        
        return kps,des, matches