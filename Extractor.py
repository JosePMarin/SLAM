# pylint: disable=no-member
import numpy as np
import cv2

from skimage.measure import ransac
from skimage.transform import FundamentalMatrixTransform
from skimage.transform import EssentialMatrixTransform
class Extractor():



    def __init__(self,F,H,W):
        self.H=H
        self.W=W

        #Se inicializa ORB, el cual detecta los corners (keypoints en adelante)
        self.orb = cv2.ORB_create()
        #Brute-Force matcher: matches entre los descriptors(vectores) de los keypoints de cada feature, usando la distancia de hamming
        self.bf=cv2.BFMatcher(cv2.NORM_HAMMING)
        self.last= None
        
        self.F=F


    


     #Se DESNORMALIZAN las coordenadas para poder hacer un display de los puntos sobre la imagen   
    def denormalize(self, pt):
        #K = intrinsic matrix -> camera_resectioning (wikipedia)
        K=np.array([[self.F,0,-self.H//2],[0,self.F,-self.W//2],[0,0,1]])
        Kinv=np.linalg.inv(K)
        b=np.array([pt[0],pt[1],1])
        c=np.dot(Kinv, b)
        
        return int(round(c[0])), int(round(c[1]))          

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
            #Se utiliza una función para pillar los k=2 posibles_matches que correspondan a cada verdadero_match
            matches=self.bf.knnMatch(des, self.last["des"],k=2)
           
            for m,n in matches:
                #Recorremos todos los matches entre frames que esten cercanos
                if m.distance < 0.75*n.distance:
                    #Para cada match cercano entre frames (frame1 y frame2, consecutivo) se obtienen la indexacion de keypoints (kp1[] y kp2[]) de cada keypoint (kps)                
                    kp1=np.float32(kps[m.queryIdx].pt)
                    
                    kp2=np.float32(self.last["kps"][m.trainIdx].pt)
                    
                    #Se unen en una lista (ret) los keypoints entre frames por cada match 
                    ret.append((kp1,kp2))
                  
        #FILTER: se filtran los keypoints que sean falsos positivos 
        
        #Para cada par de keypoints cercanos
        if len(ret)>0:
            #Se hace una matriz con los pares de keypoints de frames consecutivas
            
            ret=np.array(ret)
            print(img.shape[0]/2)
            
            #Se NORMALIZAN las coordenadas [:,:,0] y [:,:,1] significa que de cada vector i,j que compone la matriz ret, se coja el primer elemento (0) y se le reste W/2 y el segundo elemento 1 y se le reste H/2 
            # (es decir que cada elemento i,j tiene dimension 2. (a,b) = (a=0, b=1))
            ret[:,:,0]-=img.shape[0]/2
            ret[:,:,1]-=img.shape[1]/2
                        
            #Se utiliza RANSAC para filtrar estos keypoints
            model,inliers=ransac((ret[:,0],ret[:,1]),
                                FundamentalMatrixTransform,
                                min_samples=8,
                                residual_threshold=1,
                                max_trials=100)
            #El array ret contiene los keypoints filtrados, con lo que eliminamos falsos positivos (ruido)
            ret=ret[inliers]
            

        
        #Returning array of keypoints of consecutive frames, already filtered out
        self.last={"kps":kps, "des":des}
        return ret