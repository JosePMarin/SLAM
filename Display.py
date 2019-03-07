# pylint: disable=no-member
#!/usr/bin/env python3
from Process import Process 
import cv2


p=Process()

class Display(object):

    def displayVideoRecord(self,path):
    #if __name__=="__main__":   
        path="./videos/test.mp4"     
        #Al ejecutar el .py, ejecuta la captura de video 
        self.cap=cv2.VideoCapture(path)

        #Se crea un bucle para ejecutar procesado de imagen mientras el video se reproduzca
        while self.cap.isOpened():
            # Capture frame-by-frame
            ret, frame = self.cap.read()
            if ret == True:
                
                p.process_frame(frame)
                #Press Q on keyboard to exit
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break
            #Break the loop        
            else:
                break
        #When everything done, release the video capture object        
        self.cap.release()
        #Closes all the frames
        cv2.destroyAllWindows() 

        


