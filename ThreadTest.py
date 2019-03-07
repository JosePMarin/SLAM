import numpy as np
import argparse
import time
import cv2
from Process import Process
from VideoThread import VideoThread







#Inicializamos la instancia de la clase process
p=Process()

 
# start the file video stream thread and allow the buffer to
# start to fill
print("[INFO] starting video file thread...")

videoThr= VideoThread("./videos/test.mp4").start()
time.sleep(1.0)
print("[INFO]: loading video...")

 

    
print ("[INFO]: processing image...")

while videoThr.queueChecker():
    
    frame=videoThr.read()
    p.process_frame(frame)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break


#Closes all the frames
cv2.destroyAllWindows()


    






