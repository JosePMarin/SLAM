# pylint: disable=no-member

import cv2
import pygame

W=1920//2
H=1080//2

pygame.init()
screen=pygame.display.set_mode((W,H))


def process_frame(img):
    img=cv2.resize(img,(W,H))
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
