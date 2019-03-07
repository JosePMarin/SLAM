from threading import Thread
import sys
import cv2
import time



#Controller to handle the import of the module for the correct python version.
if sys.version_info>= (3,0):
    from queue import Queue
else:
    from queue import Queue

class VideoThread:

    #Constructor de VideoThread, incluyendo el buffer donde guardamos los frames
    def __init__(self, path, queueSize=120):
        self.queueSize=queueSize
        self.path=path
        self.stream=cv2.VideoCapture(path)
        self.stopped=False
        #tamaño del buffer en el nuevo thread donde se van a guardar los frames
        self.queue=Queue(maxsize=self.queueSize)

    def get(self):
        #Definimos un bucle de control para salir del thread cuando se para el mismo
        
        while True:
            
            #if self.stopped
            #    import pdb; pdb.set_trace()   
            #    return

            #En caso de de funcionar el hilo, asegurar que queda espacio en el buffer
            if not self.queue.full():
                #Lee el siguiente frame del file
                (grabbed, frame)=self.stream.read()

                #la funcion .read() devuelve los frames y un boolean (grab) que será False si el proximo frame es null (final del video)
                if not grabbed:
                    self.stopped=True
                    return
                
                #añadir el frame al buffer queue:
                self.queue.put(frame)


        #Function member para iniciar el thread
    def start(self):
        thread=Thread(target=self.get, args=())

        #Se define un thread daemon como principal (al terminar el hilo, termina el programa). Siempre antes de start()
        thread.daemon=True
        thread.start()
        return self
    
    def read(self):

        #return el siguiente frame almacenado en el buffer (queue)
        return self.queue.get()

    def queueChecker(self):

        #return True si el buffer (queue) aun tiene frames, en caso contrario return 0 (False)
        
        return self.queue.qsize()>0

    def stop(self):

        #Para parar el thread
        self.stopped=True

        



    def running (self):

        return self.queueChecker() or not self.stopped
    







