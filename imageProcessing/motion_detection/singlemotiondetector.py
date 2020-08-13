import numpy as np
import imutils
import cv2

class SingleMotionDetector:
    def __init__(self,accumWeight=0.5):
        #store the accumalated weight factor
        self.accumWeight = accumWeight
        #initialize the background model
        self.bg = None

    def update(self,image):
        #if the background model is None,itialize it
        if self.bg is None:
            self.bg = image.copy().astype("float")
            return

        # update the background model by accumalating the weighted
        # average
        cv2.accumulateWeighted(image,self.bg,self.accumWeight)