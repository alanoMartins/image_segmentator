import numpy as np
import copy
from profilehooks import profile
import cv2
import time



class Segmentor_grab:

    def __init__(self, img):
        self.img = copy.copy(img)
        self.mask_color = (1, 255, 255)

    @profile
    def segment(self, rect):
        start = time.time()
        mask = np.zeros(self.img.shape[:2], np.uint8)
        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)
        # rect = (165, 125, 200, 200)
        cv2.grabCut(self.img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
        mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
        img = self.img * mask2[:, :, np.newaxis]
        end = time.time()
        return end - start, img





