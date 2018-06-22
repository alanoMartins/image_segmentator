import numpy as np
import copy
from profilehooks import profile
import cv2



class Segmentor_grab:

    def __init__(self, img):
        self.img = copy.copy(img)
        self.mask_color = (1, 255, 255)

    def segment(self):
        mask = np.zeros(self.img.shape[:2], np.uint8)
        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)
        rect = (165, 125, 200, 200)
        cv2.grabCut(self.img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
        mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
        img = self.img * mask2[:, :, np.newaxis]
        return img


    def __plot(self, mask):
        height, width = self.img.shape
        out = np.ones((height, width, 3))  # Inicializar com 3 canais
        for i in range(height):
            for j in range(width):
                if mask[i, j]:
                    out[i, j, 0], out[i, j, 1], out[i, j, 2] = self.mask_color
                else:
                    out[i, j] = self.img[i, j]
        # plt.imshow(out)
        # plt.show()




