import numpy as np
import maxflow
import copy
from profilehooks import profile
import time



class Segmentor:

    def __init__(self, img):
        self.img = copy.copy(img)
        self.mask_color = (255, 1, 255)

    @profile
    def max_flow_gray(self):
        start = time.time()
        height, width = self.img.shape
        graph = maxflow.Graph[int](height, width)
        nodes = graph.add_grid_nodes(self.img.shape)
        graph.add_grid_edges(nodes, 0), graph.add_grid_tedges(nodes, self.img, 255 - self.img)
        graph.maxflow()
        mask = graph.get_grid_segments(nodes)
        end = time.time()
        return end - start, self.__plot(mask)

    def __plot(self, mask):
        height, width = self.img.shape
        out = np.zeros((height, width),  dtype=np.uint8)
        for i in range(height):
            for j in range(width):
                if mask[i, j]:
                    out[i, j] = self.img[i, j]
                else:
                    out[i, j] = 0

        return out


    # def __plot(self, mask):
    #     height, width = self.img.shape
    #     out = np.zeros((height, width, 3),  dtype=np.uint8)  # Inicializar com 3 canais
    #     for i in range(height):
    #         for j in range(width):
    #             if mask[i, j]:
    #                 out[i, j, 0], out[i, j, 1], out[i, j, 2] = self.mask_color
    #             else:
    #                 out[i, j] = self.img[i, j]
    #
    #     return out






