import matplotlib.pyplot as plt
import numpy as np
import maxflow
import copy


class Segmentor:

    def __init__(self, img):
        self.img = copy.copy(img)
        self.mask_color = (1, 255, 255)

    def max_flow_gray(self):
        height, width = self.img.shape
        graph = maxflow.Graph[int](height, width)
        nodes = graph.add_grid_nodes(self.img.shape)
        graph.add_grid_edges(nodes, 0), graph.add_grid_tedges(nodes, self.img, 255 - self.img)
        graph.maxflow()
        mask = graph.get_grid_segments(nodes)
        out = np.ones((height, width, 3))  # Inicializar com 3 canais
        for i in range(height):
            for j in range(width):
                if mask[i, j]:
                    out[i, j, 0], out[i, j, 1], out[i, j, 2] = self.mask_color
                else:
                    out[i, j] = self.img[i, j]
        plt.imshow(out, vmin=0, vmax=255)
        plt.show()
