import cv2

from segmentor import Segmentor

if __name__ == "__main__":
    image_path = "dataset/ISIC_0000029.jpg"

    image = cv2.imread(image_path, 0)
    segmentor = Segmentor(image)
    segmentor.max_flow_gray()