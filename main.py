import cv2

from segmentor import Segmentor
from segmentor_grab import Segmentor_grab
from compare import *


if __name__ == "__main__":
    image_path = "dataset/ISIC_0000001.jpg"

    image_path1 = "dataset/ISIC_0000028.jpg"

    image = cv2.imread(image_path, 0)
    image = cv2.resize(image, (512, 512))
    image1 = cv2.imread(image_path1, 0)
    image1 = cv2.resize(image1, (512, 512))

    segmentor = Segmentor(image)
    out = segmentor.max_flow_gray()


    image_ = cv2.imread(image_path)
    image_ = cv2.resize(image_, (512, 512))
    segmentor = Segmentor_grab(image_)
    out1 = segmentor.segment()

    out1 = cv2.cvtColor(out1, cv2.COLOR_BGR2GRAY)

    compare_images(out, out1)


    cv2.imshow("Im1", image)
    cv2.imshow("Im2", image1)
    cv2.imshow("Seg1", out)
    cv2.imshow("Seg2", out1)

    cv2.waitKey(0)
    cv2.destroyAllWindows()