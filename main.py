import cv2

from segmentor import Segmentor
from segmentor_grab import Segmentor_grab
from compare import *

def get_rect(img):
    ret, thresh = cv2.threshold(img, 10, 255, cv2.THRESH_OTSU)
    im2, contours, hierarchy = cv2.findContours(thresh, 1, 2)
    cnt = max(contours, key=cv2.contourArea)
    return cv2.boundingRect(cnt)


if __name__ == "__main__":
    images_path = ["dataset/ISIC_0000001.jpg", "dataset/ISIC_0000028.jpg", "dataset/ISIC_0000029.jpg", "dataset/pes1.png"]
    image_path = images_path[3]
    # image_path = "dataset/pes1.png"

    image = cv2.imread(image_path)
    image = cv2.resize(image, (512, 512))

    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    segmentor = Segmentor(image_gray)
    duration, out = segmentor.max_flow_gray()

    rect = get_rect(out)
    segmentor = Segmentor_grab(image)
    duration1, out1 = segmentor.segment(rect)

    out1 = cv2.cvtColor(out1, cv2.COLOR_BGR2GRAY)

    compare_images(out, out1)

    print("Time FM: {}".format(duration))
    print("Time Grab: {}".format(duration1))

    cv2.imshow("Im1", image)
    cv2.imshow("Seg1", out)
    cv2.imshow("Seg2", out1)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


