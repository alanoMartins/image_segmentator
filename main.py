import cv2
import matplotlib.pyplot as plt
import os

from segmentor import Segmentor
from segmentor_grab import Segmentor_grab
from compare import *

SAMPLE_DIR = "dataset/"
OUTPUT_DIR = "output/"

def get_rect(img):
    ret, thresh = cv2.threshold(img, 50, 120, cv2.THRESH_OTSU)
    im2, contours, hierarchy = cv2.findContours(thresh, 1, 2)
    cnt = max(contours, key=cv2.contourArea)
    return cv2.boundingRect(cnt)

def list_files():
    folder = next(os.walk(SAMPLE_DIR))[2]
    for file in folder:
        yield SAMPLE_DIR + file


if __name__ == "__main__":

    images_path = list(list_files())
    #image_path = images_path[3]
    # image_path = "dataset/pes1.png"

    times = []
    set_stats = []

    for image_path in images_path:

        image = cv2.imread(image_path)
        image = cv2.resize(image, (512, 512))

        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        segmentor = Segmentor(image_gray)
        duration, out = segmentor.max_flow_gray()

        rect = get_rect(out)
        segmentor = Segmentor_grab(image)
        duration1, out1 = segmentor.segment(rect)

        out1 = cv2.cvtColor(out1, cv2.COLOR_BGR2GRAY)

        set_stats.append(compare_images(out, out1))
        times.append((duration, duration1))

        output_path = OUTPUT_DIR + image_path.split('/')[1].split('.')[0] + "_graph_cut.jpg"
        output_path1 = OUTPUT_DIR + image_path.split('/')[1].split('.')[0] + "_grab_cut.jpg"

        cv2.imwrite(output_path, out)
        cv2.imwrite(output_path1, out1)

    time_fm, time_gm = zip(*times)
    mse = list(map(lambda x: x['mse'], set_stats))
    cosine = list(map(lambda x: x['cosine'], set_stats))
    template = list(map(lambda x: x['template'], set_stats))
    histogram = list(map(lambda x: x['histogram'], set_stats))
    correlation = list(map(lambda x: x['correlation'], set_stats))
    chi_sqr = list(map(lambda x: x['chi_sqr'], set_stats))
    intersect = list(map(lambda x: x['intersect'], set_stats))

    plt.title("Time Execution")
    plt.plot(time_fm, label="Graph cut")
    plt.plot(time_gm, label="Grab cut")
    plt.legend()
    plt.show()

    plt.subplot(221)
    plt.title("MSE")
    plt.plot(mse)

    plt.subplot(222)
    plt.title("Cosine")
    plt.plot(cosine)

    plt.subplot(223)
    plt.title("Histogram")
    plt.plot(histogram)

    plt.subplot(224)
    plt.title("Template")
    plt.plot(template)


    plt.tight_layout()
    plt.show()

    plt.subplot(221)
    plt.title("Correlation")
    plt.plot(correlation)

    plt.subplot(222)
    plt.title("Chi Square")
    plt.plot(chi_sqr)

    plt.tight_layout()
    plt.show()

    #
    # print("Time FM: {}".format(duration))
    # print("Time Grab: {}".format(duration1))
    #
    # cv2.imshow("Im1", image)
    # cv2.imshow("Seg1", out)
    # cv2.imshow("Seg2", out1)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


