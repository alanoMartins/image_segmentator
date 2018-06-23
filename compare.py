import cv2
from scipy import spatial
import numpy as np


def compare_images(image1, image2):
    stats = {
        "mse": _mse(image1, image2),
        "cosine": _cosine(image1, image2),
        'template': _match_template(image1, image2)[0][0],
        'histogram': _hist(image1, image2),
        'correlation': _hist_correl(image1, image2),
        'chi_sqr': _hist_chisqr(image1, image2),
        'intersect': _hist_intersetct(image1, image2)

    }

    return stats


def _mse(image1, image2):
    err = np.sum((image1.astype("float") - image2.astype("float")) ** 2)
    err /= float(image1.shape[0] * image1.shape[1])
    return err


def _cosine(image1, image2):
    im1 = image1.flatten()
    im2 = image2.flatten()
    result = spatial.distance.cosine(im2, im1)
    return result


def _match_template(image1, image2):
    res = cv2.matchTemplate(image1, image2, cv2.TM_SQDIFF_NORMED)
    return 1 - res


def _surf(image1, image2):
    surf = cv2.xfeatures2d.SURF_create()

    keypoints1, descriptor1 = surf.detectAndCompute(image1, None)
    keypoints2, descriptor2 = surf.detectAndCompute(image2, None)

    # Match descriptor vectors using FLANN match
    FLANN_INDEX_KDTREE = 1
    flann_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)

    matcher = cv2.BFMatcher()
    matches = matcher.match(descriptor1[1], descriptor2[1])
    matches = sorted(matches, key=lambda val: val.distance)
    distances = [match.distance for match in matches if match.distance < 0.2]


    im_with_k2 = cv2.drawKeypoints(image1, keypoints1, np.array([]), color=255,
                                   flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    im_with_k1 = cv2.drawKeypoints(image2, keypoints2, np.array([]), color=255,
                                   flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    cv2.imshow("k1", im_with_k1)
    cv2.imshow("k2", im_with_k2)

    return len(distances)

def _hist_correl(image1, image2):
    hist1 = cv2.calcHist([image1], [0], None, [256], [0, 256])
    hist2 = cv2.calcHist([image2], [0], None, [256], [0, 256])
    res = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
    return res

def _hist_chisqr(image1, image2):
    hist1 = cv2.calcHist([image1], [0], None, [256], [0, 256])
    hist2 = cv2.calcHist([image2], [0], None, [256], [0, 256])
    res = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CHISQR)
    return res

def _hist_intersetct(image1, image2):
    hist1 = cv2.calcHist([image1], [0], None, [256], [0, 256])
    hist2 = cv2.calcHist([image2], [0], None, [256], [0, 256])
    res = cv2.compareHist(hist1, hist2, cv2.HISTCMP_INTERSECT)
    return res

def _hist(image1, image2):
    hist1 = cv2.calcHist([image1], [0], None, [256], [0, 256])
    hist2 = cv2.calcHist([image2], [0], None, [256], [0, 256])
    res = cv2.compareHist(hist1, hist2, cv2.HISTCMP_BHATTACHARYYA)
    return res