"""
This module provides functions to convert polarization images between different representations.
The following representations are supported:
- raw: each 2x2 pixel block contains the intensities of the four polarization channels in the order
    [[45, 90],
     [0, -45]]
    degrees. "Unprocessed" polarization image in ThorCam.
- quadlist: a list of four images, each containing the intensity of one polarization channel in the order
    [0, -45, 45, 90]
    degrees.
- quadarray: a 4xHxW numpy array containing the intensity of each polarization channel. Same order as the quadlist.
- quad: a stack of four images, each containing the intensity of one polarization channel in the order
    [[0, -45],
     [45, 90]]
    degrees. "QuadView" polarization image in ThorCam.
- intensity: an image containing the intensity of the four polarization channels,
    calculated as the square root of the sum of the squares of the four channels.
- normalized_intensity: an image containing the mean of the four normalized polarization channels (divided by the mean intensity).
- mean: an image containing the mean intensity of the four polarization channels.
- median: an image containing the median intensity of the four polarization channels.
"""

import cv2
import numpy as np

RAW_POL_DEGREES = [[45, 90], [0, -45]]
QUAD_POL_DEGREES = [[0, -45], [45, 90]]
QUADLIST_POL_DEGREES = [0, -45, 45, 90]
QUADLIST_INDEX_Y = np.array([1, 1, 0, 0])
QUADLIST_INDEX_X = np.array([0, 1, 0, 1])

# Auto with QUADLIST_INDEX_Y and QUADLIST_INDEX_X
QUADLIST_DISPLACEMENT_XY = (
    np.array([[QUADLIST_INDEX_X[i], QUADLIST_INDEX_Y[i]] for i in range(4)]) - 0.5
) / 2
# QUADLIST_DISPLACEMENT_XY = (
#     np.array([[QUADLIST_INDEX_X[i], QUADLIST_INDEX_Y[i]] for i in range(4)]) / 2
# )


def raw_to_quadlist(img):
    return [img[1::2, ::2], img[1::2, 1::2], img[::2, ::2], img[::2, 1::2]]


def raw_to_quadarray(img):
    return np.array(raw_to_quadlist(img))


def raw_to_intensity(img):
    return np.mean(raw_to_quadarray(img), axis=0)


def raw_to_median(img):
    return np.median(raw_to_quadlist(img), axis=0)


def raw_to_mean(img):
    return np.mean(raw_to_quadarray(img), axis=0)


def quad_to_quadlist(img):
    imgy, imgx = img.shape
    imgy2, imgx2 = imgy // 2, imgx // 2
    return [
        img[:imgy2, :imgx2],
        img[:imgy2, imgx2:],
        img[imgy2:, :imgx2],
        img[imgy2:, imgx2:],
    ]


def quadlist_to_intensity(imgs):
    return np.mean(imgs, axis=0)


def quad_to_intensity(img):
    return quadlist_to_intensity(quad_to_quadlist(img))


def quad_to_normalized_intensity(img):
    imgs = np.array([img / img.mean() for img in quad_to_quadlist(img)])
    return quadlist_to_intensity(imgs)


def quadlist_to_quad(imgs):
    return np.vstack((np.hstack((imgs[0], imgs[1])), np.hstack((imgs[2], imgs[3]))))


def quadlist_to_quad_pad(imgs, pad=0, padvalue=0):
    ysize, xsize = imgs[0].shape
    return np.vstack(
        (
            np.hstack((imgs[0], np.full((ysize, pad), padvalue), imgs[1])),
            np.full((pad, xsize * 2 + pad), padvalue),
            np.hstack((imgs[2], np.full((ysize, pad), padvalue), imgs[3])),
        )
    )


def raw_to_quad(img):
    return quadlist_to_quad(raw_to_quadlist(img))


positions = [np.zeros((2, 2)) for _ in range(4)]
positions[0][1, 0] = 1
positions[1][1, 1] = 1
positions[2][0, 0] = 1
positions[3][0, 1] = 1


def quadlist_to_raw(imgs):
    return np.sum(
        [np.kron(img, position) for (img, position) in zip(imgs, positions)], axis=0
    )


def quad_to_raw(img):
    return quadlist_to_raw(quad_to_quadlist(img))


def double(img, interpolation=0):
    return cv2.resize(img, dsize=None, fx=2, fy=2, interpolation=interpolation)


def undouble(img):
    return (
        img[:-1:2, :-1:2] + img[1::2, :-1:2] + img[:-1:2, 1::2] + img[1::2, 1::2]
    ) / 4


def raw_to_quadlist2(img):
    return [
        double(img[1::2, ::2])[1:, :-1],
        double(img[1::2, 1::2])[1:, 1:],
        double(img[::2, ::2])[:-1, :-1],
        double(img[::2, 1::2])[:-1, 1:],
    ]


def quad_to_quadlist2(img):
    return raw_to_quadlist2(quad_to_raw(img))
