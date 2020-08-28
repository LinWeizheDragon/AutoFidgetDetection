import cv2
import numpy as np
import pandas


def extend_joint(point1, point2, width = 30):
    x1 = point1[0]
    y1 = point1[1]
    x2 = point2[0]
    y2 = point2[1]

    tan_theta = (x2 - x1) / (y1 - y2)
    sin_theta = (x2-x1) / np.sqrt((x2 - x1)**2 + (y1 - y2)**2)
    cos_theta = (y1-y2) / np.sqrt((x2 - x1)**2 + (y1 - y2)**2)

    xa = int(x1 + width * cos_theta)
    ya = int(y1 + width * sin_theta)
    pointa = tuple([xa, ya])

    xb = int(x2 + width * cos_theta)
    yb = int(y2 + width * sin_theta)
    pointb = tuple([xb, yb])

    xc = int(x2 - width * cos_theta)
    yc = int(y2 - width * sin_theta)
    pointc = tuple([xc, yc])

    xd = int(x1 - width * cos_theta)
    yd = int(y1 - width * sin_theta)
    pointd = tuple([xd, yd])

    return pointa, pointb, pointc, pointd
