import cv2
import numpy as np

def paint_point(img, point, color=(0, 0, 255)):
    '''
    This function paints point group
    :param img:
    :param point: shape of (-1, 2)
    :param color:
    :return:
    '''
    cv2.circle(img, (int(point[0]), int(point[1])), 1, color, -1)
    return img


def paint_text(img, text, point, color=(0, 205, 193)):
    '''
    This function add text to a point
    :param img:
    :param text:
    :param point:
    :param color:
    :return:
    '''
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, text, (int(point[0]), int(point[1])), font, 0.4, color, 1)
    return img


def paint_line(img, point1, point2, color=(0, 205, 193)):
    '''
    This function paints body joints
    :param img:
    :param point_1: shape of (2, )
    :param point_2:  shape of (2, )
    :param colour:
    :return:
    '''
    cv2.line(img, (int(point1[0]), int(point1[1])), (int(point2[0]), int(point2[1])), color, 2)
    return img


def paint_rectangle_to_points(img, points, color=(0, 255, 0)):
    '''
    This function paints rectangle to a group of points
    :param img:
    :param points: shape of (-1, 2)
    :param color:
    :return:
    '''
    points = np.array(points)
    cv2.rectangle(img,
                  tuple(np.min(points, axis=0).astype(int)),
                  tuple(np.max(points, axis=0).astype(int)),
                  color,
                  1)
    return img