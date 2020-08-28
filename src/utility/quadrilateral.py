import numpy as np
import cv2
from utility.line import Line
from utility.colors import COLOR_YELLOW

class Quadrilateral():

    def __init__(self, pointa, pointb, pointc, pointd):
        # a, b, c, d must be in clockwise or counter clockwise order
        self.pointa = pointa
        self.pointb = pointb
        self.pointc = pointc
        self.pointd = pointd

    def paint_quadrilateral(self, frame):
        frame = cv2.line(frame, self.pointa, self.pointb, color=COLOR_YELLOW, thickness=2)
        frame = cv2.line(frame, self.pointb, self.pointc, color=COLOR_YELLOW, thickness=2)
        frame = cv2.line(frame, self.pointc, self.pointd, color=COLOR_YELLOW, thickness=2)
        frame = cv2.line(frame, self.pointd, self.pointa, color=COLOR_YELLOW, thickness=2)

        return frame

    def quadrilateral_intersection(self, other, tolerance=0):
        alpha, beta = self, other

        alpha1 = Line(alpha.pointa, alpha.pointb)
        alpha2 = Line(alpha.pointb, alpha.pointc)
        alpha3 = Line(alpha.pointc, alpha.pointd)
        alpha4 = Line(alpha.pointd, alpha.pointa)
        alpha_list = [alpha1, alpha2, alpha3, alpha4]

        beta1 = Line(beta.pointa, beta.pointb)
        beta2 = Line(beta.pointb, beta.pointc)
        beta3 = Line(beta.pointc, beta.pointd)
        beta4 = Line(beta.pointd, beta.pointa)

        beta1 = Line.extend_segment(beta1, length=tolerance)
        beta2 = Line.extend_segment(beta2, length=tolerance)
        beta3 = Line.extend_segment(beta3, length=tolerance)
        beta4 = Line.extend_segment(beta4, length=tolerance)
        beta_list = [beta1, beta2, beta3, beta4]

        intersection_bool = False
        for alpha_side in alpha_list:
            for beta_side in beta_list:
                intersection = Line.line_intersection(alpha_side, beta_side)
                if intersection[0] is not None:
                    intersection_bool = True

        return intersection_bool

    def quadrilateral_with_point(self, point):
        quadrilateral, point = self, point

        alpha1 = Line(quadrilateral.pointa, quadrilateral.pointb)
        alpha2 = Line(quadrilateral.pointb, quadrilateral.pointc)
        alpha3 = Line(quadrilateral.pointc, quadrilateral.pointd)
        alpha4 = Line(quadrilateral.pointd, quadrilateral.pointa)

        line1 = Line(quadrilateral.pointa, point)
        line2 = Line(quadrilateral.pointb, point)
        line3 = Line(quadrilateral.pointc, point)
        line4 = Line(quadrilateral.pointd, point)

        pairs = [[line1, alpha2], [line1, alpha3], [line2, alpha3], [line2, alpha4],
                 [line3, alpha4], [line3, alpha1], [line4, alpha1], [line4, alpha2]]

        inside_bool = True
        for pair in pairs:
            intersection = Line.line_intersection(pair[0], pair[1])
            if intersection[0] is not None:
                inside_bool = False

        return inside_bool
