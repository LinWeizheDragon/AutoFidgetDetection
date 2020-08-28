import numpy as np


class Line:
    def __init__(self, point1, point2):
        self.x1, self.y1, self.x2, self.y2 = point1[0], point1[1], point2[0], point2[1]

    def slope(self):
        if (self.x2 - self.x1) == 0:
            return 1e10
        else:
            m = (self.y2 - self.y1) / (self.x2 - self.x1)
            return m

    def x_bound(self, other):
        a, b = self, other
        x_lower = max(min(a.x1, a.x2), min(b.x1, b.x2))
        x_upper = min(max(a.x1, a.x2), max(b.x1, b.x2))
        return x_lower, x_upper

    def y_bound(self, other):
        a, b = self, other
        y_lower = max(min(a.y1, a.y2), min(b.y1, b.y2))
        y_upper = min(max(a.y1, a.y2), max(b.y1, b.y2))
        return y_lower, y_upper

    def extend_segment(self, length):
        cos_theta = (self.x2 - self.x1) / np.sqrt((self.x2-self.x1)**2 + (self.y2-self.y1)**2)
        sin_theta = (self.y2 - self.y1) / np.sqrt((self.x2-self.x1)**2 + (self.y2-self.y1)**2)
        x1_ = self.x1 - length * cos_theta
        y1_ = self.y1 - length * sin_theta
        x2_ = self.x2 + length * cos_theta
        y2_ = self.y2 + length * sin_theta
        extended_line = Line((x1_, y1_), (x2_, y2_))

        return extended_line


    def line_intersection(self, other):
        a, b = self, other

        m_a = a.slope()
        m_b = b.slope()
        x_lower, x_upper = Line.x_bound(a, b)
        y_lower, y_upper = Line.y_bound(a, b)

        if m_a == m_b:
            # print('Two segments are parallel')
            return None, None

        else:
            x_intersection = ((b.y1 - m_b * b.x1) - (a.y1 - m_a * a.x1)) / (m_a - m_b)
            y_intersection = m_a * x_intersection + a.y1 - m_a * a.x1

            if x_lower <= x_intersection <= x_upper:
                # print('Two segments intersects')
                return x_intersection, y_intersection
            # if y_lower <= y_intersection <= y_upper:
            #     return x_intersection, y_intersection
            else:
                # print('Two segments does not intersect')
                return None, None

    __and__ = line_intersection
