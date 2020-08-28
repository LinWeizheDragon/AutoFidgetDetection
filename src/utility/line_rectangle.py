from utility.line import Line
from utility.rectangle import Rectangle

class Line_Rectangle:
    def __init__(self, line, rectangle):
        self.line = line
        self.rectangle = rectangle

        self.rect_x_lower = self.rectangle.x1
        self.rect_x_upper = self.rectangle.x2
        self.rect_y_lower = self.rectangle.y1
        self.rect_y_upper = self.rectangle.y2

        self.diagonal1 = Line(self.rect_x_lower, self.rect_y_lower, self.rect_x_upper, self.rect_y_upper)
        self.diagonal2 = Line(self.rect_x_lower, self.rect_y_upper, self.rect_x_upper, self.rect_y_lower)

        self.top = Line(self.rect_x_lower, self.rect_y_upper, self.rect_x_upper, self.rect_y_upper)
        self.bottom = Line(self.rect_x_lower, self.rect_y_lower, self.rect_x_upper, self.rect_y_lower)
        self.left = Line(self.rect_x_lower, self.rect_y_lower, self.rect_x_lower, self.rect_y_upper)
        self.right = Line(self.rect_x_upper, self.rect_y_lower, self.rect_x_upper, self.rect_y_upper)

    # If the line intersects with the rectangle, it must intersect with the diagonal of the rectangle
    def line_rectangle_intersection(self):
        line = self.line
        rectangle = self.rectangle
        top = self.top
        bottom = self.bottom
        left = self.left
        right = self.right

        intersection1 = Line.line_intersection(line, top)
        intersection2 = Line.line_intersection(line, right)
        intersection3 = Line.line_intersection(line, bottom)
        intersection4 = Line.line_intersection(line, left)
        intersection_list = [intersection1, intersection2, intersection3, intersection4]
        # print(intersection_list)

        for intersection in intersection_list:
            if intersection[0] is not None:
                return True

        return False
