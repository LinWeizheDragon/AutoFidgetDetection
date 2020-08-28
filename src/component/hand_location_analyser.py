import os
import json
import cv2
import math
import numpy as np
import pandas as pd
from utility.base_config import *
from scipy.signal import savgol_filter
from utility.colors import *
from utility.rectangle import Rectangle
from utility.line import Line
from utility.line_rectangle import Line_Rectangle
from utility.label_parser import Label_Parser
from utility.const import BODY_CONNECTION
from utility.quadrilateral import Quadrilateral
from utility.elan_portal import ElanPortal

from component.basic_processor import BasicProcessor
from component.hand_cross_analyser import HandCrossAnalyser


class HandLocationAnalyser(BasicProcessor):

    def __init__(self, name, path_data, hand='left'):
        BasicProcessor.__init__(self, name, path_data, None)
        self.hand = hand

    def compute_hand_intersection(self, min_length=100, cutoff=0):
        if self.hand == 'left':
            return self.compute_left_hand_intersection(min_length=min_length, cutoff=cutoff)
        else:
            return self.compute_right_hand_intersection(min_length=min_length, cutoff=cutoff)

    def joint_to_quad(self, point1, point2, width=30):
        x1 = point1[0]
        y1 = point1[1]
        x2 = point2[0]
        y2 = point2[1]
        # tan_theta = (x2 - x1) / (y1 - y2)

        with np.errstate(divide='ignore', invalid='ignore'):
            sin_theta = (x2 - x1) / np.sqrt((x2 - x1) ** 2 + (y1 - y2) ** 2)
            cos_theta = (y1 - y2) / np.sqrt((x2 - x1) ** 2 + (y1 - y2) ** 2)
            sin_theta = np.nan_to_num(sin_theta)
            cos_theta = np.nan_to_num(cos_theta)

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

        quad = Quadrilateral(pointa, pointb, pointc, pointd)

        return quad

    def check_quad_overlap(self, alpha, beta, tolerance=0):
        intersect_check = Quadrilateral.quadrilateral_intersection(alpha, beta, tolerance=tolerance)

        return intersect_check

    def compute_right_hand_intersection(self, min_length=100, cutoff=0):
        cap = cv2.VideoCapture(self.video_path)
        data = np.load(self.processed_smooth_file)

        # try:
        hand_arm_intersect_data = {}
        hand_leg_intersect_data = {}
        hand_face_intersect_data = {}

        instance_hand_cross = HandCrossAnalyser(self.name, self.path_data)
        continuous_segments, hand_cross_intersect_data = instance_hand_cross.compute_stationary_rectangles(cutoff=0,
                                                                                                           min_length=20)

        t = 0
        while (t < np.shape(data)[0]):
            ret, frame = cap.read()
            print('progress', t / data.shape[0], end='\r')

            # Load data
            left_hand_data = data[t, 194:232].reshape(-1, 2)
            right_hand_data = data[t, 236:274].reshape(-1, 2)
            left_upper_arm = [(int(data[t, 2 * 5]), int(data[t, 2 * 5 + 1])),
                              (int(data[t, 2 * 6]), int(data[t, 2 * 6 + 1]))]
            left_lower_arm = [(int(data[t, 2 * 6]), int(data[t, 2 * 6 + 1])),
                              (int(data[t, 2 * 7]), int(data[t, 2 * 7 + 1]))]
            right_upper_leg = [(int(data[t, 2 * 9]), int(data[t, 2 * 9 + 1])),
                               (int(data[t, 2 * 10]), int(data[t, 2 * 10 + 1]))]
            right_lower_leg = [(int(data[t, 2 * 10]), int(data[t, 2 * 10 + 1])),
                               (int(data[t, 2 * 11]), int(data[t, 2 * 11 + 1]))]
            left_upper_leg = [(int(data[t, 2 * 12]), int(data[t, 2 * 12 + 1])),
                              (int(data[t, 2 * 13]), int(data[t, 2 * 13 + 1]))]
            left_lower_leg = [(int(data[t, 2 * 13]), int(data[t, 2 * 13 + 1])),
                              (int(data[t, 2 * 14]), int(data[t, 2 * 14 + 1]))]
            face_data = data[t, 50:190].reshape(-1, 2)

            # Define hand quad
            hand_quad_a = [np.min(right_hand_data, axis=0).astype(int)[0],
                           np.min(right_hand_data, axis=0).astype(int)[1]]
            hand_quad_b = [np.max(right_hand_data, axis=0).astype(int)[0],
                           np.min(right_hand_data, axis=0).astype(int)[1]]
            hand_quad_c = [np.max(right_hand_data, axis=0).astype(int)[0],
                           np.max(right_hand_data, axis=0).astype(int)[1]]
            hand_quad_d = [np.min(right_hand_data, axis=0).astype(int)[0],
                           np.max(right_hand_data, axis=0).astype(int)[1]]
            hand_quad = Quadrilateral(hand_quad_a, hand_quad_b, hand_quad_c, hand_quad_d)

            # Property
            intersection = False

            # Check hands overlapping
            if t in hand_cross_intersect_data.keys():
                intersection = True
                # x1, y1, x2, y2 = hand_intersect.get_cordinate()
                # cv2.rectangle(frame,
                #               (x1, y1),
                #               (x2, y2),
                #               COLOR_YELLOW,
                #               2)

            # Check left_hand-right_arm overlapping
            if not intersection:
                right_upper_arm_quad = self.joint_to_quad(left_upper_arm[0], left_upper_arm[1], width=10)
                right_lower_arm_quad = self.joint_to_quad(left_lower_arm[0], left_lower_arm[1], width=10)
                right_upper_arm_overlap = self.check_quad_overlap(right_upper_arm_quad, hand_quad)
                right_lower_arm_overlap = self.check_quad_overlap(right_lower_arm_quad, hand_quad)

                if right_upper_arm_overlap or right_lower_arm_overlap:
                    intersection = True
                    hand_arm_intersect_data[t] = 1

                    # frame = right_upper_arm_quad.paint_quadrilateral(frame)
                    # frame = right_lower_arm_quad.paint_quadrilateral(frame)
                    # for i in range(np.shape(right_hand_data)[0]):
                    #     frame = self.paint_point(frame, right_hand_data[i], color=COLOR_YELLOW)
                    # frame = self.paint_rectangle_to_points(frame, right_hand_data, color=(0, 255, 0))

            # Check left_hand-right_leg overlapping
            if not intersection:
                right_upper_leg_quad = self.joint_to_quad(right_upper_leg[0], right_upper_leg[1], width=15)
                right_lower_leg_quad = self.joint_to_quad(right_lower_leg[0], right_lower_leg[1], width=15)
                left_upper_leg_quad = self.joint_to_quad(left_upper_leg[0], left_upper_leg[1], width=15)
                left_lower_leg_quad = self.joint_to_quad(left_lower_leg[0], left_lower_leg[1], width=15)

                right_upper_leg_overlap = self.check_quad_overlap(right_upper_leg_quad, hand_quad)
                right_lower_leg_overlap = self.check_quad_overlap(right_lower_leg_quad, hand_quad)
                left_upper_leg_overlap = self.check_quad_overlap(left_upper_leg_quad, hand_quad)
                left_lower_leg_overlap = self.check_quad_overlap(left_lower_leg_quad, hand_quad)

                condition = right_upper_leg_overlap or right_lower_leg_overlap or \
                            left_upper_leg_overlap or left_lower_leg_overlap

                if condition:
                    intersection = True
                    hand_leg_intersect_data[t] = 1

                    # frame = right_upper_leg_quad.paint_quadrilateral(frame)
                    # frame = right_lower_leg_quad.paint_quadrilateral(frame)
                    # frame = left_upper_leg_quad.paint_quadrilateral(frame)
                    # frame = left_lower_leg_quad.paint_quadrilateral(frame)

                    # for i in range(np.shape(right_hand_data)[0]):
                    #     frame = self.paint_point(frame, right_hand_data[i], color=COLOR_YELLOW)

                    # frame = self.paint_rectangle_to_points(frame, right_hand_data, color=(0, 255, 0))

                # frame = self.paint_rectangle_to_points(frame, right_hand_data, color=COLOR_YELLOW)

            # Check left_hand-face overlapping
            if not intersection:
                intersect = self.check_rect_overlap(right_hand_data, face_data, tolerance=5)
                if intersect is not None:
                    intersection = True
                    hand_face_intersect_data[t] = 1

                #     x1, y1, x2, y2 = intersect.get_cordinate()
                #     cv2.rectangle(frame,
                #                   (x1, y1),
                #                   (x2, y2),
                #                   COLOR_GREEN,
                #                   2)
                # # frame = self.paint_rectangle_to_points(frame, right_hand_data, color=(0, 255, 0))
                # # frame = self.paint_rectangle_to_points(frame, face_data, color=(0, 255, 0))

            # if not intersection:
            #     print('no intersection found')

            # cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            # if t == 0:
            t += 1

        cap.release()
        cv2.destroyAllWindows()

        # Hand arm intersection
        valid_intersect_data = hand_arm_intersect_data

        continuous_segments = []
        for i in valid_intersect_data.keys():
            if len(continuous_segments) == 0:
                continuous_segments.append([i, i + 1])
            else:
                if continuous_segments[-1][1] == i:
                    continuous_segments[-1][1] += 1
                else:
                    continuous_segments.append([i, i + 1])

        new_segments = []
        for segment in continuous_segments:
            if segment[1] - segment[0] >= (min_length + cutoff * 2):
                new_segments.append([segment[0] + cutoff, segment[1] - cutoff])
                if cutoff != 0:
                    for x in range(segment[0], segment[0] + cutoff):
                        del valid_intersect_data[x]
                    for x in range(segment[1] - cutoff, segment[1]):
                        del valid_intersect_data[x]
            else:
                for x in range(segment[0], segment[1]):
                    del valid_intersect_data[x]
        hand_arm_continuous_segments = new_segments

        # Hand leg intersection
        valid_intersect_data = hand_leg_intersect_data

        continuous_segments = []
        for i in valid_intersect_data.keys():
            if len(continuous_segments) == 0:
                continuous_segments.append([i, i + 1])
            else:
                if continuous_segments[-1][1] == i:
                    continuous_segments[-1][1] += 1
                else:
                    continuous_segments.append([i, i + 1])

        new_segments = []
        for segment in continuous_segments:
            if segment[1] - segment[0] >= (min_length + cutoff * 2):
                new_segments.append([segment[0] + cutoff, segment[1] - cutoff])
                if cutoff != 0:
                    for x in range(segment[0], segment[0] + cutoff):
                        del valid_intersect_data[x]
                    for x in range(segment[1] - cutoff, segment[1]):
                        del valid_intersect_data[x]
            else:
                for x in range(segment[0], segment[1]):
                    del valid_intersect_data[x]
        hand_leg_continuous_segments = new_segments

        # Hand face intersection
        valid_intersect_data = hand_face_intersect_data

        continuous_segments = []
        for i in valid_intersect_data.keys():
            if len(continuous_segments) == 0:
                continuous_segments.append([i, i + 1])
            else:
                if continuous_segments[-1][1] == i:
                    continuous_segments[-1][1] += 1
                else:
                    continuous_segments.append([i, i + 1])

        new_segments = []
        for segment in continuous_segments:
            if segment[1] - segment[0] >= (min_length + cutoff * 2):
                new_segments.append([segment[0] + cutoff, segment[1] - cutoff])
                if cutoff != 0:
                    for x in range(segment[0], segment[0] + cutoff):
                        del valid_intersect_data[x]
                    for x in range(segment[1] - cutoff, segment[1]):
                        del valid_intersect_data[x]
            else:
                for x in range(segment[0], segment[1]):
                    del valid_intersect_data[x]
        hand_face_continuous_segments = new_segments

        return hand_arm_continuous_segments, hand_leg_continuous_segments, hand_face_continuous_segments

    def compute_left_hand_intersection(self, min_length=100, cutoff=0):
        cap = cv2.VideoCapture(self.video_path)
        data = np.load(self.processed_smooth_file)

        # try:
        hand_arm_intersect_data = {}
        hand_leg_intersect_data = {}
        hand_face_intersect_data = {}

        instance_hand_cross = HandCrossAnalyser(self.name, self.path_data)
        continuous_segments, hand_cross_intersect_data = instance_hand_cross.compute_stationary_rectangles(cutoff=0,
                                                                                                           min_length=20)

        t = 0
        while (t < np.shape(data)[0]):
            ret, frame = cap.read()
            print('progress', t / data.shape[0], end='\r')

            # Load data
            left_hand_data = data[t, 194:232].reshape(-1, 2)
            right_hand_data = data[t, 236:274].reshape(-1, 2)
            right_upper_arm = [(int(data[t, 2 * 2]), int(data[t, 2 * 2 + 1])),
                               (int(data[t, 2 * 3]), int(data[t, 2 * 3 + 1]))]
            right_lower_arm = [(int(data[t, 2 * 3]), int(data[t, 2 * 3 + 1])),
                               (int(data[t, 2 * 4]), int(data[t, 2 * 4 + 1]))]
            right_upper_leg = [(int(data[t, 2 * 9]), int(data[t, 2 * 9 + 1])),
                               (int(data[t, 2 * 10]), int(data[t, 2 * 10 + 1]))]
            right_lower_leg = [(int(data[t, 2 * 10]), int(data[t, 2 * 10 + 1])),
                               (int(data[t, 2 * 11]), int(data[t, 2 * 11 + 1]))]
            left_upper_leg = [(int(data[t, 2 * 12]), int(data[t, 2 * 12 + 1])),
                              (int(data[t, 2 * 13]), int(data[t, 2 * 13 + 1]))]
            left_lower_leg = [(int(data[t, 2 * 13]), int(data[t, 2 * 13 + 1])),
                              (int(data[t, 2 * 14]), int(data[t, 2 * 14 + 1]))]
            face_data = data[t, 50:190].reshape(-1, 2)

            # Define hand quad
            hand_quad_a = [np.min(left_hand_data, axis=0).astype(int)[0], np.min(left_hand_data, axis=0).astype(int)[1]]
            hand_quad_b = [np.max(left_hand_data, axis=0).astype(int)[0], np.min(left_hand_data, axis=0).astype(int)[1]]
            hand_quad_c = [np.max(left_hand_data, axis=0).astype(int)[0], np.max(left_hand_data, axis=0).astype(int)[1]]
            hand_quad_d = [np.min(left_hand_data, axis=0).astype(int)[0], np.max(left_hand_data, axis=0).astype(int)[1]]
            hand_quad = Quadrilateral(hand_quad_a, hand_quad_b, hand_quad_c, hand_quad_d)

            # Property
            intersection = False

            # Check hands overlapping
            if t in hand_cross_intersect_data.keys():
                intersection = True
                # x1, y1, x2, y2 = hand_intersect.get_cordinate()
                # cv2.rectangle(frame,
                #               (x1, y1),
                #               (x2, y2),
                #               COLOR_YELLOW,
                #               2)

            # Check left_hand-right_arm overlapping
            if not intersection:
                right_upper_arm_quad = self.joint_to_quad(right_upper_arm[0], right_upper_arm[1], width=10)
                right_lower_arm_quad = self.joint_to_quad(right_lower_arm[0], right_lower_arm[1], width=10)
                right_upper_arm_overlap = self.check_quad_overlap(right_upper_arm_quad, hand_quad)
                right_lower_arm_overlap = self.check_quad_overlap(right_lower_arm_quad, hand_quad)

                if right_upper_arm_overlap or right_lower_arm_overlap:
                    intersection = True
                    hand_arm_intersect_data[t] = 1

                    # frame = right_upper_arm_quad.paint_quadrilateral(frame)
                    # frame = right_lower_arm_quad.paint_quadrilateral(frame)
                    # for i in range(np.shape(left_hand_data)[0]):
                    #     frame = self.paint_point(frame, left_hand_data[i], color=COLOR_YELLOW)
                    # frame = self.paint_rectangle_to_points(frame, left_hand_data, color=(0, 255, 0))

            # Check left_hand-right_leg overlapping
            if not intersection:
                right_upper_leg_quad = self.joint_to_quad(right_upper_leg[0], right_upper_leg[1], width=15)
                right_lower_leg_quad = self.joint_to_quad(right_lower_leg[0], right_lower_leg[1], width=15)
                left_upper_leg_quad = self.joint_to_quad(left_upper_leg[0], left_upper_leg[1], width=15)
                left_lower_leg_quad = self.joint_to_quad(left_lower_leg[0], left_lower_leg[1], width=15)

                right_upper_leg_overlap = self.check_quad_overlap(right_upper_leg_quad, hand_quad)
                right_lower_leg_overlap = self.check_quad_overlap(right_lower_leg_quad, hand_quad)
                left_upper_leg_overlap = self.check_quad_overlap(left_upper_leg_quad, hand_quad)
                left_lower_leg_overlap = self.check_quad_overlap(left_lower_leg_quad, hand_quad)

                condition = right_upper_leg_overlap or right_lower_leg_overlap or \
                            left_upper_leg_overlap or left_lower_leg_overlap

                if condition:
                    intersection = True
                    hand_leg_intersect_data[t] = 1

                #     frame = right_upper_leg_quad.paint_quadrilateral(frame)
                #     frame = right_lower_leg_quad.paint_quadrilateral(frame)
                #     frame = left_upper_leg_quad.paint_quadrilateral(frame)
                #     frame = left_lower_leg_quad.paint_quadrilateral(frame)
                #
                #     for i in range(np.shape(left_hand_data)[0]):
                #         frame = self.paint_point(frame, left_hand_data[i], color=COLOR_YELLOW)
                #
                #     frame = self.paint_rectangle_to_points(frame, left_hand_data, color=(0, 255, 0))
                #
                # # frame = self.paint_rectangle_to_points(frame, left_hand_data, color=COLOR_YELLOW)

            # Check left_hand-face overlapping
            if not intersection:
                intersect = self.check_rect_overlap(left_hand_data, face_data, tolerance=10)
                if intersect is not None:
                    intersection = True
                    hand_face_intersect_data[t] = 1

                    # x1, y1, x2, y2 = intersect.get_cordinate()
                    # cv2.rectangle(frame,
                    #               (x1, y1),
                    #               (x2, y2),
                    #               COLOR_GREEN,
                    #               2)
                # frame = self.paint_rectangle_to_points(frame, left_hand_data, color=(0, 255, 0))
                # frame = self.paint_rectangle_to_points(frame, face_data, color=(0, 255, 0))

            # if not intersection:
            #     no_intersect_data[t] = 1

            # cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            t += 1

        cap.release()
        cv2.destroyAllWindows()
        
        # Hand arm intersection
        valid_intersect_data = hand_arm_intersect_data

        continuous_segments = []
        for i in valid_intersect_data.keys():
            if len(continuous_segments) == 0:
                continuous_segments.append([i, i + 1])
            else:
                if continuous_segments[-1][1] == i:
                    continuous_segments[-1][1] += 1
                else:
                    continuous_segments.append([i, i + 1])

        new_segments = []
        for segment in continuous_segments:
            if segment[1] - segment[0] >= (min_length + cutoff * 2):
                new_segments.append([segment[0] + cutoff, segment[1] - cutoff])
                if cutoff != 0:
                    for x in range(segment[0], segment[0] + cutoff):
                        del valid_intersect_data[x]
                    for x in range(segment[1] - cutoff, segment[1]):
                        del valid_intersect_data[x]
            else:
                for x in range(segment[0], segment[1]):
                    del valid_intersect_data[x]
        hand_arm_continuous_segments = new_segments

        # Hand leg intersection
        valid_intersect_data = hand_leg_intersect_data

        continuous_segments = []
        for i in valid_intersect_data.keys():
            if len(continuous_segments) == 0:
                continuous_segments.append([i, i + 1])
            else:
                if continuous_segments[-1][1] == i:
                    continuous_segments[-1][1] += 1
                else:
                    continuous_segments.append([i, i + 1])

        new_segments = []
        for segment in continuous_segments:
            if segment[1] - segment[0] >= (min_length + cutoff * 2):
                new_segments.append([segment[0] + cutoff, segment[1] - cutoff])
                if cutoff != 0:
                    for x in range(segment[0], segment[0] + cutoff):
                        del valid_intersect_data[x]
                    for x in range(segment[1] - cutoff, segment[1]):
                        del valid_intersect_data[x]
            else:
                for x in range(segment[0], segment[1]):
                    del valid_intersect_data[x]
        hand_leg_continuous_segments = new_segments

        # Hand face intersection
        valid_intersect_data = hand_face_intersect_data

        continuous_segments = []
        for i in valid_intersect_data.keys():
            if len(continuous_segments) == 0:
                continuous_segments.append([i, i + 1])
            else:
                if continuous_segments[-1][1] == i:
                    continuous_segments[-1][1] += 1
                else:
                    continuous_segments.append([i, i + 1])

        new_segments = []
        for segment in continuous_segments:
            if segment[1] - segment[0] >= (min_length + cutoff * 2):
                new_segments.append([segment[0] + cutoff, segment[1] - cutoff])
                if cutoff != 0:
                    for x in range(segment[0], segment[0] + cutoff):
                        del valid_intersect_data[x]
                    for x in range(segment[1] - cutoff, segment[1]):
                        del valid_intersect_data[x]
            else:
                for x in range(segment[0], segment[1]):
                    del valid_intersect_data[x]
        hand_face_continuous_segments = new_segments

        # # No intersection
        # valid_intersect_data = no_intersect_data
        #
        # continuous_segments = []
        # for i in valid_intersect_data.keys():
        #     if len(continuous_segments) == 0:
        #         continuous_segments.append([i, i + 1])
        #     else:
        #         if continuous_segments[-1][1] == i:
        #             continuous_segments[-1][1] += 1
        #         else:
        #             continuous_segments.append([i, i + 1])
        #
        # new_segments = []
        # for segment in continuous_segments:
        #     if segment[1] - segment[0] >= (min_length + cutoff * 2):
        #         new_segments.append([segment[0] + cutoff, segment[1] - cutoff])
        #         if cutoff != 0:
        #             for x in range(segment[0], segment[0] + cutoff):
        #                 del valid_intersect_data[x]
        #             for x in range(segment[1] - cutoff, segment[1]):
        #                 del valid_intersect_data[x]
        #     else:
        #         for x in range(segment[0], segment[1]):
        #             del valid_intersect_data[x]
        # no_continuous_segments = new_segments

        return hand_arm_continuous_segments, hand_leg_continuous_segments, hand_face_continuous_segments

    def export_elan_portal(self):
        portal = ElanPortal()
        hand_arm_continuous_segments, hand_leg_continuous_segments, hand_face_continuous_segments \
            = self.compute_hand_intersection(cutoff=10, min_length=20)
        cap = cv2.VideoCapture(self.video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        print('fps:', fps)

        def transfer_to_secs(segments):
            return (np.array(segments) / fps).tolist()

        hand_arm_continuous_segments = transfer_to_secs(hand_arm_continuous_segments)
        hand_leg_continuous_segments = transfer_to_secs(hand_leg_continuous_segments)
        hand_face_continuous_segments = transfer_to_secs(hand_face_continuous_segments)

        if self.hand == 'right':
            portal.add_tier('Right Hand Location', 'on arm', hand_arm_continuous_segments)
            portal.add_tier('Right Hand Location', 'on leg', hand_leg_continuous_segments)
            portal.add_tier('Right Hand Location', 'on face', hand_face_continuous_segments)
        else:
            portal.add_tier('Left Hand Location', 'on arm', hand_arm_continuous_segments)
            portal.add_tier('Left Hand Location', 'on leg', hand_leg_continuous_segments)
            portal.add_tier('Left Hand Location', 'on face', hand_face_continuous_segments)

        portal.export('test.txt')
