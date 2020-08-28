import os
import json
import cv2
from utility.base_config import *
from scipy.signal import savgol_filter
from utility.colors import *
from utility.line import Line
from utility.const import BODY_CONNECTION
from utility.elan_portal import ElanPortal

from component.basic_processor import BasicProcessor


class LegLocationAnalyser(BasicProcessor):

    def __init__(self, name, path_data):
        BasicProcessor.__init__(self, name, path_data, None)

    # Plot Leg
    def plot_leg_joints(self):
        cap = cv2.VideoCapture(self.video_path)
        data = np.load(self.processed_file)
        t = 0
        while (cap.isOpened()):
            ret, frame = cap.read()

            for i in range(0, 25):
                frame = self.paint_point(frame, [data[t, 2 * i], data[t, 2 * i + 1]], color=COLOR_YELLOW)
                frame = self.paint_text(frame, str(i), [data[t, 2 * i], data[t, 2 * i + 1]],
                                        font_size=0.5, color=COLOR_YELLOW)

            for connection in BODY_CONNECTION[12:]:
                point1 = connection[0]
                point2 = connection[1]
                frame = self.paint_line(frame,
                                        [data[t, 2 * point1], data[t, 2 * point1 + 1]],
                                        [data[t, 2 * point2], data[t, 2 * point2 + 1]],
                                        color=COLOR_YELLOW)

            cv2.imshow('frame', frame)
            if cv2.waitKey(40) & 0xFF == ord('q'):
                break
            # input()
            t += 1

        cap.release()
        cv2.destroyAllWindows()

    # Leg Intersection
    def compute_leg_intersection(self, min_length=100, cutoff=0):
        cap = cv2.VideoCapture(self.video_path, 0)
        data = np.load(self.processed_smooth_file)

        intersect_data_raw = []
        intersect_data = {}

        t = 0
        while (t < np.shape(data)[0]):
            ret, frame = cap.read()
            print('progress', t / data.shape[0], end='\r')
            intersection = False

            # Criterion 1: Toe Position
            rtoe = [data[t, 2 * 22], data[t, 2 * 22 + 1]]
            ltoe = [data[t, 2 * 19], data[t, 2 * 19 + 1]]

            if rtoe[0] > ltoe[0]:
                intersection = True

            # Criterion 2: Intersection
            rlowerleg = Line((data[t, 2 * 10], data[t, 2 * 10 + 1]), (data[t, 2 * 11], data[t, 2 * 11 + 1]))
            llowerleg = Line((data[t, 2 * 13], data[t, 2 * 13 + 1]), (data[t, 2 * 14], data[t, 2 * 14 + 1]))
            rthigh = Line((data[t, 2 * 9], data[t, 2 * 9 + 1]), (data[t, 2 * 10], data[t, 2 * 10 + 1]))
            lthigh = Line((data[t, 2 * 12], data[t, 2 * 12 + 1]), (data[t, 2 * 13], data[t, 2 * 13 + 1]))
            rfoot = Line((data[t, 2 * 11], data[t, 2 * 11 + 1]), (data[t, 2 * 22], data[t, 2 * 22 + 1]))
            lfoot = Line((data[t, 2 * 14], data[t, 2 * 14 + 1]), (data[t, 2 * 19], data[t, 2 * 19 + 1]))
            rheel = Line((data[t, 2 * 11], data[t, 2 * 11 + 1]), (data[t, 2 * 24], data[t, 2 * 24 + 1]))
            lheel = Line((data[t, 2 * 14], data[t, 2 * 14 + 1]), (data[t, 2 * 21], data[t, 2 * 21 + 1]))
            rtoe = Line((data[t, 2 * 22], data[t, 2 * 22 + 1]), (data[t, 2 * 23], data[t, 2 * 23 + 1]))
            ltoe = Line((data[t, 2 * 19], data[t, 2 * 19 + 1]), (data[t, 2 * 20], data[t, 2 * 20 + 1]))

            right_list = [rlowerleg, rthigh, rfoot, rheel, rtoe]
            left_list = [llowerleg, lthigh, lfoot, lheel, ltoe]

            for right_joint in right_list:
                for left_joint in left_list:
                    lr_intersection = Line.line_intersection(right_joint, left_joint)

                    if lr_intersection[0] is not None:
                        # # Right Thigh
                        # frame = self.paint_line(frame,
                        #                         [right_joint.x1, right_joint.y1],
                        #                         [right_joint.x2, right_joint.y2],
                        #                         color=COLOR_YELLOW)
                        #
                        # # Left Thigh
                        # frame = self.paint_line(frame,
                        #                         [left_joint.x1, left_joint.y1],
                        #                         [left_joint.x2, left_joint.y2],
                        #                         color=COLOR_YELLOW)
                        #
                        # frame = self.paint_point(frame, lr_intersection, color=COLOR_YELLOW)

                        intersection = True

            # Save Raw Data
            if intersection == True:
                intersect_data_raw.append(1)
            else:
                intersect_data_raw.append(0)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            # if t == 0:
            # input()
            t += 1

        cap.release()

        # Smoothing
        intersect_data_raw = savgol_filter(intersect_data_raw, 21, 2)
        intersect_data_raw = [int(round(i)) for i in intersect_data_raw]
        # print(intersect_data_raw)

        for t in range(len(intersect_data_raw)):
            if intersect_data_raw[t] == 1:
                intersect_data[t] = None

        # Visualize Joint
        # cap = cv2.VideoCapture(self.video_path, 0)
        # t = 0
        # while (t < np.shape(data)[0]):
        #     ret, frame = cap.read()
        #
        #     if t in intersect_data:
        #         for connection in BODY_CONNECTION[12:]:
        #             point1 = connection[0]
        #             point2 = connection[1]
        #             frame = self.paint_line(frame,
        #                                     [data[t, 2 * point1], data[t, 2 * point1 + 1]],
        #                                     [data[t, 2 * point2], data[t, 2 * point2 + 1]],
        #                                     color=COLOR_YELLOW)
        #
        #     cv2.imshow('frame', frame)
        #     if cv2.waitKey(1) & 0xFF == ord('q'):
        #         break
        #     t += 1

        cap.release()
        cv2.destroyAllWindows()

        # Continuous segments
        continuous_segments = []
        for i in intersect_data.keys():
            if len(continuous_segments) == 0:
                continuous_segments.append([i, i + 1])
            else:
                if continuous_segments[-1][1] == i:
                    continuous_segments[-1][1] += 1
                else:
                    continuous_segments.append([i, i + 1])

        # min length of stationary rectangle
        new_segments = []
        for segment in continuous_segments:
            if segment[1] - segment[0] >= (min_length + cutoff * 2):
                new_segments.append([segment[0] + cutoff, segment[1] - cutoff])
                if cutoff != 0:
                    for x in range(segment[0], segment[0] + cutoff):
                        del intersect_data[x]
                    for x in range(segment[1] - cutoff, segment[1]):
                        del intersect_data[x]
            else:
                for x in range(segment[0], segment[1]):
                    del intersect_data[x]
        continuous_segments = new_segments

        return continuous_segments, intersect_data

    def export_elan_portal(self):
        portal = ElanPortal()
        continuous_segments, valid_intersect_data = self.compute_leg_intersection(cutoff=10, min_length=20)
        cap = cv2.VideoCapture(self.video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        print('fps:', fps)

        def transfer_to_secs(segments):
            return (np.array(segments) / fps).tolist()

        continuous_segments = transfer_to_secs(continuous_segments)
        portal.add_tier('Leg Location', 'on leg', continuous_segments)
        portal.export('test.txt')