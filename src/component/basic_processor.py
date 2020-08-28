import os
import json
import cv2
import math
import numpy as np
import pandas as pd
from utility.base_config import *
from utility.colors import *
from utility.rectangle import Rectangle
from utility.label_parser import Label_Parser


class BasicProcessor():

    def __init__(self, name, path_data, batch_data=None):
        self.name = name
        if path_data is not None:
            # single video processing
            self.path_data = path_data
            self.video_path = path_data['video']
            self.openpose_output_path = path_data['openpose_data']
            self.openface_output_file = path_data['openface_data']
            self.processed_file = path_data['processed_data']
            self.processed_smooth_file = path_data['processed_data_smooth']
            self.participant_id = path_data['participant_id']
            self.session_id = path_data['session_id']
        else:
            # general processing
            self.batch_data = batch_data

    def get_label(self):
        '''
        This function returns label
        :return:
        '''
        parser = Label_Parser(self.path_data['fidgeting_label'])
        return {
            'fidgeting': parser.parse(),
        }

    # Painting Functions

    def paint_point(self, img, point, color=(0, 0, 255)):
        '''
        This function paints point group
        :param img:
        :param point: shape of (-1, 2)
        :param color:
        :return:
        '''
        cv2.circle(img, (int(point[0]), int(point[1])), 1, color, -1)
        return img

    def paint_text(self, img, text, point, font_size=3, color=(0, 205, 193)):
        '''
        This function add text to a point
        :param img:
        :param text:
        :param point:
        :param color:
        :return:
        '''
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, text, (int(point[0]), int(point[1])), font, font_size, color, math.ceil(font_size * 2))
        return img

    def paint_line(self, img, point1, point2, color=(0, 205, 193)):
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

    def paint_rectangle_to_points(self, img, points, color=(0, 255, 0)):
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

    def check_overlap(self, points_1, points_2, tolerance=0):
        '''
        This function check overlapping of point groups
        :param points_1:
        :param points_2:
        :return: intersecting area (rect), None if not intersecting
        '''
        a = Rectangle(np.min(points_1, axis=0).astype(int)[0] - tolerance,
                      np.min(points_1, axis=0).astype(int)[1] - tolerance,
                      np.max(points_1, axis=0).astype(int)[0] + tolerance,
                      np.max(points_1, axis=0).astype(int)[1] + tolerance)
        b = Rectangle(np.min(points_2, axis=0).astype(int)[0] - tolerance,
                      np.min(points_2, axis=0).astype(int)[1] - tolerance,
                      np.max(points_2, axis=0).astype(int)[0] + tolerance,
                      np.max(points_2, axis=0).astype(int)[1] + tolerance)
        intersect = a & b
        return intersect

    def check_rect_overlap(self, points_1, points_2, tolerance=0):
        '''
        This function check overlapping of point groups
        :param points_1:
        :param points_2:
        :return: intersecting area (rect), None if not intersecting
        '''
        a = Rectangle(np.min(points_1, axis=0).astype(int)[0] - tolerance,
                      np.min(points_1, axis=0).astype(int)[1] - tolerance,
                      np.max(points_1, axis=0).astype(int)[0] + tolerance,
                      np.max(points_1, axis=0).astype(int)[1] + tolerance)
        b = Rectangle(np.min(points_2, axis=0).astype(int)[0] - tolerance,
                      np.min(points_2, axis=0).astype(int)[1] - tolerance,
                      np.max(points_2, axis=0).astype(int)[0] + tolerance,
                      np.max(points_2, axis=0).astype(int)[1] + tolerance)
        intersect = a & b
        return intersect

    def analyse_sequence(self, X):
        FFT = np.abs(np.fft.fft(X, axis=0))
        t = np.arange(X.shape[0])
        freq = np.fft.fftfreq(t.shape[-1])
        frate = 25
        freq_in_hertz = np.abs(freq * frate)
        FFT[(freq_in_hertz > 2.5) | (freq_in_hertz < 0.5), :] = 0

        avg_fft = np.sum(FFT) / np.count_nonzero(FFT)

        avg_std = np.mean(np.std(X, axis=0))
        return avg_fft, avg_std

    def analyse_sequence_new(self, X):
        '''
        This function analyses sequence data X
            FFT = FFT[(freq_in_hertz > 2.5) | (freq_in_hertz < 0.5), :]
            filters FFT data we are interested in
        :param X:  to be analysed
        :return: FFT, STD, MEAN
        '''
        FFT = np.abs(np.fft.fft(X, axis=0))
        t = np.arange(X.shape[0])
        freq = np.fft.fftfreq(t.shape[-1])
        frate = 25
        freq_in_hertz = np.abs(freq * frate)
        FFT = FFT[(freq_in_hertz > 2.5) | (freq_in_hertz < 0.5), :]
        FFT = FFT[:int(FFT.shape[0] / 2), :]

        STD = np.std(X, axis=0)
        MEAN = np.mean(X, axis=0)

        return FFT, STD, MEAN

    def get_first_derivative(self, X_0th):
        time = 0.04
        X_1st = np.zeros((X_0th.shape[0] - 1, X_0th.shape[1]))
        for i in range(X_0th.shape[0] - 1):
            X_1st[i] = (X_0th[i + 1] - X_0th[i]) / time
        return X_1st

    def transfer_to_array(self, segments):
        X = np.zeros((np.load(self.processed_file).shape[0], 1))
        for segment in segments:
            X[segment[0]:segment[1], :] = 1
        return X

    def transfer_to_segments(self, data, min_length=100, cutoff=0):
        '''
        This function transfers an array to segment list
        :param data:
        :return:
        '''
        continuous_segments = []
        for i in range(data.shape[0]):
            if data[i, 0] != 0:
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

        continuous_segments = new_segments

        return continuous_segments

    def show_frames(self, starting, ending, label_data=None, save_video=False):
        '''
        Show specific frames of a video
        :param starting: int
        :param ending: int
        :return:
        '''
        cap = cv2.VideoCapture(self.video_path)
        data = np.load(self.processed_file)

        # try:
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(length)
        cap.set(1, starting)

        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        print(cap.get(cv2.CAP_PROP_FRAME_HEIGHT), '~~~', cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        if save_video:
            out = cv2.VideoWriter('output.avi', fourcc, cap.get(cv2.CAP_PROP_FPS), (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

        t = starting
        print(data.shape)
        while (t < ending):
            # print(t)
            ret, frame = cap.read()
            # Display all the data points

            for i in range(25):
                frame = self.paint_point(frame, [data[t, i * 2], data[t, i * 2 + 1]])
            for i in range(25, 95):
                frame = self.paint_point(frame, [data[t, i * 2], data[t, i * 2 + 1]], color=COLOR_BLUE)
            # for i in range(95, 116):
            #     frame = self.paint_point(frame, [data[t, i * 2], data[t, i * 2 + 1]], color=COLOR_GREEN)
            # for i in range(116, 137):
            #     frame = self.paint_point(frame, [data[t, i * 2], data[t, i * 2 + 1]], color=COLOR_YELLOW)

            left_hand_data = data[t, 194:232].reshape(-1, 2)
            right_hand_data = data[t, 236:274].reshape(-1, 2)
            face_data = data[t, 50:190].reshape(-1, 2)

            # frame = self.paint_rectangle_to_points(frame, left_hand_data, color=COLOR_GREEN)
            # frame = self.paint_rectangle_to_points(frame, right_hand_data, color=COLOR_YELLOW)

            # Check hands overlapping
            intersect = self.check_overlap(left_hand_data, right_hand_data)
            if intersect is not None:
                x1, y1, x2, y2 = intersect.get_cordinate()

                # for i in range(left_hand_data.shape[0]):
                #     frame = self.paint_point(frame, left_hand_data[i, :], color=COLOR_YELLOW)
                #     frame = self.paint_point(frame, right_hand_data[i, :], color=COLOR_BLUE)
                # frame = self.paint_rectangle_to_points(frame,
                #                                       np.vstack((left_hand_data, right_hand_data)),
                #                                       color=COLOR_GREEN)



            # Check hand-face overlapping
            # intersect = self.check_overlap(left_hand_data, face_data)
            # if intersect is not None:
            #     x1, y1, x2, y2 = intersect.get_cordinate()
            #     cv2.rectangle(frame,
            #                   (x1, y1),
            #                   (x2, y2),
            #                   COLOR_GREEN,
            #                   2)
            #
            # intersect = self.check_overlap(right_hand_data, face_data)
            # if intersect is not None:
            #     x1, y1, x2, y2 = intersect.get_cordinate()
            #     cv2.rectangle(frame,
            #                   (x1, y1),
            #                   (x2, y2),
            #                   COLOR_BLUE,
            #                   2)
            if label_data is not None:
                if label_data[t, 0] == 1:
                    cv2.rectangle(frame,
                                  (500, 200),
                                  (550, 250),
                                  COLOR_RED,
                                  2)
                else:
                    cv2.rectangle(frame,
                                  (500, 200),
                                  (550, 250),
                                  COLOR_GREEN,
                                  2)
            if save_video:
                out.write(frame)
            else:
                cv2.imshow('frame', frame)

            if cv2.waitKey(40) & 0xFF == ord('q'):
                break
            # if t == starting:
            #     input()
            t += 1
        # except Exception as e:
        # print(e)

        cap.release()
        if save_video:
            out.release()
        # cv2.destroyAllWindows()

