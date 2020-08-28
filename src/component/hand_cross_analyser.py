import os
import json
import cv2
import math
import numpy as np
import pandas as pd
from utility.base_config import *
from scipy.signal import savgol_filter
from utility.colors import *
from utility.decompose_string import decompose_string, decompose_string_hand

from component.basic_processor import BasicProcessor

class HandCrossAnalyser(BasicProcessor):

    def __init__(self, name, path_data):
        BasicProcessor.__init__(self, name, path_data, None)

    def compute_stationary_rectangles(self, min_length=100, cutoff=0):
        '''
        This function compute stationary rectangles out of all the rectangles detected in time series.
        :return:
        continuous_segments: list of segments
        valid_intersect_data: stored rectangle coordinates for each valid segments
        '''
        cap = cv2.VideoCapture(self.video_path)
        data = np.load(self.processed_file)
        intersect_data = {}
        # try:
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(length)
        t = 0
        while (t < data.shape[0]):
            print('progress', t / data.shape[0], end='\r')
            # Display all the data points
            '''
            for i in range(25):
                frame = self.paint_point(frame, [data[t, i * 2], data[t, i * 2 + 1]])
            for i in range(25, 95):
                frame = self.paint_point(frame, [data[t, i * 2], data[t, i * 2 + 1]], color=COLOR_BLUE)
            for i in range(95, 116):
                frame = self.paint_point(frame, [data[t, i * 2], data[t, i * 2 + 1]], color=COLOR_GREEN)
            for i in range(116, 137):
                frame = self.paint_point(frame, [data[t, i * 2], data[t, i * 2 + 1]], color=COLOR_YELLOW)
            #'''
            left_hand_data = data[t, 194:232].reshape(-1, 2)
            right_hand_data = data[t, 236:274].reshape(-1, 2)
            face_data = data[t, 50:190].reshape(-1, 2)

            # frame = self.paint_rectangle_to_points(frame, left_hand_data, color=COLOR_GREEN)
            # frame = self.paint_rectangle_to_points(frame, right_hand_data, color=COLOR_YELLOW)

            # Check hands overlapping
            intersect = self.check_overlap(left_hand_data, right_hand_data, tolerance=5)
            if intersect is not None:
                points = np.vstack((left_hand_data, right_hand_data))
                cordinates = list(np.min(points, axis=0).astype(int)) + list(np.max(points, axis=0).astype(int))
                intersect_data[t] = cordinates
            t += 1
        # except Exception as e:
        # print(e)

        cap.release()

        # compute continuous segment
        continuous_segments = []
        for i in intersect_data.keys():
            if len(continuous_segments) == 0:
                continuous_segments.append([i, i + 1])
            else:
                if continuous_segments[-1][1] == i:
                    continuous_segments[-1][1] += 1
                else:
                    continuous_segments.append([i, i + 1])

        # validate stationarity
        # valid_intersect_data = {}
        # for session in continuous_segments:
        #     starting_time = session[0]
        #     ending_time = session[1]
        #     for i in range(starting_time + 1, ending_time):
        #         previous_rect = np.array(intersect_data[i - 1])
        #         current_rect = np.array(intersect_data[i])
        #         change = np.sum(np.power(current_rect - previous_rect, 2))
        #         if change <= 6:
        #             # rectangle stationary
        #             valid_intersect_data[i] = current_rect

        valid_intersect_data = intersect_data

        # recompute continuous segments
        continuous_segments = []
        for i in valid_intersect_data.keys():
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
                        del valid_intersect_data[x]
                    for x in range(segment[1] - cutoff, segment[1]):
                        del valid_intersect_data[x]
            else:
                for x in range(segment[0], segment[1]):
                    del valid_intersect_data[x]
        continuous_segments = new_segments

        # print(continuous_segments)
        # print(intersect_data.keys())
        # print(valid_intersect_data.keys())
        print('\n')
        return continuous_segments, valid_intersect_data

    def compute_static_hands_without_crossing(self, hand='left', min_length=100, cutoff=0):

        # read hand cross data
        hand_cross_segments, hand_cross_intersect_data = self.compute_stationary_rectangles(min_length=20)
        data = np.load(self.processed_smooth_file)
        cap = cv2.VideoCapture(self.video_path)

        static_data = {}
        # try:
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(length)
        t = 1
        while (t < data.shape[0]):
            print('progress', t / data.shape[0], end='\r')
            if hand == 'left':
                hand_data = data[t, 194:232].reshape(-1, 2)
                previous_hand_data = data[t - 1, 194:232].reshape(-1, 2)
            else:
                hand_data = data[t, 236:274].reshape(-1, 2)
                previous_hand_data = data[t - 1, 236:274].reshape(-1, 2)

            ret, frame = cap.read()

            # without hand cross segments
            if t not in hand_cross_intersect_data.keys():
                difference = hand_data - previous_hand_data
                difference = np.mean(np.power(difference, 2))
                # print(difference)
                if difference < 0.7:
                    frame = self.paint_rectangle_to_points(frame, hand_data, color=COLOR_YELLOW)
                    points = hand_data
                    cordinates = list(np.min(points, axis=0).astype(int)) + list(np.max(points, axis=0).astype(int))
                    static_data[t] = cordinates

            # cv2.imshow('frame', frame)
            k = cv2.waitKey(40) & 0xff
            if k == 27:
                break

            t += 1
            # input()
        cv2.destroyAllWindows()
        cap.release()

        # smoothing the data

        smooth_array = np.zeros((data.shape[0], 1))
        for t in static_data.keys():
            smooth_array[t, :] = 1
        print(smooth_array.shape)
        y = savgol_filter(np.ravel(smooth_array), 11, 3)
        smooth_array = np.array(y).reshape((-1, 1))
        smooth_array[smooth_array >= 0.5] = 1
        smooth_array[smooth_array < 0.5] = 0

        # compute continuous segment
        continuous_segments = []
        for i in range(data.shape[0]):
            if smooth_array[i, 0] == 1:
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

    def compute_static_and_rhythmic_with_hand_cross(self):
        hand_cross_segments, hand_cross_intersect_data = self.compute_stationary_rectangles(min_length=20)

        label_data = json.load(open('optical_flow_results_for_export_label.json', 'r'))
        try:
            label_data = label_data[str(self.participant_id)][str(self.session_id)]
        except Exception as e:
            print('no hands playing data...')
            label_data = {}

        window_size = 100
        window_step = 50
        FFT_thres = 30
        STD_thres = 8
        data = np.load(self.processed_file)

        # generate label array
        label_array = np.zeros((data.shape[0], 1))
        label_centroid = {}

        for segment in label_data.keys():
            starting = int(segment.split(',')[0])
            ending = int(segment.split(',')[1])
            centroid = int(math.floor((starting + ending) / 2))
            p = (centroid, label_data[segment][0], label_data[segment][1])
            label_centroid[centroid] = label_data[segment]

        print('preprocessing label data')
        for t in range(data.shape[0]):
            related_centroids = [(i, label_centroid[i])
                                 for i in range(int(t - 0.5 * window_size), int(t + 0.5 * window_size))
                                 if i in label_centroid.keys()]

            if len(related_centroids) == 0:
                continue
            if len(related_centroids) == 1:
                closest_centroid = related_centroids[0]
            else:
                id_1 = related_centroids[0][0]
                id_2 = related_centroids[1][0]
                if abs(id_1 - t) < abs(id_2 - t):
                    closest_centroid = related_centroids[0]
                else:
                    closest_centroid = related_centroids[1]
            avg_fft = closest_centroid[1][0]
            avg_std = closest_centroid[1][1]

            if avg_fft >= FFT_thres and avg_std >= STD_thres:
                label_array[t, :] = 3  # dynamic + rhythmic
            elif avg_fft >= FFT_thres and avg_std < STD_thres:
                label_array[t, :] = 2  # rhythmic
            elif avg_fft < FFT_thres and avg_std >= STD_thres:
                label_array[t, :] = 1  # dynamic
            elif avg_fft < FFT_thres and avg_std < STD_thres:
                label_array[t, :] = 0  # static

        for x in range(4):
            # compute continuous segment
            continuous_segments = []
            for i in range(data.shape[0]):
                if i in hand_cross_intersect_data.keys():
                    if label_array[i, 0] == x:
                        if len(continuous_segments) == 0:
                            continuous_segments.append([i, i + 1])
                        else:
                            if continuous_segments[-1][1] == i:
                                continuous_segments[-1][1] += 1
                            else:
                                continuous_segments.append([i, i + 1])
            if x == 0:
                static_segments = continuous_segments
            if x == 1:
                dynamic_segments = continuous_segments
            if x == 2:
                rhythmic_segments = continuous_segments
            if x == 3:
                dynamic_rythmic_segments = continuous_segments
        return static_segments, dynamic_segments, rhythmic_segments, dynamic_rythmic_segments


    def analyse_hand_cross_optical_flow(self):
        data = {}
        for root, dirs, files in os.walk(os.path.join(DATA_FOLDER, 'hand_cross_analysis_optical_flow')):
            for file in files:
                if '.npy' in file:
                    data[file] = np.load(os.path.join(root, file))
                    if data[file].shape[0] == 0:
                        print(file, data[file].shape)

        label_data = {}
        from keras.models import load_model
        model = load_model(
            os.path.join(DATA_FOLDER, 'pre-trained', 'hierarchical_DNN.h5')
        )
        for file in data.keys():
            participant_id, session_id, starting, ending = decompose_string(file)
            sub_data = data[file]
            if sub_data.shape[0] != 100:
                continue
            FFT, STD, MEAN = self.analyse_sequence_new(self.get_first_derivative(sub_data))

            FFT = np.mean(FFT, axis=1)
            STD = STD  # np.mean(STD)
            MEAN = MEAN  # np.mean(MEAN, axis=0)

            single_x = [
                FFT.reshape((1, -1)), STD.reshape((1, -1)), MEAN.reshape((1, -1))
            ]
            label = int(np.argmax(model.predict(single_x), axis=1)[0])
            print(label)
            label_data.setdefault(participant_id, {}).setdefault(session_id, {})['{},{}'.format(starting, ending)] = label

        json.dump(label_data, open(
            os.path.join(DATA_FOLDER, 'hand_cross_analysis_optical_flow', 'optical_flow_result.json'),
            'w'))
        print('saving completed.')


    def analyse_hand_action_optical_flow(self):
        data = {}
        for root, dirs, files in os.walk(os.path.join(DATA_FOLDER, 'hand_action_analysis_optical_flow')):
            for file in files:
                if '.npy' in file:
                    data[file] = np.load(os.path.join(root, file))
                    if data[file].shape[0] == 0:
                        print(file, data[file].shape)

        label_data = {}
        from keras.models import load_model
        model = load_model(
            os.path.join(DATA_FOLDER, 'pre-trained', 'hierarchical_DNN_hand.h5')
        )
        for file in data.keys():
            participant_id, session_id, starting, ending, hand = decompose_string_hand(file)
            sub_data = data[file]
            if sub_data.shape[0] != 100:
                continue
            FFT, STD, MEAN = self.analyse_sequence_new(self.get_first_derivative(sub_data))

            FFT = np.mean(FFT, axis=1)
            STD = STD  # np.mean(STD)
            MEAN = MEAN  # np.mean(MEAN, axis=0)

            single_x = [
                FFT.reshape((1, -1)), STD.reshape((1, -1)), MEAN.reshape((1, -1))
            ]
            label = int(np.argmax(model.predict(single_x), axis=1)[0])
            print(label)
            label_data.setdefault(participant_id, {}).setdefault(session_id, {}).setdefault(hand, {})['{},{}'.format(starting, ending)] = label

        json.dump(label_data, open(
            os.path.join(DATA_FOLDER, 'hand_action_analysis_optical_flow', 'optical_flow_result.json'),
            'w'))
        print('saving completed.')
