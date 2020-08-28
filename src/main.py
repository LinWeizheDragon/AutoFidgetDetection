from utility.base_config import *
from pprint import pprint

import os
import json
import cv2
import math
import random
import numpy as np
import pandas as pd
import pickle
from scipy.signal import savgol_filter
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold
from utility.colors import *
from model.fidgeting_dnn import Fidgeting_DNN
import matplotlib.pyplot as plt
from utility.elan_portal import ElanPortal
from utility.decompose_string import decompose_string, decompose_string_hand
from utility.dirs import create_dirs
from component.basic_processor import BasicProcessor
from component.video_processor import VideoProcessor
from component.optical_flow_analyser import OpticalFlowAnalyser
from component.hand_cross_analyser import HandCrossAnalyser
from component.hand_location_analyser import HandLocationAnalyser
from component.leg_action_analyser import LegActionAnalyser
from component.leg_location_analyser import LegLocationAnalyser
from component.label_machine import LabelMachine



class MainPipeline(BasicProcessor):

    def __init__(self, name, path_data, batch_data=None):
        self.name = name
        if path_data is not None:
            # single video processing
            self.path_data = path_data
            self.video_path = path_data['video']
            self.openpose_output_path = path_data['openpose_data']
            self.openface_output_file = path_data['openface_data']
            self.processed_file = os.path.join(DATA_FOLDER, 'processed_data',
                                               path_data['file_format'] + '.npy')
            self.processed_smooth_file = os.path.join(DATA_FOLDER, 'processed_data_smooth',
                                                      path_data['file_format'] + '.npy')
            self.participant_id = path_data['participant_id']
            self.session_id = path_data['session_id']

        else:
            # general processing
            self.batch_data = batch_data

    def read_labels(self, label_file):
        cap = cv2.VideoCapture(self.video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        print('fps:', fps)

        portal = ElanPortal()
        portal.read(label_file, fps)
        print(portal.get_segments('Leg Action', 'static'))

        # hand location
        left_hand_on_hand = self.transfer_to_array(portal.get_segments('Left Hand Location', 'on hand'))
        right_hand_on_hand = self.transfer_to_array(portal.get_segments('Left Hand Location', 'on hand'))
        left_hand_on_leg = self.transfer_to_array(portal.get_segments('Left Hand Location', 'on leg'))
        right_hand_on_leg = self.transfer_to_array(portal.get_segments('Right Hand Location', 'on leg'))
        left_hand_on_arm = self.transfer_to_array(portal.get_segments('Left Hand Location', 'on arm'))
        right_hand_on_arm = self.transfer_to_array(portal.get_segments('Right Hand Location', 'on arm'))
        left_hand_on_face = self.transfer_to_array(portal.get_segments('Left Hand Location', 'on face'))
        right_hand_on_face = self.transfer_to_array(portal.get_segments('Right Hand Location', 'on face'))

        # leg location
        leg_on_leg = self.transfer_to_array(portal.get_segments('Leg Location', 'on leg'))
        leg_on_ground = self.transfer_to_array(portal.get_segments('Leg Location', 'on leg'))

        # action
        left_hand_static = self.transfer_to_array(portal.get_segments('Left Hand Action', 'static'))
        left_hand_rhythmic = self.transfer_to_array(portal.get_segments('Left Hand Action', 'rhythmic'))
        right_hand_static = self.transfer_to_array(portal.get_segments('Right Hand Action', 'static'))
        right_hand_rhythmic = self.transfer_to_array(portal.get_segments('Right Hand Action', 'rhythmic'))
        leg_static = self.transfer_to_array(portal.get_segments('Leg Action', 'static'))
        leg_rhythmic = self.transfer_to_array(portal.get_segments('Leg Action', 'rhythmic'))

        # copy left to right
        right_hand_static[(left_hand_on_hand == 1) & (left_hand_static == 1)] = 1
        right_hand_rhythmic[(left_hand_on_hand == 1) & (left_hand_rhythmic == 1)] = 1

        result = {
            'left_hand_on_hand': left_hand_on_hand,
            'right_hand_on_hand': right_hand_on_hand,
            'left_hand_on_leg': left_hand_on_leg,
            'right_hand_on_leg': right_hand_on_leg,
            'left_hand_on_arm': left_hand_on_arm,
            'right_hand_on_arm': right_hand_on_arm,
            'left_hand_on_face': left_hand_on_face,
            'right_hand_on_face': right_hand_on_face,
            'leg_on_leg': leg_on_leg,
            'leg_on_ground': leg_on_ground,
            'left_hand_static': left_hand_static,
            'left_hand_rhythmic': left_hand_rhythmic,
            'right_hand_static': right_hand_static,
            'right_hand_rhythmic': right_hand_rhythmic,
            'leg_static': leg_static,
            'leg_rhythmic': leg_rhythmic,
        }
        return result

    def export_elan_portal(self):
        cap = cv2.VideoCapture(self.video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        print('fps:', fps)

        portal = ElanPortal()

        print('computing leg intersection...')
        instance_leg = LegLocationAnalyser(self.name, self.path_data)
        leg_continuous_segments, intersect_data = instance_leg.compute_leg_intersection(cutoff=0, min_length=20)

        print('computing left hand location...')
        instance_left_hand = HandLocationAnalyser(self.name, self.path_data, hand='left')
        left_hand_arm_continuous_segments, left_hand_leg_continuous_segments, left_hand_face_continuous_segments = \
            instance_left_hand.compute_hand_intersection(cutoff=0, min_length=80)

        print('computing right hand location...')
        instance_right_hand = HandLocationAnalyser(self.name, self.path_data, hand='right')
        right_hand_arm_continuous_segments, right_hand_leg_continuous_segments, right_hand_face_continuous_segments = \
            instance_right_hand.compute_hand_intersection(cutoff=0, min_length=80)

        print('computing hand cross')
        instance_hand_cross = HandCrossAnalyser(self.name, self.path_data)
        continuous_segments, valid_intersect_data = instance_hand_cross.compute_stationary_rectangles(cutoff=0,
                                                                                                      min_length=20)

        print('computing left hand static')
        left_static_segments = instance_hand_cross.compute_static_hands_without_crossing('left')
        print('computing right hand static')
        right_static_segments = instance_hand_cross.compute_static_hands_without_crossing('right')
        static_segments, dynamic_segments, rhythmic_segments, dynamic_rythmic_segments = instance_hand_cross.compute_static_and_rhythmic_with_hand_cross()

        print('computing leg action')
        instance_leg_action = LegActionAnalyser(self.name, self.path_data)
        foot_static_segments, foot_dynamic_segments, foot_rhythmic_segments, foot_dynamic_rythmic_segments = instance_leg_action.compute_static_and_rhythmic_feet()

        def transfer_to_secs(segments):
            return (np.array(segments) / fps).tolist()

        continuous_segments = transfer_to_secs(continuous_segments)
        left_static_segments = transfer_to_secs(left_static_segments)
        right_static_segments = transfer_to_secs(right_static_segments)
        static_segments = transfer_to_secs(static_segments)
        dynamic_segments = transfer_to_secs(dynamic_segments)
        rhythmic_segments = transfer_to_secs(rhythmic_segments)
        dynamic_rythmic_segments = transfer_to_secs(dynamic_rythmic_segments)
        foot_static_segments = transfer_to_secs(foot_static_segments)
        foot_dynamic_rythmic_segments = transfer_to_secs(foot_dynamic_rythmic_segments)
        foot_rhythmic_segments = transfer_to_secs(foot_rhythmic_segments)
        left_hand_arm_continuous_segments = transfer_to_secs(left_hand_arm_continuous_segments)
        left_hand_leg_continuous_segments = transfer_to_secs(left_hand_leg_continuous_segments)
        left_hand_face_continuous_segments = transfer_to_secs(left_hand_face_continuous_segments)
        right_hand_arm_continuous_segments = transfer_to_secs(right_hand_arm_continuous_segments)
        right_hand_leg_continuous_segments = transfer_to_secs(right_hand_leg_continuous_segments)
        right_hand_face_continuous_segments = transfer_to_secs(right_hand_face_continuous_segments)
        leg_continuous_segments = transfer_to_secs(leg_continuous_segments)

        # No hand cross event
        portal.add_tier('Left Hand Action', 'static', left_static_segments)
        portal.add_tier('Left Hand Location', 'on arm', left_hand_arm_continuous_segments)
        portal.add_tier('Left Hand Location', 'on leg', left_hand_leg_continuous_segments)
        portal.add_tier('Left Hand Location', 'on face', left_hand_face_continuous_segments)

        portal.add_tier('Right Hand Action', 'static', right_static_segments)
        portal.add_tier('Right Hand Location', 'on arm', right_hand_arm_continuous_segments)
        portal.add_tier('Right Hand Location', 'on leg', right_hand_leg_continuous_segments)
        portal.add_tier('Right Hand Location', 'on face', right_hand_face_continuous_segments)

        # Hand cross event
        portal.add_tier('Left Hand Location', 'on hand', continuous_segments)
        portal.add_tier('Left Hand Action', 'static', static_segments)
        portal.add_tier('Left Hand Action', 'rhythmic', rhythmic_segments)

        # Leg Event
        portal.add_tier('Leg Action', 'static', foot_static_segments)
        portal.add_tier('Leg Action', 'rhythmic', foot_dynamic_rythmic_segments)
        portal.add_tier('Leg Action', 'rhythmic', foot_rhythmic_segments)
        portal.add_tier('Leg Location', 'on leg', leg_continuous_segments)

        portal.export(os.path.join(DATA_FOLDER, 'label', 'generated', '{}_{}.txt'.format(
            self.participant_id, self.session_id
        )))

    def generate_hand_cross_slice(self):
        '''
        generate raw hand cross slices
        :return:
        '''
        instance_hand_cross_analyser = HandCrossAnalyser(self.name, self.path_data)
        continuous_segments, valid_intersect_data = instance_hand_cross_analyser.compute_stationary_rectangles(
            cutoff=0, min_length=20)
        print(continuous_segments)
        # best_rects = {}
        # for segment in continuous_segments:
        #     # find largest rectangle for each segment
        #     rects = [valid_intersect_data[i] for i in valid_intersect_data.keys()
        #              if i >= segment[0] and i < segment[1]]
        #     rects = np.array(rects)
        #     best_rect = np.hstack((np.min(rects, axis=0)[:2], np.max(rects, axis=0)[2:]))
        #     for i in range(segment[0], segment[1]):
        #         best_rects[i] = best_rect

        window_size = 100
        window_step = 50
        min_size = 100

        cap = cv2.VideoCapture(self.video_path)
        of_analyser = OpticalFlowAnalyser('test', self.path_data)
        init_points = np.hstack(
            (of_analyser.data[:, 194:232],
             of_analyser.data[:, 236:274])
        )

        participant_id = self.participant_id
        session_id = self.session_id

        for segment in continuous_segments:
            starting = int(segment[0])
            ending = int(segment[1])
            max_length = ending - starting
            print('-------->', starting, ending)
            for i in range(math.floor((max_length - window_size) / window_step) + 2):
                sub_starting, sub_ending = i * window_step, i * window_step + window_size
                sub_starting += starting
                sub_ending += starting

                if sub_ending > ending:
                    sub_ending = ending

                if sub_ending - sub_starting < min_size:
                    sub_starting = sub_ending - window_size

                    if sub_starting < starting:
                        # can't take at least one window
                        continue

                assert sub_ending - sub_starting <= window_size, 'sub slice must == to window size!'

                print('start slicing:', sub_starting, sub_ending)

                new_file_name = 'participant_video_{}_{}_<{}_{}>.npy'.format(
                    participant_id, session_id, sub_starting, sub_ending
                )

                # run optical flow instance
                optical_flow_data = of_analyser.run_optical_flow(cap, starting_time=sub_starting,
                                                                 ending_time=sub_ending, init_points=init_points,
                                                                 visualise=False)

                result = []

                for i in range(sub_starting, sub_ending):
                    # print(optical_flow_data[i].reshape((1, -1)).shape)
                    # print(i)
                    if i not in optical_flow_data:
                        print('Error detected, closing segment.')
                        break
                    result.append(optical_flow_data[i].reshape((1, -1)))

                result_path = os.path.join(DATA_FOLDER,
                                           'hand_cross_analysis_optical_flow',
                                           new_file_name
                                           )

                result_array = np.zeros((len(result), 38 * 2))
                for index, frame_data in enumerate(result):
                    result_array[index, :frame_data.shape[1]] = frame_data
                    result_array[index, frame_data.shape[1]:] = result_array[index - 1, frame_data.shape[1]:]
                # print(result_array)
                result = result_array
                # result = result.reshape((result.shape[0], result.shape[2]))
                create_dirs([os.path.split(result_path)[0]])
                np.save(result_path, result)
                FFT, STD, MEAN = self.analyse_sequence_new(self.get_first_derivative(result))
                print(np.mean(FFT, axis=0))
                print(np.mean(STD))
                print(np.mean(MEAN))


        cap.release()
        cv2.destroyAllWindows()

        print('saving completed.')

    def generate_leg_slice(self):
        '''
        generate raw leg slices
        :return:
        '''

        window_size = 100
        window_step = 50
        min_size = 100

        cap = cv2.VideoCapture(self.video_path)
        of_analyser = OpticalFlowAnalyser('test', self.path_data)
        init_points = np.hstack(
            (of_analyser.data[:, 44:50],
             # of_analyser.data[:, 20:22],
             of_analyser.data[:, 38:44],
             # of_analyser.data[:, 26:28],
             )
        )

        participant_id = self.participant_id
        session_id = self.session_id

        starting = 0
        ending = of_analyser.data.shape[0]
        max_length = ending - starting
        print('-------->', starting, ending)
        for i in range(math.floor((max_length - window_size) / window_step) + 2):
            sub_starting, sub_ending = i * window_step, i * window_step + window_size
            sub_starting += starting
            sub_ending += starting

            if sub_ending > ending:
                sub_ending = ending

            if sub_ending - sub_starting < min_size:
                sub_starting = sub_ending - window_size

                if sub_starting < starting:
                    # can't take at least one window
                    continue

            assert sub_ending - sub_starting <= window_size, 'sub slice must == to window size!'

            print('start slicing:', sub_starting, sub_ending)

            new_file_name = 'participant_video_{}_{}_<{}_{}>.npy'.format(
                participant_id, session_id, sub_starting, sub_ending
            )

            # run optical flow instance
            optical_flow_data = of_analyser.run_optical_flow(cap, starting_time=sub_starting, ending_time=sub_ending,
                                                             init_points=init_points,
                                                             visualise=False)

            result = []

            for i in range(sub_starting, sub_ending):
                # print(optical_flow_data[i].reshape((1, -1)).shape)
                # print(i)
                if i not in optical_flow_data:
                    print('Error detected, closing segment.')
                    break
                result.append(optical_flow_data[i].reshape((1, -1)))

            result_path = os.path.join(DATA_FOLDER,
                                       'leg_action_analysis_optical_flow',
                                       new_file_name
                                       )

            result_array = np.zeros((len(result), 6 * 2))
            for index, frame_data in enumerate(result):
                result_array[index, :frame_data.shape[1]] = frame_data
                result_array[index, frame_data.shape[1]:] = result_array[index - 1, frame_data.shape[1]:]
            # print(result_array)
            result = result_array
            # result = result.reshape((result.shape[0], result.shape[2]))

            FFT, STD, MEAN = self.analyse_sequence_new(self.get_first_derivative(result))
            print(np.mean(FFT, axis=0))
            print(np.mean(STD))
            print(np.mean(MEAN))
            #
            # input()
            create_dirs([os.path.split(result_path)[0]])
            np.save(result_path, result)

        cap.release()
        cv2.destroyAllWindows()

        print('saving completed.')

    def generate_hand_slice(self):
        '''
        generate raw hand cross slices
        :return:
        '''
        instance_hand_cross_analyser = HandCrossAnalyser(self.name, self.path_data)
        continuous_segments, hand_cross_intersect_data = instance_hand_cross_analyser.compute_stationary_rectangles(
            min_length=20, cutoff=0)
        print(continuous_segments)

        window_size = 100
        window_step = 50
        min_size = 100

        cap = cv2.VideoCapture(self.video_path)
        of_analyser = OpticalFlowAnalyser('test', self.path_data)

        no_cross_continuous_segments = []
        no_cross_list = [i for i in range(of_analyser.data.shape[0]) if i not in hand_cross_intersect_data.keys()]
        for i in no_cross_list:
            if len(no_cross_continuous_segments) == 0:
                no_cross_continuous_segments.append([i, i + 1])
            else:
                if no_cross_continuous_segments[-1][1] == i:
                    no_cross_continuous_segments[-1][1] += 1
                else:
                    no_cross_continuous_segments.append([i, i + 1])

        continuous_segments = no_cross_continuous_segments
        print(continuous_segments)

        # init_points = np.hstack(
        #     (of_analyser.data[:, 194:232],
        #      of_analyser.data[:, 236:274])
        # )
        init_points = of_analyser.data[:, 194:232]

        participant_id = self.participant_id
        session_id = self.session_id

        for hand in ['left', 'right']:
            if hand == 'left':
                init_points = of_analyser.data[:, 194:232]
            else:
                init_points = of_analyser.data[:, 236:274]


            for segment in continuous_segments:
                starting = int(segment[0])
                ending = int(segment[1])
                max_length = ending - starting
                print('-------->', starting, ending)
                for i in range(math.floor((max_length - window_size) / window_step) + 2):
                    sub_starting, sub_ending = i * window_step, i * window_step + window_size
                    sub_starting += starting
                    sub_ending += starting

                    if sub_ending > ending:
                        sub_ending = ending

                    if sub_ending - sub_starting < min_size:
                        sub_starting = sub_ending - window_size

                        if sub_starting < starting:
                            # can't take at least one window
                            continue

                    assert sub_ending - sub_starting <= window_size, 'sub slice must == to window size!'

                    print('start slicing:', sub_starting, sub_ending)

                    new_file_name = 'participant_video_{}_{}_<{}_{}>_{}.npy'.format(
                        participant_id, session_id, sub_starting, sub_ending, hand
                    )

                    # run optical flow instance
                    optical_flow_data = of_analyser.run_optical_flow(cap, starting_time=sub_starting,
                                                                     ending_time=sub_ending, init_points=init_points,
                                                                     visualise=False)

                    result = []

                    for i in range(sub_starting, sub_ending):
                        # print(optical_flow_data[i].reshape((1, -1)).shape)
                        # print(i)
                        if i not in optical_flow_data:
                            print('Error detected, closing segment.')
                            break
                        result.append(optical_flow_data[i].reshape((1, -1)))

                    result_path = os.path.join(DATA_FOLDER,
                                               'hand_action_analysis_optical_flow',
                                               new_file_name
                                               )

                    result_array = np.zeros((len(result), 19 * 2))
                    for index, frame_data in enumerate(result):
                        result_array[index, :frame_data.shape[1]] = frame_data
                        result_array[index, frame_data.shape[1]:] = result_array[index - 1, frame_data.shape[1]:]
                    # print(result_array)
                    result = result_array
                    # result = result.reshape((result.shape[0], result.shape[2]))

                    create_dirs([os.path.split(result_path)[0]])

                    np.save(result_path, result)
                    # FFT, STD, MEAN = self.analyse_sequence_new(self.get_first_derivative(result))
                    # print(np.mean(FFT, axis=0))
                    # print(np.mean(STD))
                    # print(np.mean(MEAN))
                    #
                    # input()

        cap.release()
        cv2.destroyAllWindows()

        print('saving completed.')

    def generate_hand_slice_from_label(self):
        '''
        generate hand cross slices from label TODO
        :return:
        '''
        # label_data = main_pipeline.read_labels(main_pipeline.label_files[1])
        # left_hand_on_hand = label_data['left_hand_on_hand']
        # continuous_segments = self.transfer_to_segments(left_hand_on_hand, min_length=20, cutoff=0)
        # print(continuous_segments)

        instance_hand_cross_analyser = HandCrossAnalyser(self.name, self.path_data)
        continuous_segments, hand_cross_intersect_data = instance_hand_cross_analyser.compute_stationary_rectangles(min_length=20, cutoff=0)
        print(continuous_segments)

        window_size = 100
        window_step = 50
        min_size = 100


        cap = cv2.VideoCapture(self.video_path)
        of_analyser = OpticalFlowAnalyser('test', self.path_data)

        no_cross_continuous_segments = []
        no_cross_list = [i for i in range(of_analyser.data.shape[0]) if i not in hand_cross_intersect_data.keys()]
        for i in no_cross_list:
            if len(no_cross_continuous_segments) == 0:
                no_cross_continuous_segments.append([i, i + 1])
            else:
                if no_cross_continuous_segments[-1][1] == i:
                    no_cross_continuous_segments[-1][1] += 1
                else:
                    no_cross_continuous_segments.append([i, i + 1])

        continuous_segments = no_cross_continuous_segments
        print(continuous_segments)

        # init_points = np.hstack(
        #     (of_analyser.data[:, 194:232],
        #      of_analyser.data[:, 236:274])
        # )

        participant_id = self.participant_id
        session_id = self.session_id

        for hand in ['left', 'right']:
            if hand == 'left':
                init_points = of_analyser.data[:, 194:232]
            else:
                init_points = of_analyser.data[:, 236:274]


            for segment in continuous_segments:
                starting = int(segment[0])
                ending = int(segment[1])
                max_length = ending - starting
                print('-------->', starting, ending)
                for i in range(math.floor((max_length - window_size) / window_step) + 2):
                    sub_starting, sub_ending = i * window_step, i * window_step + window_size
                    sub_starting += starting
                    sub_ending += starting

                    if sub_ending > ending:
                        sub_ending = ending

                    if sub_ending - sub_starting < min_size:
                        sub_starting = sub_ending - window_size

                        if sub_starting < starting:
                            # can't take at least one window
                            continue

                    assert sub_ending - sub_starting <= window_size, 'sub slice must == to window size!'

                    print('start slicing:', sub_starting, sub_ending)

                    new_file_name = 'participant_video_{}_{}_<{}_{}>_{}.npy'.format(
                        participant_id, session_id, sub_starting, sub_ending, hand
                    )

                    # run optical flow instance
                    optical_flow_data = of_analyser.run_optical_flow(cap, starting_time=sub_starting,
                                                                     ending_time=sub_ending, init_points=init_points,
                                                                     visualise=False)

                    result = []

                    for i in range(sub_starting, sub_ending):
                        # print(optical_flow_data[i].reshape((1, -1)).shape)
                        # print(i)
                        if i not in optical_flow_data:
                            print('Error detected, closing segment.')
                            break
                        result.append(optical_flow_data[i].reshape((1, -1)))

                    result_path = os.path.join(DATA_FOLDER,
                                               'hand_action_analysis_optical_flow_label',
                                               new_file_name
                                               )

                    result_array = np.zeros((len(result), 19 * 2))
                    for index, frame_data in enumerate(result):
                        result_array[index, :frame_data.shape[1]] = frame_data
                        result_array[index, frame_data.shape[1]:] = result_array[index - 1, frame_data.shape[1]:]
                    # print(result_array)
                    result = result_array
                    # result = result.reshape((result.shape[0], result.shape[2]))
                    create_dirs([os.path.split(result_path)[0]])
                    np.save(result_path, result)
                    # FFT, STD, MEAN = self.analyse_sequence_new(self.get_first_derivative(result))
                    # print(np.mean(FFT, axis=0))
                    # print(np.mean(STD))
                    # print(np.mean(MEAN))
                    #
                    # input()

        cap.release()
        cv2.destroyAllWindows()

        print('saving completed.')


    def hand_fidgeting_training_DNN(self):
        from sklearn.model_selection import train_test_split

        data = {}
        for root, dirs, files in os.walk(os.path.join(DATA_FOLDER, 'hand_action_analysis_optical_flow_label')):
            for file in files:
                if '.npy' in file:
                    data[file] = np.load(os.path.join(root, file))

        X = []
        y = []

        label_data_collection = {}

        for file_name in data.keys():
            participant_id, session_id, starting, ending = decompose_string(file_name)
            sub_data = data[file_name]

            label_file_path = os.path.join(DATA_FOLDER, 'hand_action_analysis_optical_flow_label',
                                           file_name.replace('.npy', '.label1'))
            if not os.path.exists(label_file_path):
                continue

            with open(label_file_path, 'r') as f:
                label = f.read()

            label = int(label)
            if label == 2:
                label = 1
            if label == -1:
                continue

            FFT, STD, MEAN = self.analyse_sequence_new(self.get_first_derivative(sub_data))
            FFT = np.mean(FFT, axis=1)
            STD = STD  # np.mean(STD)
            MEAN = MEAN  # np.mean(MEAN, axis=0)

            # ratio = np.count_nonzero(label_hand_cross_dynamic_rhythmic[starting:ending, :]) / (ending - starting)
            # if ratio >= 0.8:
            #     y.append(1)
            # else:
            #     y.append(0)
            y.append(label)
            single_x = np.hstack(
                (FFT.reshape((1, -1)), STD.reshape((1, -1)), MEAN.reshape((1, -1)))
            )
            X.append(
                single_x
            )
            print(file_name)

        print(y)
        X = np.array(X)
        print(X.shape)

        # divide partition

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        X_train, X_dev, y_train, y_dev = train_test_split(X_train, y_train, test_size=0.25)

        def reshape_after_division(X):
            return X.reshape((X.shape[0], X.shape[2]))

        X_train = reshape_after_division(X_train)
        X_dev = reshape_after_division(X_dev)
        X_test = reshape_after_division(X_test)

        dnn = Fidgeting_DNN(input_dim=[41, 76, 76], num_classes=2)
        dnn.build_multi_class_model()

        dnn.train_multi_class_model(X_train,
                                    y_train,
                                    X_dev,
                                    y_dev, class_weight={0: 1, 1: 1.3})
        dnn.evaluate_multi_class(X_train, y_train)
        dnn.evaluate_multi_class(X_dev, y_dev)
        dnn.evaluate_multi_class(X_test, y_test)

        dnn.save_model(
            os.path.join(DATA_FOLDER, 'pre-trained', 'hierarchical_DNN.h5')
        )
        # dnn = Fidgeting_DNN(input_dim=X_train.shape[1], num_classes=2)
        # dnn.build_model()
        # dnn.train_model(X_train, y_train, X_dev, y_dev, class_weight={0: 1, 1: 5})
        # dnn.evaluate(X_train, y_train)
        # dnn.evaluate(X_dev, y_dev)
        # dnn.evaluate(X_test, y_test)

    def hand_fidgeting_training_cross_validation(self):
        from sklearn.model_selection import train_test_split
        def reshape_after_division(X):
            return X.reshape((X.shape[0], X.shape[2]))

        data = {}
        for root, dirs, files in os.walk(os.path.join(DATA_FOLDER, 'hand_cross_analysis_optical_flow_label')):
            for file in files:
                if '.npy' in file:
                    data[file] = np.load(os.path.join(root, file))

        X = []
        y = []

        all_data = {}

        label_data_collection = {}

        for file_name in data.keys():
            print(file_name)
            participant_id, session_id, starting, ending = decompose_string(file_name)
            all_data.setdefault(participant_id, {'data_list': [], 'label_list': []})
            sub_data = data[file_name]

            label_file_path = os.path.join(DATA_FOLDER, 'hand_cross_analysis_optical_flow_label',
                                           file_name.replace('.npy', '.label1'))
            if not os.path.exists(label_file_path):
                continue
            else:
                with open(label_file_path, 'r') as f:
                    label1 = f.read()

            label_file_path = os.path.join(DATA_FOLDER, 'hand_cross_analysis_optical_flow_label',
                                           file_name.replace('.npy', '.label2'))
            if not os.path.exists(label_file_path):
                label2 = label1
            else:
                with open(label_file_path, 'r') as f:
                    label2 = f.read()

            if label1 != label2:
                print('drop due to disagreement')
                continue

            print(label1, label2)

            label = int(label1)
            if label == 2:
                label = 1
            if label == -1:
                continue

            FFT, STD, MEAN = self.analyse_sequence_new(self.get_first_derivative(sub_data))
            FFT = np.mean(FFT, axis=1)
            STD = STD  # np.mean(STD)
            MEAN = MEAN  # np.mean(MEAN, axis=0)

            all_data[participant_id]['label_list'].append(label)
            single_x = np.hstack(
                (FFT.reshape((1, -1)), STD.reshape((1, -1)), MEAN.reshape((1, -1)))
            )
            all_data[participant_id]['data_list'].append(
                single_x
            )

        print(all_data.keys())
        id_list = np.array(list(all_data.keys()))
        print(id_list)
        kf = KFold(n_splits=5)
        reports = []
        for train_index, test_index in kf.split(id_list):
            train_id_list = id_list[train_index]
            test_id_list = id_list[test_index]
            print(train_id_list, test_id_list)

            X_train = []
            y_train = []
            X_test = []
            y_test = []
            for id in list(train_id_list):
                X_train += all_data[id]['data_list']
                y_train += all_data[id]['label_list']

            for id in list(test_id_list):
                X_test += all_data[id]['data_list']
                y_test += all_data[id]['label_list']

            print(len(X_train), len(y_train))
            print(len(X_test), len(y_test))

            X_train = np.array(X_train)
            X_test = np.array(X_test)
            X_train = reshape_after_division(X_train)
            X_test = reshape_after_division(X_test)

            from sklearn.utils.class_weight import compute_class_weight
            class_weights = compute_class_weight('balanced', [0,1], y_train)

            dnn = Fidgeting_DNN(input_dim=[41, 76, 76], num_classes=2)
            dnn.build_multi_class_model()
            print('class_weights:', class_weights)
            dnn.train_multi_class_model(X_train,
                                        y_train,
                                        X_test,
                                        y_test,
                                        class_weight=class_weights)
            dnn.evaluate_multi_class(X_train, y_train)
            reports.append(dnn.evaluate_multi_class(X_test, y_test))
        return





        # dnn.save_model(
        #     os.path.join(DATA_FOLDER, 'pre-trained', 'hierarchical_DNN.h5')
        # )
        # dnn = Fidgeting_DNN(input_dim=X_train.shape[1], num_classes=2)
        # dnn.build_model()
        # dnn.train_model(X_train, y_train, X_dev, y_dev, class_weight={0: 1, 1: 5})
        # dnn.evaluate(X_train, y_train)
        # dnn.evaluate(X_dev, y_dev)
        # dnn.evaluate(X_test, y_test)

    def single_hand_fidgeting_training_DNN(self):
        from sklearn.model_selection import train_test_split

        data = {}
        for root, dirs, files in os.walk(os.path.join(DATA_FOLDER, 'hand_action_analysis_optical_flow_label')):
            for file in files:
                if '.npy' in file:
                    data[file] = np.load(os.path.join(root, file))

        X = []
        y = []

        label_data_collection = {}

        for file_name in data.keys():
            participant_id, session_id, starting, ending = decompose_string(file_name)
            sub_data = data[file_name]

            label_file_path = os.path.join(DATA_FOLDER, 'hand_action_analysis_optical_flow_label',
                                           file_name.replace('.npy', '.label1'))
            if not os.path.exists(label_file_path):
                continue

            print(file_name)

            with open(label_file_path, 'r') as f:
                label = f.read()

            label = int(label)
            if label == 2:
                label = 1
            if label == -1:
                continue

            FFT, STD, MEAN = self.analyse_sequence_new(self.get_first_derivative(sub_data))
            FFT = np.mean(FFT, axis=1)
            STD = STD  # np.mean(STD)
            MEAN = MEAN  # np.mean(MEAN, axis=0)

            # ratio = np.count_nonzero(label_hand_cross_dynamic_rhythmic[starting:ending, :]) / (ending - starting)
            # if ratio >= 0.8:
            #     y.append(1)
            # else:
            #     y.append(0)
            y.append(label)
            single_x = np.hstack(
                (FFT.reshape((1, -1)), STD.reshape((1, -1)), MEAN.reshape((1, -1)))
            )
            X.append(
                single_x
            )


        print(y)
        X = np.array(X)
        print(X.shape)

        # divide partition

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        X_train, X_dev, y_train, y_dev = train_test_split(X_train, y_train, test_size=0.25)

        def reshape_after_division(X):
            return X.reshape((X.shape[0], X.shape[2]))

        X_train = reshape_after_division(X_train)
        X_dev = reshape_after_division(X_dev)
        X_test = reshape_after_division(X_test)

        dnn = Fidgeting_DNN(input_dim=[41, 38, 38], num_classes=2)
        dnn.build_multi_class_model()

        dnn.train_multi_class_model(X_train,
                                    y_train,
                                    X_dev,
                                    y_dev, class_weight={0: 1, 1: 1.1})
        dnn.evaluate_multi_class(X_train, y_train)
        dnn.evaluate_multi_class(X_dev, y_dev)
        dnn.evaluate_multi_class(X_test, y_test)

        dnn.save_model(
            os.path.join(DATA_FOLDER, 'pre-trained', 'hierarchical_DNN_hand.h5')
        )
        # dnn = Fidgeting_DNN(input_dim=X_train.shape[1], num_classes=2)
        # dnn.build_model()
        # dnn.train_model(X_train, y_train, X_dev, y_dev, class_weight={0: 1, 1: 5})
        # dnn.evaluate(X_train, y_train)
        # dnn.evaluate(X_dev, y_dev)
        # dnn.evaluate(X_test, y_test)

    def foot_fidgeting_training(self):
        from sklearn.model_selection import train_test_split

        data = {}
        for root, dirs, files in os.walk(os.path.join(DATA_FOLDER, 'leg_action_analysis_optical_flow_label')):
            for file in files:
                if '.npy' in file:
                    data[file] = np.load(os.path.join(root, file))

        X = []
        y = []

        label_data_collection = {}

        for file_name in data.keys():
            participant_id, session_id, starting, ending = decompose_string(file_name)
            sub_data = data[file_name]
            label_file_path = os.path.join(DATA_FOLDER, 'leg_action_analysis_optical_flow_label',
                                           file_name.replace('.npy', '.label1'))
            if not os.path.exists(label_file_path):
                continue
            with open(label_file_path, 'r') as f:
                label = f.read()
            # print(sub_data.shape)
            key_name = '{}_{}'.format(participant_id, session_id)

            FFT, STD, MEAN = self.analyse_sequence_new(self.get_first_derivative(sub_data))
            FFT = np.mean(FFT, axis=1)
            STD = STD  # np.mean(STD)
            MEAN = MEAN  # np.mean(MEAN, axis=0)
            if label != '-1':
                y.append(label)
                single_x = np.hstack(
                    (FFT.reshape((1, -1)), STD.reshape((1, -1)), MEAN.reshape((1, -1)))
                )

                X.append(
                    single_x
                )

        print(y)
        return

        # print(X)
        X = np.array(X)
        X = X.reshape((X.shape[0], X.shape[2]))
        print(X.shape)

        # divide partition

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        X_train, X_dev, y_train, y_dev = train_test_split(X_train, y_train, test_size=0.25)

        def reshape_after_division(X):
            return X.reshape((X.shape[0], X.shape[2]))

        # X_train = reshape_after_division(X_train)
        # X_dev = reshape_after_division(X_dev)
        # X_test = reshape_after_division(X_test)

        dnn = Fidgeting_DNN(input_dim=[41, 12, 12], num_classes=3)
        dnn.build_multi_class_model()

        dnn.train_multi_class_model(X_train,
                                    y_train,
                                    X_dev,
                                    y_dev, class_weight={0: 1, 1: 3, 2: 3})
        dnn.evaluate_multi_class(X_train, y_train)
        dnn.evaluate_multi_class(X_dev, y_dev)
        dnn.evaluate_multi_class(X_test, y_test)

        dnn.save_model(
            os.path.join(DATA_FOLDER, 'pre-trained', 'hierarchical_DNN_leg.h5')
        )

        # dnn = Fidgeting_DNN(input_dim=X_train.shape[1])
        # dnn.build_model()
        # dnn.train_model(X_train, y_train, X_dev, y_dev, class_weight={0: 1, 1: 50})
        # dnn.evaluate(X_train, y_train)
        # dnn.evaluate(X_dev, y_dev)
        # dnn.evaluate(X_test, y_test)
        #
        # lstm = Fidgeting_LSTM(data_dim=X.shape[2], timesteps=X.shape[1], num_classes=3)
        # lstm.build_model()
        # lstm.train_model(X_train, y_train, X_dev, y_dev)
        # lstm.evaluate(X_train, y_train)
        # lstm.evaluate(X_dev, y_dev)
        # lstm.evaluate(X_test, y_test)
    '''
    DEMO
    '''
    def show_demo(self, play_starting=0, save_video=False):
        '''
        Show demo
        :param starting: int
        :param ending: int
        :return:
        '''
        if save_video:
            if os.path.exists(self.path_data['generated_demo_video']):
                print('video file exists, move on.')
                return

        cap = cv2.VideoCapture(self.video_path)
        data = np.load(self.processed_file)
        fps = cap.get(cv2.CAP_PROP_FPS)
        print('fps:', fps)
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(length)

        fused_data_path = self.path_data['fused_data']

        instance_hand_cross_analyser = HandCrossAnalyser('test', participant_data[participant_id][session_id])
        continuous_segments, hand_cross_valid_intersect_data = instance_hand_cross_analyser.compute_stationary_rectangles(
            cutoff=0, min_length=20)

        best_rects = {}
        for segment in continuous_segments:
            # find largest rectangle for each segment
            rects = [hand_cross_valid_intersect_data[i] for i in hand_cross_valid_intersect_data.keys()
                     if i >= segment[0] and i < segment[1]]
            rects = np.array(rects)
            best_rect = np.hstack((np.min(rects, axis=0)[:2], np.max(rects, axis=0)[2:]))
            for i in range(segment[0], segment[1]):
                best_rects[i] = best_rect

        print(continuous_segments)

        if os.path.exists(fused_data_path):
            # Read fused data directly
            fused_data = np.load(fused_data_path)
            label_array = fused_data[:, 0].reshape((-1, 1))  # H2H + fidgeting
            left_hand_arm_label_array = fused_data[:, 1].reshape((-1, 1))
            left_hand_leg_label_array = fused_data[:, 2].reshape((-1, 1))
            left_hand_face_label_array = fused_data[:, 3].reshape((-1, 1))
            right_hand_arm_label_array = fused_data[:, 4].reshape((-1, 1))
            right_hand_leg_label_array = fused_data[:, 5].reshape((-1, 1))
            right_hand_face_label_array = fused_data[:, 6].reshape((-1, 1))
            leg_location_label_array = fused_data[:, 7].reshape((-1, 1))
            leg_action_label_array = fused_data[:, 8].reshape((-1, 1))
            hand_action_label_array = fused_data[:, 9:11]  # shape (n, 2) NOTE!
            speaker_array = fused_data[:, 11].reshape((-1, 1))
            voice_array = fused_data[:, 12].reshape((-1, 1))

        else:
            # compute all necessary data
            # START!~~~

            ############################
            # processing hand cross info
            hand_cross_label_data = json.load(open(
                os.path.join(DATA_FOLDER, 'hand_cross_analysis_optical_flow', 'optical_flow_result.json'),
                'r'))
            try:
                hand_cross_label_data = hand_cross_label_data[str(self.participant_id)][str(self.session_id)]
            except Exception as e:
                print('no hands playing data...')
                hand_cross_label_data = {}

            window_size = 100
            window_step = 50

            # generate label array
            label_array = np.zeros((data.shape[0], 1))
            label_centroid = {}

            for segment in hand_cross_label_data.keys():
                starting = int(segment.split(',')[0])
                ending = int(segment.split(',')[1])
                centroid = int(math.floor((starting + ending) / 2))
                # p = (centroid, hand_cross_label_data[segment][0], hand_cross_label_data[segment][1])
                label_centroid[centroid] = hand_cross_label_data[segment]

            # print(label_centroid)

            print('preprocessing hand cross data')
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
                # print(closest_centroid)
                label = closest_centroid[1]

                label_array[t, 0] = label

            # print(label_array)


            ############################
            # processing leg location info
            print('computing leg intersection...')
            instance_leg = LegLocationAnalyser(self.name, self.path_data)
            _, leg_intersect_data = instance_leg.compute_leg_intersection(cutoff=0, min_length=20)
            leg_location_label_array = np.zeros((data.shape[0], 1))
            for frame_index in leg_intersect_data.keys():
                leg_location_label_array[frame_index, 0] = 1

            print('computing left hand location...')
            instance_left_hand = HandLocationAnalyser(self.name, self.path_data, hand='left')
            left_hand_arm_continuous_segments, left_hand_leg_continuous_segments, left_hand_face_continuous_segments = \
                instance_left_hand.compute_hand_intersection(cutoff=0, min_length=80)
            left_hand_arm_label_array = self.transfer_to_array(left_hand_arm_continuous_segments)
            left_hand_leg_label_array = self.transfer_to_array(left_hand_leg_continuous_segments)
            left_hand_face_label_array = self.transfer_to_array(left_hand_face_continuous_segments)

            print('computing right hand location...')
            instance_right_hand = HandLocationAnalyser(self.name, self.path_data, hand='right')
            right_hand_arm_continuous_segments, right_hand_leg_continuous_segments, right_hand_face_continuous_segments = \
                instance_right_hand.compute_hand_intersection(cutoff=0, min_length=80)
            right_hand_arm_label_array = self.transfer_to_array(right_hand_arm_continuous_segments)
            right_hand_leg_label_array = self.transfer_to_array(right_hand_leg_continuous_segments)
            right_hand_face_label_array = self.transfer_to_array(right_hand_face_continuous_segments)

            ############################
            # processing leg action info
            leg_action_label_data = json.load(open(
                os.path.join(DATA_FOLDER, 'leg_action_analysis_optical_flow', 'optical_flow_result.json'),
                'r'))
            try:
                leg_action_label_data = leg_action_label_data[str(self.participant_id)][str(self.session_id)]
            except Exception as e:
                print('no leg action data...')
                leg_action_label_data = {}

            print(leg_action_label_data)
            # generate label array
            leg_action_label_array = np.zeros((data.shape[0], 1))
            label_centroid = {}

            for segment in leg_action_label_data.keys():
                starting = int(segment.split(',')[0])
                ending = int(segment.split(',')[1])
                centroid = int(math.floor((starting + ending) / 2))
                # p = (centroid, hand_cross_label_data[segment][0], hand_cross_label_data[segment][1])
                label_centroid[centroid] = leg_action_label_data[segment]

            print('preprocessing leg action data')
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
                # print(closest_centroid)
                label = closest_centroid[1]

                leg_action_label_array[t, 0] = label

            # print(leg_action_label_array)

            ############################
            # processing hand action info
            hand_action_label_data = json.load(open(
                os.path.join(DATA_FOLDER, 'hand_action_analysis_optical_flow', 'optical_flow_result.json'),
                'r'))
            try:
                hand_action_label_data = hand_action_label_data[str(self.participant_id)][str(self.session_id)]
            except Exception as e:
                print('no hands action data...')
                hand_action_label_data = {}

            window_size = 100
            window_step = 50

            # generate label array
            hand_action_label_array = np.zeros((data.shape[0], 2))

            for hand in hand_action_label_data.keys():
                label_centroid = {}
                for segment in hand_action_label_data[hand].keys():
                    starting = int(segment.split(',')[0])
                    ending = int(segment.split(',')[1])
                    centroid = int(math.floor((starting + ending) / 2))
                    # p = (centroid, hand_cross_label_data[segment][0], hand_cross_label_data[segment][1])
                    label_centroid[centroid] = hand_action_label_data[hand][segment]

                # print(label_centroid)

                print('preprocessing hand action data')
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
                    # print(closest_centroid)
                    label = closest_centroid[1]
                    if hand == 'left':
                        hand_action_label_array[t, 0] = label
                    else:
                        hand_action_label_array[t, 1] = label

            # print(hand_action_label_array)

            ############################
            # processing speaker info

            print('Reading speaker info...')
            speaker_data = json.load(open(self.path_data['speaker_data'], 'r'))
            speaker_array = np.full((data.shape[0], 1), -1)

            for spk in speaker_data.keys():
                sub_spk_data = speaker_data[spk]
                spk = int(spk)
                # print(spk, '--->')
                for segment in sub_spk_data:
                    starting = segment[0]
                    ending = segment[1]
                    starting = math.floor(starting / 1000 * fps)
                    starting = min(starting, speaker_array.shape[0])
                    ending = math.ceil(ending / 1000 * fps)
                    ending = min(ending, speaker_array.shape[0])
                    speaker_array[starting:ending, :] = spk
                    # print(starting, ending)

            speaker_array = speaker_array.reshape((-1, 1))

            ############################
            # processing voice info
            voice_data = json.load(open(self.path_data['voice_data'], 'r'))
            voice_array = np.full((data.shape[0], 1), 0)
            for segment in voice_data:
                starting = segment[0]
                ending = segment[1]
                starting = math.floor(starting * fps)
                starting = min(starting, voice_array.shape[0])
                ending = math.ceil(ending * fps)
                ending = min(ending, voice_array.shape[0])
                voice_array[starting:ending, :] = 1
                print(starting, ending)

            # plt.plot(range(data.shape[0]), list(voice_array.reshape(-1)))
            # plt.show()
            voice_array = np.array(savgol_filter(list(voice_array.reshape(-1)), 51, 3)).reshape((-1, 1))
            voice_array[voice_array >= 0.3] = 1
            voice_array[voice_array < 0.3] = 0
            # plt.plot(range(data.shape[0]), list(voice_array.reshape(-1)))
            # plt.show()

            ############################
            # Data Fusion and save
            print(label_array.shape)
            print(left_hand_arm_label_array.shape)
            print(left_hand_leg_label_array.shape)
            print(left_hand_face_label_array.shape)
            print(right_hand_arm_label_array.shape)
            print(right_hand_leg_label_array.shape)
            print(right_hand_face_label_array.shape)
            print(leg_action_label_array.shape)
            print(leg_location_label_array.shape)
            print(hand_action_label_array.shape)
            print(speaker_array.shape)
            print(voice_array.shape)

            fused_data = np.hstack(
                (
                    label_array, # hand cross data
                    left_hand_arm_label_array,
                    left_hand_leg_label_array,
                    left_hand_face_label_array,
                    right_hand_arm_label_array,
                    right_hand_leg_label_array,
                    right_hand_face_label_array,
                    leg_location_label_array,
                    leg_action_label_array,
                    hand_action_label_array, # shape (n, 2) NOTE!
                    speaker_array,
                    voice_array,
                 )
            )
            np.save(fused_data_path, fused_data)

        #############################################
        # some preprocessing of data

        left_hand_to_leg_fidget_array = np.zeros((data.shape[0], 1))
        left_hand_to_leg_fidget_array[
            (hand_action_label_array[:, 0].reshape((-1, 1)) == 1) & (left_hand_leg_label_array == 1)] = 1

        right_hand_to_leg_fidget_array = np.zeros((data.shape[0], 1))
        right_hand_to_leg_fidget_array[
            (hand_action_label_array[:, 1].reshape((-1, 1)) == 1) & (right_hand_leg_label_array == 1)] = 1

        left_hand_to_arm_fidget_array = np.zeros((data.shape[0], 1))
        left_hand_to_arm_fidget_array[
            (hand_action_label_array[:, 0].reshape((-1, 1)) == 1) & (left_hand_arm_label_array == 1)] = 1

        right_hand_to_arm_fidget_array = np.zeros((data.shape[0], 1))
        right_hand_to_arm_fidget_array[
            (hand_action_label_array[:, 1].reshape((-1, 1)) == 1) & (right_hand_arm_label_array == 1)] = 1

        left_hand_to_face_fidget_array = np.zeros((data.shape[0], 1))
        left_hand_to_face_fidget_array[
            (hand_action_label_array[:, 0].reshape((-1, 1)) == 1) & (left_hand_face_label_array == 1)] = 1

        right_hand_to_face_fidget_array = np.zeros((data.shape[0], 1))
        right_hand_to_face_fidget_array[
            (hand_action_label_array[:, 1].reshape((-1, 1)) == 1) & (right_hand_face_label_array == 1)] = 1

        leg_fidget_array = leg_action_label_array
        leg_fidget_array[leg_fidget_array>1] = 1
        leg_fidget_array = np.array(savgol_filter(leg_fidget_array.reshape(-1).tolist(), 51, 3)).reshape((-1, 1))
        leg_fidget_array[leg_fidget_array >= 0.8] = 1
        leg_fidget_array[leg_fidget_array < 0.8] = 0

        #############################################
        if not save_video:
            input('ready to play!')


        cap.set(1, play_starting)
        t = play_starting

        if save_video:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(self.path_data['generated_demo_video'], fourcc, cap.get(cv2.CAP_PROP_FPS), (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

        while (t < data.shape[0]):
            print('video analysing in progress:', t / data.shape[0], end='\r')
            ret, frame = cap.read()
            # Display all the data points

            try:
                if int(speaker_array[t]) in speaker_label_data[self.participant_id][self.session_id]:
                    if voice_array[t, 0] == 1:
                        frame = self.paint_text(frame, 'Participant Speaking', (350, 200), font_size=1)
                # elif int(speaker_array[t]) == -1:
                #    frame = self.paint_text(frame, 'Unknown', (400, 200), font_size=1)
                # else:
                #    frame = self.paint_text(frame, 'Interviewer speaking', (400, 200), font_size=1)
            except:
                frame = self.paint_text(frame,
                                        '{} of {} speaking'.format(str(speaker_array[t]), len(speaker_data.keys()) - 1),
                                        (100, 100))



            # for i in range(25):
            #     frame = self.paint_point(frame, [data[t, i * 2], data[t, i * 2 + 1]])
            # for i in range(25, 95):
            #     frame = self.paint_point(frame, [data[t, i * 2], data[t, i * 2 + 1]], color=COLOR_BLUE)
            # for i in range(95, 116):
            #     frame = self.paint_point(frame, [data[t, i * 2], data[t, i * 2 + 1]], color=COLOR_GREEN)
            # for i in range(116, 137):
            #     frame = self.paint_point(frame, [data[t, i * 2], data[t, i * 2 + 1]], color=COLOR_YELLOW)

            left_hand_data = data[t, 194:232].reshape(-1, 2)
            right_hand_data = data[t, 236:274].reshape(-1, 2)
            face_data = data[t, 50:190].reshape(-1, 2)
            left_foot_data = data[t, 38:44].reshape(-1, 2)
            right_foot_data = data[t, 44:50].reshape(-1, 2)

            # frame = self.paint_rectangle_to_points(frame, left_hand_data, color=COLOR_GREEN)
            # frame = self.paint_rectangle_to_points(frame, right_hand_data, color=COLOR_YELLOW)

            if leg_location_label_array[t, 0] == 1:
                frame = self.paint_text(frame, 'Leg cross', (790, 630), font_size=1)

            if leg_action_label_array[t, 0] == 2:
                frame = self.paint_text(frame, 'Leg dynamic', (790, 560), font_size=1)
                #frame = self.paint_rectangle_to_points(frame, [(540, 430), (790, 630)], color=color)

            if leg_fidget_array[t, 0] == 1:
                for i in range(25):
                    frame = self.paint_point(frame, [data[t, i * 2], data[t, i * 2 + 1]], color=COLOR_RED)
                frame = self.paint_text(frame, 'Leg Fidgeting', (790, 500), font_size=0.5)


            if t in hand_cross_valid_intersect_data.keys():
                # print(t)
                rect = best_rects[t]
                label = label_array[t, 0]
                # print(label)
                color = COLOR_YELLOW
                if label == 1:
                    color = COLOR_RED
                    frame = self.paint_text(frame, 'Hand Cross Fidgeting', (rect[2], rect[3]), font_size=0.5)
                else:
                    color = COLOR_GREEN
                    frame = self.paint_text(frame, 'Hand Cross', (rect[2], rect[3]), font_size=0.5)

                frame = cv2.rectangle(frame,
                                      (rect[0] - 10, rect[1] - 10),
                                      (rect[2] + 10, rect[3] + 10),
                                      color,
                                      2)
            else:
                left_hand_action_label = hand_action_label_array[t, 0]
                right_hand_action_label = hand_action_label_array[t, 1]
                flag_left = ''
                flag_right = ''

                if left_hand_action_label == 1:
                    if left_hand_to_leg_fidget_array[t, 0] == 1:
                        flag_left = 'Left hand to leg fidgeting'
                    if left_hand_to_arm_fidget_array[t, 0] == 1:
                        flag_left = 'Left hand to arm fidgeting'
                    if left_hand_to_face_fidget_array[t, 0] == 1:
                        flag_left = 'Left hand to face'
                if right_hand_action_label == 1:
                    if right_hand_to_leg_fidget_array[t, 0] == 1:
                        flag_right = 'Right hand to leg fidgeting'
                    if right_hand_to_arm_fidget_array[t, 0] == 1:
                        flag_right = 'Right hand to arm fidgeting'
                    if right_hand_to_face_fidget_array[t, 0] == 1:
                        flag_right = 'Right hand to face'

                if flag_left:
                    frame = self.paint_rectangle_to_points(frame, left_hand_data, color=COLOR_RED)
                    frame = self.paint_text(frame, flag_left, (400, 400), font_size=0.5)
                if flag_right:
                    frame = self.paint_rectangle_to_points(frame, right_hand_data, color=COLOR_RED)
                    frame = self.paint_text(frame, flag_right, (400, 450), font_size=0.5)

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
        if save_video:
            out.release()
        cap.release()
        cv2.destroyAllWindows()

    def show_actor_demo(self, play_starting=0, save_video=False):
        '''
        Show demo
        :param starting: int
        :param ending: int
        :return:
        '''

        if save_video:
            create_dirs([os.path.split(self.path_data['generated_demo_video'])[0]])
            print(self.path_data['generated_demo_video'])
            if os.path.exists(self.path_data['generated_demo_video']):
                print('video file exists, move on.')
                return

        cap = cv2.VideoCapture(self.video_path)
        data = np.load(self.processed_file)
        fps = cap.get(cv2.CAP_PROP_FPS)
        print('fps:', fps)
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(length)

        fused_data_path = self.path_data['fused_data']
        create_dirs([os.path.split(fused_data_path)[0]])

        instance_hand_cross_analyser = HandCrossAnalyser('test', participant_data[participant_id][session_id])
        continuous_segments, hand_cross_valid_intersect_data = instance_hand_cross_analyser.compute_stationary_rectangles(
            cutoff=0, min_length=20)

        best_rects = {}
        for segment in continuous_segments:
            # find largest rectangle for each segment
            rects = [hand_cross_valid_intersect_data[i] for i in hand_cross_valid_intersect_data.keys()
                     if i >= segment[0] and i < segment[1]]
            rects = np.array(rects)
            best_rect = np.hstack((np.min(rects, axis=0)[:2], np.max(rects, axis=0)[2:]))
            for i in range(segment[0], segment[1]):
                best_rects[i] = best_rect

        print(continuous_segments)

        if os.path.exists(fused_data_path):
            # Read fused data directly
            fused_data = np.load(fused_data_path)
            label_array = fused_data[:, 0].reshape((-1, 1))  # H2H + fidgeting
            left_hand_arm_label_array = fused_data[:, 1].reshape((-1, 1))
            left_hand_leg_label_array = fused_data[:, 2].reshape((-1, 1))
            left_hand_face_label_array = fused_data[:, 3].reshape((-1, 1))
            right_hand_arm_label_array = fused_data[:, 4].reshape((-1, 1))
            right_hand_leg_label_array = fused_data[:, 5].reshape((-1, 1))
            right_hand_face_label_array = fused_data[:, 6].reshape((-1, 1))
            leg_location_label_array = fused_data[:, 7].reshape((-1, 1))
            leg_action_label_array = fused_data[:, 8].reshape((-1, 1))
            hand_action_label_array = fused_data[:, 9:11]  # shape (n, 2) NOTE!
            # speaker_array = fused_data[:, 11].reshape((-1, 1))
            # voice_array = fused_data[:, 12].reshape((-1, 1))

        else:
            # compute all necessary data
            # START!~~~

            ############################
            # processing hand cross info
            hand_cross_label_data = json.load(open(
                os.path.join(DATA_FOLDER, 'hand_cross_analysis_optical_flow', 'optical_flow_result.json'),
                'r'))
            try:
                hand_cross_label_data = hand_cross_label_data[str(self.participant_id)][str(self.session_id)]
            except Exception as e:
                print('no hands playing data...')
                hand_cross_label_data = {}

            window_size = 100
            window_step = 50

            # generate label array
            label_array = np.zeros((data.shape[0], 1))
            label_centroid = {}

            for segment in hand_cross_label_data.keys():
                starting = int(segment.split(',')[0])
                ending = int(segment.split(',')[1])
                centroid = int(math.floor((starting + ending) / 2))
                # p = (centroid, hand_cross_label_data[segment][0], hand_cross_label_data[segment][1])
                label_centroid[centroid] = hand_cross_label_data[segment]

            # print(label_centroid)

            print('preprocessing hand cross data')
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
                # print(closest_centroid)
                label = closest_centroid[1]

                label_array[t, 0] = label

            # print(label_array)


            ############################
            # processing leg location info
            print('computing leg intersection...')
            instance_leg = LegLocationAnalyser(self.name, self.path_data)
            _, leg_intersect_data = instance_leg.compute_leg_intersection(cutoff=0, min_length=20)
            leg_location_label_array = np.zeros((data.shape[0], 1))
            for frame_index in leg_intersect_data.keys():
                leg_location_label_array[frame_index, 0] = 1

            print('computing left hand location...')
            instance_left_hand = HandLocationAnalyser(self.name, self.path_data, hand='left')
            left_hand_arm_continuous_segments, left_hand_leg_continuous_segments, left_hand_face_continuous_segments = \
                instance_left_hand.compute_hand_intersection(cutoff=0, min_length=20)
            left_hand_arm_label_array = self.transfer_to_array(left_hand_arm_continuous_segments)
            left_hand_leg_label_array = self.transfer_to_array(left_hand_leg_continuous_segments)
            left_hand_face_label_array = self.transfer_to_array(left_hand_face_continuous_segments)

            print('computing right hand location...')
            instance_right_hand = HandLocationAnalyser(self.name, self.path_data, hand='right')
            right_hand_arm_continuous_segments, right_hand_leg_continuous_segments, right_hand_face_continuous_segments = \
                instance_right_hand.compute_hand_intersection(cutoff=0, min_length=20)
            right_hand_arm_label_array = self.transfer_to_array(right_hand_arm_continuous_segments)
            right_hand_leg_label_array = self.transfer_to_array(right_hand_leg_continuous_segments)
            right_hand_face_label_array = self.transfer_to_array(right_hand_face_continuous_segments)

            ############################
            # processing leg action info
            leg_action_label_data = json.load(open(
                os.path.join(DATA_FOLDER, 'leg_action_analysis_optical_flow', 'optical_flow_result.json'),
                'r'))
            try:
                leg_action_label_data = leg_action_label_data[str(self.participant_id)][str(self.session_id)]
            except Exception as e:
                print('no leg action data...')
                leg_action_label_data = {}

            print(leg_action_label_data)
            # generate label array
            leg_action_label_array = np.zeros((data.shape[0], 1))
            label_centroid = {}

            for segment in leg_action_label_data.keys():
                starting = int(segment.split(',')[0])
                ending = int(segment.split(',')[1])
                centroid = int(math.floor((starting + ending) / 2))
                # p = (centroid, hand_cross_label_data[segment][0], hand_cross_label_data[segment][1])
                label_centroid[centroid] = leg_action_label_data[segment]

            print('preprocessing leg action data')
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
                # print(closest_centroid)
                label = closest_centroid[1]

                leg_action_label_array[t, 0] = label

            # print(leg_action_label_array)

            ############################
            # processing hand action info
            hand_action_label_data = json.load(open(
                os.path.join(DATA_FOLDER, 'hand_action_analysis_optical_flow', 'optical_flow_result.json'),
                'r'))
            try:
                hand_action_label_data = hand_action_label_data[str(self.participant_id)][str(self.session_id)]
            except Exception as e:
                print('no hands action data...')
                hand_action_label_data = {}

            window_size = 100
            window_step = 50

            # generate label array
            hand_action_label_array = np.zeros((data.shape[0], 2))

            for hand in hand_action_label_data.keys():
                label_centroid = {}
                for segment in hand_action_label_data[hand].keys():
                    starting = int(segment.split(',')[0])
                    ending = int(segment.split(',')[1])
                    centroid = int(math.floor((starting + ending) / 2))
                    # p = (centroid, hand_cross_label_data[segment][0], hand_cross_label_data[segment][1])
                    label_centroid[centroid] = hand_action_label_data[hand][segment]

                # print(label_centroid)

                print('preprocessing hand action data')
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
                    # print(closest_centroid)
                    label = closest_centroid[1]
                    if hand == 'left':
                        hand_action_label_array[t, 0] = label
                    else:
                        hand_action_label_array[t, 1] = label

            # print(hand_action_label_array)

            ############################
            # Data Fusion and save
            print(label_array.shape)
            print(left_hand_arm_label_array.shape)
            print(left_hand_leg_label_array.shape)
            print(left_hand_face_label_array.shape)
            print(right_hand_arm_label_array.shape)
            print(right_hand_leg_label_array.shape)
            print(right_hand_face_label_array.shape)
            print(leg_action_label_array.shape)
            print(leg_location_label_array.shape)
            print(hand_action_label_array.shape)
            # print(speaker_array.shape)
            # print(voice_array.shape)

            fused_data = np.hstack(
                (
                    label_array, # hand cross data
                    left_hand_arm_label_array,
                    left_hand_leg_label_array,
                    left_hand_face_label_array,
                    right_hand_arm_label_array,
                    right_hand_leg_label_array,
                    right_hand_face_label_array,
                    leg_location_label_array,
                    leg_action_label_array,
                    hand_action_label_array, # shape (n, 2) NOTE!
                    # speaker_array,
                    # voice_array,
                 )
            )
            np.save(fused_data_path, fused_data)

        #############################################
        # some preprocessing of data

        left_hand_to_leg_fidget_array = np.zeros((data.shape[0], 1))
        left_hand_to_leg_fidget_array[
            (hand_action_label_array[:, 0].reshape((-1, 1)) == 1) & (left_hand_leg_label_array == 1)] = 1

        right_hand_to_leg_fidget_array = np.zeros((data.shape[0], 1))
        right_hand_to_leg_fidget_array[
            (hand_action_label_array[:, 1].reshape((-1, 1)) == 1) & (right_hand_leg_label_array == 1)] = 1

        left_hand_to_arm_fidget_array = np.zeros((data.shape[0], 1))
        left_hand_to_arm_fidget_array[
            (hand_action_label_array[:, 0].reshape((-1, 1)) == 1) & (left_hand_arm_label_array == 1)] = 1

        right_hand_to_arm_fidget_array = np.zeros((data.shape[0], 1))
        right_hand_to_arm_fidget_array[
            (hand_action_label_array[:, 1].reshape((-1, 1)) == 1) & (right_hand_arm_label_array == 1)] = 1

        left_hand_to_face_fidget_array = np.zeros((data.shape[0], 1))
        left_hand_to_face_fidget_array[
            (hand_action_label_array[:, 0].reshape((-1, 1)) == 1) & (left_hand_face_label_array == 1)] = 1

        right_hand_to_face_fidget_array = np.zeros((data.shape[0], 1))
        right_hand_to_face_fidget_array[
            (hand_action_label_array[:, 1].reshape((-1, 1)) == 1) & (right_hand_face_label_array == 1)] = 1

        leg_fidget_array = leg_action_label_array
        leg_fidget_array[leg_fidget_array>1] = 1
        leg_fidget_array = np.array(savgol_filter(leg_fidget_array.reshape(-1).tolist(), 51, 3)).reshape((-1, 1))
        leg_fidget_array[leg_fidget_array >= 0.8] = 1
        leg_fidget_array[leg_fidget_array < 0.8] = 0

        right_hand_to_leg_fidget_array = self.transfer_to_array(self.transfer_to_segments(right_hand_to_leg_fidget_array, min_length=100))
        left_hand_to_leg_fidget_array = self.transfer_to_array(
            self.transfer_to_segments(left_hand_to_leg_fidget_array, min_length=100))

        #############################################
        if not save_video:
            input('ready to play!')


        cap.set(1, play_starting)
        t = play_starting

        if save_video:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(self.path_data['generated_demo_video'], fourcc, cap.get(cv2.CAP_PROP_FPS), (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

        while (t < data.shape[0]):
            print('video analysing in progress:', t / data.shape[0], end='\r')
            ret, frame = cap.read()
            # Display all the data points

            # try:
            #     if int(speaker_array[t]) in speaker_label_data[self.participant_id][self.session_id]:
            #         if voice_array[t, 0] == 1:
            #             frame = self.paint_text(frame, 'Participant Speaking', (350, 200), font_size=1)
            #     # elif int(speaker_array[t]) == -1:
            #     #    frame = self.paint_text(frame, 'Unknown', (400, 200), font_size=1)
            #     # else:
            #     #    frame = self.paint_text(frame, 'Interviewer speaking', (400, 200), font_size=1)
            # except:
            #     frame = self.paint_text(frame,
            #                             '{} of {} speaking'.format(str(speaker_array[t]), len(speaker_data.keys()) - 1),
            #                             (100, 100))



            # for i in range(25):
            #     frame = self.paint_point(frame, [data[t, i * 2], data[t, i * 2 + 1]])
            # for i in range(25, 95):
            #     frame = self.paint_point(frame, [data[t, i * 2], data[t, i * 2 + 1]], color=COLOR_BLUE)
            # for i in range(95, 116):
            #     frame = self.paint_point(frame, [data[t, i * 2], data[t, i * 2 + 1]], color=COLOR_GREEN)
            # for i in range(116, 137):
            #     frame = self.paint_point(frame, [data[t, i * 2], data[t, i * 2 + 1]], color=COLOR_YELLOW)

            left_hand_data = data[t, 194:232].reshape(-1, 2)
            right_hand_data = data[t, 236:274].reshape(-1, 2)
            face_data = data[t, 50:190].reshape(-1, 2)
            left_foot_data = data[t, 38:44].reshape(-1, 2)
            right_foot_data = data[t, 44:50].reshape(-1, 2)

            # frame = self.paint_rectangle_to_points(frame, left_hand_data, color=COLOR_GREEN)
            # frame = self.paint_rectangle_to_points(frame, right_hand_data, color=COLOR_YELLOW)

            # if leg_location_label_array[t, 0] == 1:
            #     frame = self.paint_text(frame, 'Leg cross', (790, 630), font_size=1)
            #
            # if leg_action_label_array[t, 0] == 2:
            #     frame = self.paint_text(frame, 'Leg dynamic', (790, 560), font_size=1)
                #frame = self.paint_rectangle_to_points(frame, [(540, 430), (790, 630)], color=color)

            if leg_fidget_array[t, 0] == 1:
                for i in range(25):
                    frame = self.paint_point(frame, [data[t, i * 2], data[t, i * 2 + 1]], color=COLOR_RED)
                frame = self.paint_text(frame, 'Leg Fidgeting', (400, 350), font_size=0.7)


            if t in hand_cross_valid_intersect_data.keys():
                # print(t)
                rect = best_rects[t]
                label = label_array[t, 0]
                # print(label)
                color = COLOR_YELLOW
                if label == 1:
                    color = COLOR_RED
                    frame = self.paint_text(frame, 'Hand Cross Fidgeting', (rect[2], rect[3]), font_size=0.5)
                else:
                    color = COLOR_GREEN
                    frame = self.paint_text(frame, 'Hand Cross', (rect[2], rect[3]), font_size=0.5)

                frame = cv2.rectangle(frame,
                                      (rect[0] - 10, rect[1] - 10),
                                      (rect[2] + 10, rect[3] + 10),
                                      color,
                                      2)
            else:
                left_hand_action_label = hand_action_label_array[t, 0]
                right_hand_action_label = hand_action_label_array[t, 1]
                flag_left = ''
                flag_right = ''

                if left_hand_action_label == 1:
                    if left_hand_to_leg_fidget_array[t, 0] == 1:
                        flag_left = 'Left hand to leg fidgeting'
                    if left_hand_to_arm_fidget_array[t, 0] == 1:
                        flag_left = 'Left hand to arm fidgeting'
                    if left_hand_to_face_fidget_array[t, 0] == 1:
                        flag_left = 'Left hand to face'
                if right_hand_action_label == 1:
                    if right_hand_to_leg_fidget_array[t, 0] == 1:
                        flag_right = 'Right hand to leg fidgeting'
                    if right_hand_to_arm_fidget_array[t, 0] == 1:
                        flag_right = 'Right hand to arm fidgeting'
                    if right_hand_to_face_fidget_array[t, 0] == 1:
                        flag_right = 'Right hand to face'

                if flag_left:
                    frame = self.paint_rectangle_to_points(frame, left_hand_data, color=COLOR_RED)
                    frame = self.paint_text(frame, flag_left, (400, 400), font_size=0.5)
                if flag_right:
                    frame = self.paint_rectangle_to_points(frame, right_hand_data, color=COLOR_RED)
                    frame = self.paint_text(frame, flag_right, (400, 450), font_size=0.5)

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
        if save_video:
            out.release()
        cap.release()
        cv2.destroyAllWindows()


    def generate_training_data(self):
        # slicing videos and assign labels
        for participant_id in participant_data.keys():
            if participant_id in [109]:
                continue
            for session_id in participant_data[participant_id].keys():
                try:
                    sub_pipeline = MainPipeline('test', participant_data[participant_id][session_id])
                    print('working on', participant_id, session_id)
                    fused_data_path = sub_pipeline.path_data['fused_data']
                    print(fused_data_path)
                    fused_data = np.load(fused_data_path)
                    print(fused_data.shape)

                    hand_cross_fidget_label_array = fused_data[:, 0].reshape((-1, 1))  # hand cross fidget data
                    left_hand_arm_label_array = fused_data[:, 1].reshape((-1, 1))
                    left_hand_leg_label_array = fused_data[:, 2].reshape((-1, 1))
                    left_hand_face_label_array = fused_data[:, 3].reshape((-1, 1))
                    right_hand_arm_label_array = fused_data[:, 4].reshape((-1, 1))
                    right_hand_leg_label_array = fused_data[:, 5].reshape((-1, 1))
                    right_hand_face_label_array = fused_data[:, 6].reshape((-1, 1))
                    leg_location_label_array = fused_data[:, 7].reshape((-1, 1))
                    leg_action_label_array = fused_data[:, 8].reshape((-1, 1))
                    hand_action_label_array = fused_data[:, 9:11]  # shape (n, 2) NOTE!
                    speaker_array = fused_data[:, 11].reshape((-1, 1))
                    voice_array = fused_data[:, 12].reshape((-1, 1))

                    left_hand_to_leg_fidget_array = np.zeros((fused_data.shape[0], 1))
                    left_hand_to_leg_fidget_array[
                        (hand_action_label_array[:, 0].reshape((-1, 1)) == 1) & (left_hand_leg_label_array == 1)] = 1

                    right_hand_to_leg_fidget_array = np.zeros((fused_data.shape[0], 1))
                    right_hand_to_leg_fidget_array[
                        (hand_action_label_array[:, 1].reshape((-1, 1)) == 1) & (right_hand_leg_label_array == 1)] = 1

                    left_hand_to_arm_fidget_array = np.zeros((fused_data.shape[0], 1))
                    left_hand_to_arm_fidget_array[
                        (hand_action_label_array[:, 0].reshape((-1, 1)) == 1) & (left_hand_arm_label_array == 1)] = 1

                    right_hand_to_arm_fidget_array = np.zeros((fused_data.shape[0], 1))
                    right_hand_to_arm_fidget_array[
                        (hand_action_label_array[:, 1].reshape((-1, 1)) == 1) & (right_hand_arm_label_array == 1)] = 1

                    left_hand_to_face_fidget_array = np.zeros((fused_data.shape[0], 1))
                    left_hand_to_face_fidget_array[
                        (hand_action_label_array[:, 0].reshape((-1, 1)) == 1) & (left_hand_face_label_array == 1)] = 1

                    right_hand_to_face_fidget_array = np.zeros((fused_data.shape[0], 1))
                    right_hand_to_face_fidget_array[
                        (hand_action_label_array[:, 1].reshape((-1, 1)) == 1) & (right_hand_face_label_array == 1)] = 1

                    leg_fidget_array = leg_action_label_array
                    leg_fidget_array[leg_fidget_array > 1] = 1
                    leg_fidget_array = np.array(savgol_filter(leg_fidget_array.reshape(-1).tolist(), 51, 3)).reshape(
                        (-1, 1))
                    leg_fidget_array[leg_fidget_array >= 0.8] = 1
                    leg_fidget_array[leg_fidget_array < 0.8] = 0

                    speaking_array = np.zeros((fused_data.shape[0], 1))
                    for speaker_label in speaker_label_data[participant_id][session_id]:
                        speaking_array[(speaker_array == speaker_label)] = 1
                    speaking_array[voice_array == 0] = 0
                    # hand_cross_fidget_label_array[speaking_array == 0] = 0
                    # left_hand_to_leg_fidget_array[speaking_array == 0] = 0
                    # right_hand_to_leg_fidget_array[speaking_array == 0] = 0
                    # left_hand_to_arm_fidget_array[speaking_array == 0] = 0
                    # right_hand_to_arm_fidget_array[speaking_array == 0] = 0
                    # left_hand_to_face_fidget_array[speaking_array == 0] = 0
                    # right_hand_to_face_fidget_array[speaking_array == 0] = 0
                    # leg_fidget_array[speaking_array == 0] = 0

                    # fused_data = np.hstack((
                    #     hand_cross_fidget_label_array,
                    #     left_hand_to_leg_fidget_array,
                    #     right_hand_to_leg_fidget_array,
                    #     left_hand_to_arm_fidget_array,
                    #     right_hand_to_arm_fidget_array,
                    #     left_hand_to_face_fidget_array,
                    #     right_hand_to_face_fidget_array,
                    #     leg_fidget_array,
                    #     speaking_array,
                    # ))

                    fused_data = np.hstack((
                        hand_cross_fidget_label_array,
                        left_hand_arm_label_array,
                        left_hand_leg_label_array,
                        left_hand_face_label_array,
                        right_hand_arm_label_array,
                        right_hand_leg_label_array,
                        right_hand_face_label_array,
                        leg_location_label_array,
                        leg_action_label_array,
                        hand_action_label_array,  # shape (n, 2) NOTE!
                        speaker_array,
                        voice_array,
                    ))

                    print('processed:', fused_data.shape)

                    processed_data = np.load(sub_pipeline.processed_smooth_file)
                    print(processed_data.shape)

                    gaze_data = processed_data[:, list(range(274, 282))]
                    AUs_data = processed_data[:, list(range(948, 983))]

                    gaze_data = stats.zscore(gaze_data, axis=1, ddof=1)
                    # processed_data = processed_data[:, list(range(274, 282)) + list(range(948, 983))]
                    # print(processed_data.shape)
                    training_data = np.hstack((fused_data, gaze_data, AUs_data))

                    # fv_model = FisherVectorGMM(n_kernels=128)
                    # fv_model.fit(training_data)
                    # fv_training_data = fv_model.predict(training_data, normalized=False)
                    # print(fv_training_data.shape)
                    # # training_data = processed_data
                    # training_data = {
                    #     'data': training_data,
                    #     'label': participant_score_data[participant_id]
                    # }
                    full_fused_data = {
                        'data': fused_data,
                        'label': participant_depression_data[participant_id],
                    }
                    pickle.dump(full_fused_data, open(sub_pipeline.path_data['full_fused_data'], 'wb'))
                except:
                    pass


if __name__ == '__main__':
    pprint(participant_data)
    for participant_id in participant_data.keys():
        for session_id in participant_data[participant_id].keys():
            print(participant_id, session_id)
            processor = VideoProcessor('processing', participant_data[participant_id][session_id])
            processor.preprocess_actor_keypoints(smooth=True)
            processor.preprocess_actor_keypoints(smooth=False)
            main_pipeline = MainPipeline('test', participant_data[participant_id][session_id])
            main_pipeline.generate_leg_slice()
            main_pipeline.generate_hand_slice()
            main_pipeline.generate_hand_cross_slice()
            instance_hand_cross_analyser = HandCrossAnalyser('test', None)
            instance_hand_cross_analyser.analyse_hand_cross_optical_flow()
            instance_leg_action_analyser = LegActionAnalyser('test', None)
            instance_leg_action_analyser.analyse_leg_action_optical_flow()
            instance_hand_action_analyser = HandCrossAnalyser('test', None)
            instance_hand_action_analyser.analyse_hand_action_optical_flow()

            main_pipeline.show_actor_demo(save_video=True)