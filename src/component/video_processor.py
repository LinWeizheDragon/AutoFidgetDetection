import os
import json
import cv2
import math
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from utility.colors import *
from utility.interpolation import cubic_interpolate
from utility.pose_data_reader import read_reshaped_directory
from utility.base_config import *
from utility.dirs import create_dirs

from component.basic_processor import BasicProcessor


class VideoProcessor(BasicProcessor):

    def __init__(self, name, path_data):
        BasicProcessor.__init__(self, name, path_data, None)

    def preprocess_keypoints(self, cubic_interpolation=True, smooth=True, overwrite=False):
        '''
        preprocessing data and save to files for further usage
        This is for full function
        :param cubic_interpolation:
        :param smooth:
        :return:
        '''
        # Read OpenPose output and merge into a numpy array
        print('Preprocessing:', self.processed_smooth_file)
        # if os.path.exists(self.processed_file):
        #     print('already processed')
        #     return

        print('Reading OpenPose data...')

        pose_raw_data = read_reshaped_directory(
            self.openpose_output_path, 0)
        pose_data = None
        for session in pose_raw_data.keys():

            session_data = None
            for feature_name in ['pose', 'face', 'left', 'right']:
                for i in range(pose_raw_data[session][feature_name].shape[0]):
                    if session_data is None:
                        session_data = pose_raw_data[session][feature_name][i, :, :]
                    else:
                        session_data = np.hstack((session_data,
                                                  pose_raw_data[session][feature_name][i, :, :]))

            if pose_data is None:
                if session != 0:
                    # shift data back
                    pose_data = np.vstack((np.zeros((session, session_data.shape[1]))))
                else:
                    pose_data = session_data
            else:
                # TODO: test on data with several sessions
                if pose_data.shape[0] < session:
                    # need to shift back data
                    pose_data = np.vstack((pose_data, np.zeros((session - pose_data.shape[0], pose_data.shape[1]))))
                pose_data = np.vstack((pose_data, session_data))

        print(pose_data.shape)

        '''
        i = 0
        try:
            pose_data = None
            while True:
                t = str(i).rjust(12, '0')
                keypoint_file = os.path.join(openpose_output_path, 'participant_video_' + t + '_keypoints.json')
                with open(keypoint_file) as json_file:
                    data = json.load(json_file)['people'][0]
                    pose_keypoints = data['pose_keypoints_2d']
                    face_keypoints = data['face_keypoints_2d']
                    hand_left_keypoints = data['hand_left_keypoints_2d']
                    hand_right_keypoints = data['hand_right_keypoints_2d']
                    single_data = np.array(pose_keypoints + face_keypoints + hand_left_keypoints + hand_right_keypoints)
                    #print(single_data)
                    if pose_data is None:
                        pose_data = single_data.reshape((1, -1))
                    else:
                        pose_data = np.vstack((pose_data, single_data.reshape((1, -1))))
                i += 1
        except:
            print('To the end of this video')
            print(pose_data.shape)
        '''

        if cubic_interpolation or smooth:
            print('start cubic interpolation and/or smoothing...')
            for i in range(int(pose_data.shape[1] / 3)):
                x_data = pose_data[:, i * 3]
                y_data = pose_data[:, i * 3 + 1]
                confidence_data = pose_data[:, i * 3 + 2]
                if cubic_interpolation:
                    x_data = cubic_interpolate(x_data, confidence_data)
                    y_data = cubic_interpolate(y_data, confidence_data)
                if smooth:
                    x_data = savgol_filter(x_data, 11, 3)
                    y_data = savgol_filter(y_data, 11, 3)

                pose_data[:, i * 3] = x_data
                pose_data[:, i * 3 + 1] = y_data

        confidence_index = [i * 3 + 2 for i in range(int(pose_data.shape[1] / 3))]
        print('Removing confidence data from pose data...')
        pose_data = np.delete(pose_data, confidence_index, axis=1)
        print(pose_data.shape)

        print('Reading OpenFace data...')
        face_data = pd.read_csv(self.openface_output_file, header=0)
        delay_compensation = 20
        face_data = face_data.iloc[delay_compensation:, :]
        face_success = face_data.iloc[:, 4].to_numpy()
        face_data = face_data.iloc[:, 5:].to_numpy()

        # Reshape and cope the cordinates with original video
        landmarks = face_data[:, 294:430]
        eyes = face_data[:, 8:120]

        landmarks = landmarks / 1.5
        eyes = eyes / 1.5
        landmarks[:, :68] = landmarks[:, :68] + 449
        landmarks[:, 68:] = landmarks[:, 68:] + 73
        eyes[:, :68] = eyes[:, :68] + 449
        eyes[:, 68:] = eyes[:, 68:] + 73

        face_data[:, 294:430] = landmarks
        face_data[:, 8:120] = eyes

        print(face_data.shape)

        if cubic_interpolation or smooth:
            print('start cubic interpolation and/or smoothing...')
            for i in range(674):  # IMPORTANT: don't smooth/interpolate AUs
                # print('start cubic interpolation and/or smoothing...', 'feature No.', i)
                x_data = face_data[:, i]
                if cubic_interpolation:
                    x_data = cubic_interpolate(x_data, face_success)
                # if smooth:
                #   x_data = savgol_filter(x_data, 31, 3)
                face_data[:, i] = x_data

        # Take the min length of existing data
        min_length = min(face_data.shape[0], pose_data.shape[0])

        print('stacking arrays...')
        data = np.hstack((pose_data[:min_length, :],
                          face_data[:min_length, :]))
        print('final shape:', data.shape)

        # data = pose_data
        # print('final shape:', data.shape)

        np.save(self.processed_smooth_file, data)

    def preprocess_actor_keypoints(self, cubic_interpolation=True, smooth=True, overwrite=False):
        '''
        preprocessing data and save to files for further usage
        This is for only fidgeting detection
        :param cubic_interpolation:
        :param smooth:
        :return:
        '''
        # Read OpenPose output and merge into a numpy array
        create_dirs([os.path.split(self.processed_smooth_file)[0]])
        create_dirs([os.path.split(self.processed_file)[0]])

        if smooth:
            print('Preprocessing:', self.processed_smooth_file)
            if os.path.exists(self.processed_smooth_file):
                print('already processed')
                return
        else:
            print('Preprocessing:', self.processed_file)
            if os.path.exists(self.processed_file):
                print('already processed')
                return

        print('Reading OpenPose data...')
        pose_data = None
        i = 0
        while True:
            single_file_path = os.path.join(self.openpose_output_path, str(self.participant_id)[-2:] + '_' + str(i).rjust(12, '0') + '_keypoints.json')
            # print(single_file_path)
            if not os.path.exists(single_file_path):
                print('reading to the end', i)
                break
            single_data = json.load(open(single_file_path, 'r'))
            single_data = single_data['people'][0]
            pose_keypoints_2d = single_data['pose_keypoints_2d']
            face_keypoints_2d = single_data['face_keypoints_2d']
            hand_left_keypoints_2d = single_data['hand_left_keypoints_2d']
            hand_right_keypoints_2d = single_data['hand_right_keypoints_2d']

            combined = pose_keypoints_2d + face_keypoints_2d + hand_left_keypoints_2d + hand_right_keypoints_2d
            # print(combined)
            combined = np.array(combined).reshape((1, -1))
            if pose_data is None:
                pose_data = combined
            else:
                pose_data = np.vstack((pose_data, combined))
            print('reading file...', pose_data.shape, end='\r')
            i += 1

        print(pose_data.shape)

        '''
        i = 0
        try:
            pose_data = None
            while True:
                t = str(i).rjust(12, '0')
                keypoint_file = os.path.join(openpose_output_path, 'participant_video_' + t + '_keypoints.json')
                with open(keypoint_file) as json_file:
                    data = json.load(json_file)['people'][0]
                    pose_keypoints = data['pose_keypoints_2d']
                    face_keypoints = data['face_keypoints_2d']
                    hand_left_keypoints = data['hand_left_keypoints_2d']
                    hand_right_keypoints = data['hand_right_keypoints_2d']
                    single_data = np.array(pose_keypoints + face_keypoints + hand_left_keypoints + hand_right_keypoints)
                    #print(single_data)
                    if pose_data is None:
                        pose_data = single_data.reshape((1, -1))
                    else:
                        pose_data = np.vstack((pose_data, single_data.reshape((1, -1))))
                i += 1
        except:
            print('To the end of this video')
            print(pose_data.shape)
        '''

        if cubic_interpolation or smooth:
            print('start cubic interpolation and/or smoothing...')
            for i in range(int(pose_data.shape[1] / 3)):
                x_data = pose_data[:, i * 3]
                y_data = pose_data[:, i * 3 + 1]
                confidence_data = pose_data[:, i * 3 + 2]
                if cubic_interpolation:
                    x_data = cubic_interpolate(x_data, confidence_data)
                    y_data = cubic_interpolate(y_data, confidence_data)
                if smooth:
                    x_data = savgol_filter(x_data, 11, 3)
                    y_data = savgol_filter(y_data, 11, 3)

                pose_data[:, i * 3] = x_data
                pose_data[:, i * 3 + 1] = y_data

        confidence_index = [i * 3 + 2 for i in range(int(pose_data.shape[1] / 3))]
        print('Removing confidence data from pose data...')
        pose_data = np.delete(pose_data, confidence_index, axis=1)
        print(pose_data.shape)

        # print('Reading OpenFace data...')
        # face_data = pd.read_csv(self.openface_output_file, header=0)
        # delay_compensation = 20
        # face_data = face_data.iloc[delay_compensation:, :]
        # face_success = face_data.iloc[:, 4].to_numpy()
        # face_data = face_data.iloc[:, 5:].to_numpy()
        #
        # # Reshape and cope the cordinates with original video
        # landmarks = face_data[:, 294:430]
        # eyes = face_data[:, 8:120]
        #
        # landmarks = landmarks / 1.5
        # eyes = eyes / 1.5
        # landmarks[:, :68] = landmarks[:, :68] + 449
        # landmarks[:, 68:] = landmarks[:, 68:] + 73
        # eyes[:, :68] = eyes[:, :68] + 449
        # eyes[:, 68:] = eyes[:, 68:] + 73
        #
        # face_data[:, 294:430] = landmarks
        # face_data[:, 8:120] = eyes
        #
        # print(face_data.shape)
        #
        # if cubic_interpolation or smooth:
        #     print('start cubic interpolation and/or smoothing...')
        #     for i in range(674):  # IMPORTANT: don't smooth/interpolate AUs
        #         # print('start cubic interpolation and/or smoothing...', 'feature No.', i)
        #         x_data = face_data[:, i]
        #         if cubic_interpolation:
        #             x_data = cubic_interpolate(x_data, face_success)
        #         # if smooth:
        #         #   x_data = savgol_filter(x_data, 31, 3)
        #         face_data[:, i] = x_data
        #
        # # Take the min length of existing data
        # min_length = min(face_data.shape[0], pose_data.shape[0])
        #
        # print('stacking arrays...')
        # data = np.hstack((pose_data[:min_length, :],
        #                   face_data[:min_length, :]))
        # print('final shape:', data.shape)

        data = pose_data
        print('final shape:', data.shape)
        if smooth:
            create_dirs([os.path.split(self.processed_smooth_file)[0]])
            np.save(self.processed_smooth_file, data)
        else:
            create_dirs([os.path.split(self.processed_file)[0]])
            np.save(self.processed_file, data)

    '''
    Demo functions
    '''
    def show_frames(self, starting, ending):
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
        t = starting
        print(data.shape)
        while (t < ending):
            ret, frame = cap.read()
            # Display all the data points

            for i in range(25):
                frame = self.paint_point(frame, [data[t, i * 2], data[t, i * 2 + 1]])
            for i in range(25, 95):
                frame = self.paint_point(frame, [data[t, i * 2], data[t, i * 2 + 1]], color=COLOR_BLUE)
            for i in range(95, 116):
                frame = self.paint_point(frame, [data[t, i * 2], data[t, i * 2 + 1]], color=COLOR_GREEN)
            for i in range(116, 137):
                frame = self.paint_point(frame, [data[t, i * 2], data[t, i * 2 + 1]], color=COLOR_YELLOW)

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

                '''
                cv2.rectangle(frame,
                              (x1, y1),
                              (x2, y2),
                              COLOR_YELLOW,
                              2)
                '''

            # Check hand-face overlapping
            intersect = self.check_overlap(left_hand_data, face_data)
            if intersect is not None:
                x1, y1, x2, y2 = intersect.get_cordinate()
                cv2.rectangle(frame,
                              (x1, y1),
                              (x2, y2),
                              COLOR_GREEN,
                              2)

            intersect = self.check_overlap(right_hand_data, face_data)
            if intersect is not None:
                x1, y1, x2, y2 = intersect.get_cordinate()
                cv2.rectangle(frame,
                              (x1, y1),
                              (x2, y2),
                              COLOR_BLUE,
                              2)

            cv2.imshow('frame', frame)
            if cv2.waitKey(40) & 0xFF == ord('q'):
                break
            if t == starting:
                input()
            t += 1
        # except Exception as e:
        # print(e)

        cap.release()
        # cv2.destroyAllWindows()

    def optical_flow_demo(self, play_starting=0):
        '''
        This function demonstrates the result of optical flow analyses
        :param play_starting:
        :return:
        '''
        label_data = json.load(open('optical_flow_result.json', 'r'))
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
        label_array = np.zeros((data.shape[0], 2))
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

            label_array[t, :] = np.array([avg_fft, avg_std]).reshape((1, -1))

        continuous_segments, valid_intersect_data = self.compute_stationary_rectangles(cutoff=10)

        best_rects = {}
        for segment in continuous_segments:
            # find largest rectangle for each segment
            rects = [valid_intersect_data[i] for i in valid_intersect_data.keys()
                     if i >= segment[0] and i < segment[1]]
            rects = np.array(rects)
            best_rect = np.hstack((np.min(rects, axis=0)[:2], np.max(rects, axis=0)[2:]))
            for i in range(segment[0], segment[1]):
                best_rects[i] = best_rect

        print(continuous_segments)

        cap = cv2.VideoCapture(self.video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        print('fps:', fps)
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print('Reading speaker info...')
        speaker_data = json.load(open(self.path_data['speaker_data'], 'r'))
        speaker_array = np.full((data.shape[0], 1), -1)

        for spk in speaker_data.keys():
            sub_spk_data = speaker_data[spk]
            spk = int(spk)
            print(spk, '--->')
            for segment in sub_spk_data:
                starting = segment[0]
                ending = segment[1]
                starting = math.floor(starting / 1000 * fps)
                starting = min(starting, speaker_array.shape[0])
                ending = math.ceil(ending / 1000 * fps)
                ending = min(ending, speaker_array.shape[0])
                speaker_array[starting:ending, :] = spk
                print(starting, ending)

        speaker_array = speaker_array.reshape(-1)
        # for spk in speaker_data.keys():
        #      spk = int(spk)
        #      print(spk, np.count_nonzero(speaker_array == spk))
        #         participant_speak_label_id = -1
        # participant_speak_max_frames = 0
        # for spk in speaker_data.keys():
        #     spk = int(spk)
        #     print(spk, np.count_nonzero(speaker_array == spk))
        #     if np.count_nonzero(speaker_array == spk) > participant_speak_max_frames:
        #         participant_speak_label_id = spk
        #         participant_speak_max_frames = np.count_nonzero(speaker_array == spk)
        #
        # print('participant_speak_id', participant_speak_label_id)

        # Read label data
        print('Reading fidgeting label data...')
        fidgeting_data = self.get_label()['fidgeting']
        hand_fidgeting_data = [x for x in fidgeting_data if x[0] == 'Hand']
        hand_fidgeting_array = np.full((data.shape[0], 1), 0)
        for item in hand_fidgeting_data:
            starting = int(item[1] * fps)
            ending = int(item[2] * fps)
            hand_fidgeting_array[starting:ending, :] = 1

        input('ready to play!')

        # Create some random colors
        # color = np.random.randint(0, 255, (100, 3))
        cap.set(1, play_starting)
        t = play_starting

        while (cap.isOpened()):
            # print(t)
            ret, frame = cap.read()
            videotime = t / fps  # video time in secs
            # print('current time:', videotime)
            # Read and print speaker data

            try:
                if int(speaker_array[t]) in speaker_label_data[self.participant_id][self.session_id]:
                    frame = self.paint_text(frame, 'Speaking', (400, 200), font_size=1)
                # elif int(speaker_array[t]) == -1:
                #    frame = self.paint_text(frame, 'Unknown', (400, 200), font_size=1)
                # else:
                #    frame = self.paint_text(frame, 'Interviewer speaking', (400, 200), font_size=1)
            except:
                frame = self.paint_text(frame,
                                        '{} of {} speaking'.format(str(speaker_array[t]), len(speaker_data.keys()) - 1),
                                        (100, 100))

            left_hand_data = data[t, 194:232].reshape(-1, 2)
            right_hand_data = data[t, 236:274].reshape(-1, 2)
            face_data = data[t, 50:190].reshape(-1, 2)

            # frame = self.paint_rectangle_to_points(frame, left_hand_data, color=COLOR_GREEN)
            # frame = self.paint_rectangle_to_points(frame, right_hand_data, color=COLOR_YELLOW)

            # # Check hands overlapping
            # intersect = self.check_overlap(left_hand_data, right_hand_data)
            # if intersect is not None:
            #     x1, y1, x2, y2 = intersect.get_cordinate()
            #     frame = self.paint_rectangle_to_points(frame,
            #                                           np.vstack((left_hand_data, right_hand_data)),
            #                                           color=COLOR_GREEN)
            #     cv2.rectangle(frame,
            #                   (x1, y1),
            #                   (x2, y2),
            #                   COLOR_BLUE,
            #                   2)
            if hand_fidgeting_array[t, 0] == 1:
                frame = self.paint_text(frame, 'Hand fidgeting', (400, 400), font_size=2)

            if t in valid_intersect_data.keys():
                # print(t)
                rect = best_rects[t]
                avg_fft = label_array[t, 0]
                avg_std = label_array[t, 1]
                color = COLOR_YELLOW
                if (avg_fft != 0 and avg_std != 0):
                    # print(avg_fft, avg_std)
                    if avg_fft >= FFT_thres or avg_std >= STD_thres:
                        color = COLOR_RED
                    else:
                        color = COLOR_GREEN
                if avg_fft >= FFT_thres and avg_std >= STD_thres:
                    frame = self.paint_text(frame, 'Rhythmic + Move', (rect[2], rect[3]), font_size=0.5)
                elif avg_fft >= FFT_thres and avg_std < STD_thres:
                    frame = self.paint_text(frame, 'Rhythmic', (rect[2], rect[3]), font_size=0.5)
                elif avg_fft < FFT_thres and avg_std >= STD_thres:
                    frame = self.paint_text(frame, 'Move', (rect[2], rect[3]), font_size=0.5)
                elif avg_fft < FFT_thres and avg_std < STD_thres:
                    frame = self.paint_text(frame, 'Stable', (rect[2], rect[3]), font_size=0.5)

                frame = cv2.rectangle(frame,
                                      (rect[0] - 10, rect[1] - 10),
                                      (rect[2] + 10, rect[3] + 10),
                                      color,
                                      2)

            cv2.imshow('frame', frame)
            k = cv2.waitKey(40) & 0xff
            if k == 27:
                break

            t += 1
            # input()
        cv2.destroyAllWindows()
        cap.release()
