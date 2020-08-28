import os
import json
import cv2
from utility.base_config import *
from utility.colors import *

from component.basic_processor import BasicProcessor
from utility.decompose_string import decompose_string


class LegActionAnalyser(BasicProcessor):

    def __init__(self, name, path_data):
        BasicProcessor.__init__(self, name, path_data, None)

    def compute_static_and_rhythmic_feet(self):
        FFT_thres_1 = 200
        STD_thres_1 = 8
        FFT_thres_2 = 100
        STD_thres_2 = 8

        data = np.load(self.processed_smooth_file)
        final_label_array = np.zeros((data.shape[0], 2))



        for foot in ['left', 'right']:
            print('analysing', foot, '...')
            # foot + knee
            label_array = np.zeros((data.shape[0], 2))

            cap = cv2.VideoCapture(self.video_path)

            # try:
            length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            # cap.set(1, 50)
            print(length)
            t = 0

            def get_first_derivative(X_0th):
                time = 0.04
                X_1st = np.zeros((X_0th.shape[0] - 1, X_0th.shape[1]))
                for i in range(X_0th.shape[0] - 1):
                    X_1st[i] = (X_0th[i + 1] - X_0th[i]) / time
                return X_1st

            while (t < data.shape[0] - 25):
                ret, frame = cap.read()

                if t < 25:
                    print('skip')
                    t += 1
                    continue
                print('progress', t / data.shape[0], end='\r')
                if foot == 'right':
                    foot_data = data[t - 25:t + 25, 44:50]
                    knee_data = data[t - 25:t + 25, 20:22]
                else:
                    foot_data = data[t - 25:t + 25, 38:44]
                    knee_data = data[t - 25:t + 25, 26:28]


                #frame = self.paint_rectangle_to_points(frame, foot_data[50, :].reshape((-1, 2)), color=COLOR_RED)


                avg_fft, avg_std = self.analyse_sequence(get_first_derivative(foot_data))
                if avg_fft >= FFT_thres_1 and avg_std >= STD_thres_1:
                    label_array[t, 0] = 3  # dynamic + rhythmic
                    frame = self.paint_rectangle_to_points(frame, foot_data[25, :].reshape((-1, 2)), color=COLOR_RED)
                elif avg_fft >= FFT_thres_1 and avg_std < STD_thres_1:
                    label_array[t, 0] = 2  # rhythmic
                    frame = self.paint_rectangle_to_points(frame, foot_data[25, :].reshape((-1, 2)), color=COLOR_YELLOW)
                elif avg_fft < FFT_thres_1 and avg_std >= STD_thres_1:
                    label_array[t, 0] = 1  # dynamic
                elif avg_fft < FFT_thres_1 and avg_std < STD_thres_1:
                    label_array[t, 0] = 0  # static
                    frame = self.paint_rectangle_to_points(frame, foot_data[25, :].reshape((-1, 2)), color=COLOR_GREEN)

                # print(avg_fft, avg_std)

                avg_fft, avg_std = self.analyse_sequence(get_first_derivative(knee_data))
                if avg_fft >= FFT_thres_2 and avg_std >= STD_thres_2:
                    label_array[t, 1] = 3  # dynamic + rhythmic
                    frame = self.paint_point(frame, knee_data[25, :].reshape(-1).tolist(), color=COLOR_RED)
                elif avg_fft >= FFT_thres_2 and avg_std < STD_thres_2:
                    label_array[t, 1] = 2  # rhythmic
                    frame = self.paint_point(frame, knee_data[25, :].reshape(-1).tolist(), color=COLOR_YELLOW)
                elif avg_fft < FFT_thres_2 and avg_std >= STD_thres_2:
                    label_array[t, 1] = 1  # dynamic
                elif avg_fft < FFT_thres_2 and avg_std < STD_thres_2:
                    label_array[t, 1] = 0  # static
                    frame = self.paint_point(frame, knee_data[25, :].reshape(-1).tolist(), color=COLOR_GREEN)

                # print(avg_fft, avg_std, '\n')

                # cv2.imshow('frame', frame)
                k = cv2.waitKey(40) & 0xff
                if k == 27:
                    break

                t += 1
            cv2.destroyAllWindows()
            cap.release()

            if foot == 'left':
                final_label_array[:, 0] = np.max(label_array, axis=1)
            else:
                final_label_array[:, 1] = np.max(label_array, axis=1)

        for x in range(4):
            # compute continuous segment
            continuous_segments = []
            for i in range(data.shape[0]):
                if np.max(final_label_array[i, :]) == x:
                    if len(continuous_segments) == 0:
                        continuous_segments.append([i, i + 1])
                    else:
                        if continuous_segments[-1][1] == i:
                            continuous_segments[-1][1] += 1
                        else:
                            continuous_segments.append([i, i + 1])
            if x == 0:
                foot_static_segments = continuous_segments
            if x == 1:
                foot_dynamic_segments = continuous_segments
            if x == 2:
                foot_rhythmic_segments = continuous_segments
            if x == 3:
                foot_dynamic_rythmic_segments = continuous_segments
        return foot_static_segments, foot_dynamic_segments, foot_rhythmic_segments, foot_dynamic_rythmic_segments

    def analyse_leg_action_optical_flow(self):
        data = {}
        for root, dirs, files in os.walk(os.path.join(DATA_FOLDER, 'leg_action_analysis_optical_flow')):
            for file in files:
                if '.npy' in file:
                    data[file] = np.load(os.path.join(root, file))
                    if data[file].shape[0] == 0:
                        print(file, data[file].shape)

        label_data = {}
        from keras.models import load_model
        model = load_model(
            os.path.join(DATA_FOLDER, 'pre-trained', 'hierarchical_DNN_leg.h5')
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
            print(FFT)
            print(label)
            label_data.setdefault(participant_id, {}).setdefault(session_id, {})['{},{}'.format(starting, ending)] = label

        json.dump(label_data, open(
            os.path.join(DATA_FOLDER, 'leg_action_analysis_optical_flow', 'optical_flow_result.json'),
            'w'))
        print('saving completed.')
