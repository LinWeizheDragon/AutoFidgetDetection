from global_settings import DATA_FOLDER, participant_data
import numpy as np
import cv2
import os

def consistency(tolerance = 0):
    for particitant in participant_data.keys():
        for session in participant_data[particitant].keys():
            len_list = []
            data_path = os.path.join(participant_data[particitant][session]['openpose_data'], 'person_0_face_0.npy')
            data = np.load(data_path)
            data_len = np.shape(data)[1]

            video_path = participant_data[particitant][session]['video']
            cap = cv2.VideoCapture(video_path)
            video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            if abs(data_len - video_length) > tolerance:
                len_list.append(data_len)
                len_list.append(video_length)
                print('participant_{}_session_{}'.format(particitant, session), len_list)