'''
Base Config file
    This file processes necessary indexing info for all participants.
    Edit this file to load your own data.
'''

import os
import re
import pandas as pd
import numpy as np


RAW_DATA_FOLDER = '../data/raw_data'
DATA_FOLDER = '../data/experiment_data'


openpose_data_config = [
    ('pose', 0, 50),  # all x & all y
    ('face', 50, 190),
    ('left_hand', 190, 232),  # 194:232
    ('right_hand', 232, 274),  # 236:274
    ('gaze', 274, 282),
    ('eye_lmk', 282, 394),
    ('eye_lmk_3d', 394, 562),
    ('head_pose', 562, 568),
    ('landmarks', 568, 704),
    ('landmarks_3d', 704, 908),
    ('PDM', 908, 948),
    ('AUs', 948, 983),  #
]

def string2score(string):
    if 'Several days' in string:
        return 1
    if 'Not at all' in string:
        return 0
    if 'More than half the days' in string:
        return 2
    if 'Nearly every day' in string:
        return 3

print('initializing participant data...')

# Here process the indexing for participant data
participant_data = {}
for root, dirs, files in os.walk(os.path.join(RAW_DATA_FOLDER, 'all_video')):
    for file in files:
        if '.avi' in file:
            id_string = file.replace('participant_video_', '').replace('.avi', '')
            session_id = 0
            print(id_string)
            participant_id = int(id_string)
            participant_data.setdefault(participant_id, {}) \
                .setdefault(session_id, {
                'video': os.path.join(RAW_DATA_FOLDER, 'all_video', file),
                'crop_video': os.path.join(DATA_FOLDER, 'all_video_crop_high', file),
                'generated_demo_video': os.path.join(DATA_FOLDER, 'generated_demo_video', file),
                'audio': os.path.join(DATA_FOLDER, 'all_video_audio', file.replace('.avi', '.wav')),
                'fused_data': os.path.join(DATA_FOLDER, 'fused_data', file.replace('.avi', '.npy')),
                'training_data': os.path.join(DATA_FOLDER, 'training_data', file.replace('.avi', '.pkl')),
                'processed_data': os.path.join(DATA_FOLDER, 'processed_data', file.replace('.avi', '.npy')),
                'processed_data_smooth': os.path.join(DATA_FOLDER, 'processed_data_smooth', file.replace('.avi', '.npy')),
                'full_fused_data': os.path.join(DATA_FOLDER, 'full_fused_data', file.replace('.avi', '.npy')),
                'speaker_data': os.path.join(DATA_FOLDER, 'speaker_data', file.replace('.avi', '.wav.json')),
                'voice_data': os.path.join(DATA_FOLDER, 'all_video_audio', file.replace('.avi', '_voice.json')),
                'mfcc_data': os.path.join(DATA_FOLDER, 'all_video_audio', file.replace('.avi', '.wav_st.npy')),
                'openpose_data': os.path.join(DATA_FOLDER, 'numpy_openpose_data',
                                              'p_{}'.format(str(participant_id))),
                'openface_data': os.path.join(DATA_FOLDER, 'openface_output', file.replace('.avi', '.csv')),
                'meta_feature_data': os.path.join(DATA_FOLDER, 'meta_feature_files',
                                                  'p_{}'.format(str(participant_id))),
                'file_format': file.replace('.avi', ''),
                'participant_id': participant_id,
                'session_id': session_id,
            })
print(participant_data.keys())


'''
The following is disabled as you need our data to load scores.
'''
# Here load depression label
participant_depression_data = {}
# data = pd.read_excel(os.path.join(RAW_DATA_FOLDER, 'interview_questionnaire_responses.xlsx'))
# data = data.to_numpy()
#
# for i in range(data.shape[0]):
#     participant_id = data[i, 81]
#     #print(data[i, 3:11])
#     score = []
#     for j in range(3,11):
#         score.append(string2score(data[i, j]))
#     participant_depression_data[participant_id] = np.sum(np.array(score))
#     # if np.sum(np.array(score)) >= 10:
#     #     participant_score_data[participant_id] = 1
#     # else:
#     #     participant_score_data[participant_id] = 0

# Here load anxiety label
participant_anxiety_data = {}
# for i in range(data.shape[0]):
#     participant_id = data[i, 81]
#     #print(data[i, 3:11])
#     score = []
#     for j in range(11,18):
#         score.append(string2score(data[i, j]))
#     participant_anxiety_data[participant_id] = np.sum(np.array(score))

print('initializing participant data...done')
