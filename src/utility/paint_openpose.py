import cv2
import os
import numpy as np
from utility.colors import COLOR_YELLOW
from data_processing.utility import BODY_CONNECTION
from utility.painting import paint_point, paint_text, paint_line, paint_rectangle_to_points
from global_setting import DATA_FOLDER, participant_data

class paint_openpose():
    def __init__(self, path_data):
        if path_data is not None:
            # single video processing
            self.path_data = path_data
            self.video_path = path_data['video']
            self.openpose_output_path = path_data['openpose_data']
            self.openface_output_file = path_data['openface_data']
            self.processed_file = os.path.join(DATA_FOLDER, 'processed_data_smooth',
                                          path_data['file_format'] + '.npy')

            self.save_dir = '/Volumes/Seagate_Backup/paint_openpose'


    def plot_pose_hand_keypoints(self, play_starting=0):
        cap = cv2.VideoCapture(self.video_path)
        data = np.load(self.processed_file)

        cap.set(1, play_starting)
        t = play_starting
        while (cap.isOpened()):
            ret, frame = cap.read()
            left_hand_data = data[t, 194:232].reshape(-1, 2)
            right_hand_data = data[t, 236:274].reshape(-1, 2)

            for i in range(np.shape(left_hand_data)[0]):
                frame = paint_point(frame, left_hand_data[i], color=COLOR_YELLOW)
                frame = paint_point(frame, right_hand_data[i], color=COLOR_YELLOW)

            frame = paint_rectangle_to_points(frame, left_hand_data, color=(0, 255, 0))
            frame = paint_rectangle_to_points(frame, right_hand_data, color=(0, 255, 0))

            for i in range(0, 25):
                frame = paint_point(frame, [data[t, 2 * i], data[t, 2 * i + 1]], color=COLOR_YELLOW)
                frame = paint_text(frame, str(i), [data[t, 2 * i], data[t, 2 * i + 1]], color=COLOR_YELLOW)

            for connection in BODY_CONNECTION:
                point1 = connection[0]
                point2 = connection[1]
                frame = paint_line(frame,
                                   [data[t, 2 * point1], data[t, 2 * point1 + 1]],
                                   [data[t, 2 * point2], data[t, 2 * point2 + 1]],
                                   color=COLOR_YELLOW)

            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            # if t == 0:
            # input()
            t += 1

        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    for participant in participant_data.keys():
        for session in participant_data[participant].keys():
            instance = paint_openpose(participant_data[participant][session])
            instance.plot_pose_hand_keypoints(play_starting=0)