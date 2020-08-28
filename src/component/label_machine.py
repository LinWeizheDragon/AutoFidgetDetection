
from utility.base_config import *

from component.basic_processor import BasicProcessor
from utility.decompose_string import decompose_string, decompose_string_hand

class LabelMachine(BasicProcessor):

    def __init__(self, name, path_data, label_type='hand_cross'):
        BasicProcessor.__init__(self, name, path_data, None)
        self.label_type = label_type

    def start_labelling(self):
        if self.label_type == 'hand_cross':
            Label_Folder = 'hand_cross_analysis_optical_flow_label'
        elif self.label_type == 'leg':
            Label_Folder = 'leg_action_analysis_optical_flow_label'
        elif self.label_type == 'left_hand':
            Label_Folder = 'hand_action_analysis_optical_flow_label'
        elif self.label_type == 'right_hand':
            Label_Folder = 'hand_action_analysis_optical_flow_label'
        data = {}
        for root, dirs, files in os.walk(os.path.join(DATA_FOLDER, Label_Folder)):
            for file in files:
                if self.label_type == 'left_hand':
                    if '_right' in file:
                        continue
                if self.label_type == 'right_hand':
                    if '_left' in file:
                        continue
                if '.npy' in file:
                    data[file] = np.load(os.path.join(root, file))

        i = 0
        for file_name in data.keys():
            i += 1
            if self.label_type == 'left_hand' or self.label_type == 'right_hand':
                participant_id, session_id, starting, ending, hand = decompose_string_hand(file_name)
            else:
                participant_id, session_id, starting, ending = decompose_string(file_name)
            sub_data = data[file_name]
            label_file_path = os.path.join(DATA_FOLDER, Label_Folder, file_name).replace('.npy', '.label2')
            if not os.path.exists(label_file_path):
                try:
                    # start labelling
                    print('currently working on:', file_name)
                    print('no:', str(i))
                    sub_pipeline = BasicProcessor('data', participant_data[participant_id][session_id])

                    FFT, STD, MEAN = self.analyse_sequence_new(self.get_first_derivative(sub_data))
                    print(np.mean(FFT, axis=1))
                    print(np.mean(STD))
                    print(np.mean(MEAN))

                    num = ''
                    if np.max(np.mean(FFT, axis=1)) < 15:
                        num = '0'
                        print('auto skipped')

                    while num == '':
                        sub_pipeline.show_frames(starting, ending)
                        num = input('input option: blank for replay, 0 for static, 1 for dynamic, 2 for rhythmic')
                        if num == 'q':
                            return
                except:
                    num = '-1'
                with open(label_file_path, 'w') as f:
                    print('saving to', label_file_path)
                    f.write(num)


            else:
                print('already labelled:', file_name)


