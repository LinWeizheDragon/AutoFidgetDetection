import numpy as np
import pandas as pd


class ElanPortal():
    def __init__(self):
        self.data = {}

    def add_tier(self, tier_name, type, tier_data):
        '''
        This function add data to portal
        :param tier_name:
        :param tier_data:
        :return:
        '''
        self.data.setdefault(tier_name, {}).setdefault(type, [])
        self.data[tier_name][type] = self.data[tier_name][type] + tier_data

    def export(self, file_path):
        '''
        This function exports segments data to txt file
        :param file_path:
        :return:
        '''
        data = {
            'tier_name': [],
            'start_time': [],
            'end_time': [],
            'type': [],
        }
        for tier_name in self.data.keys():
            for type in self.data[tier_name]:
                for segment in self.data[tier_name][type]:
                    data['tier_name'].append(tier_name)
                    data['start_time'].append(segment[0])
                    data['end_time'].append(segment[1])
                    data['type'].append(type)
        df = pd.DataFrame(data)
        print(df)
        df.to_csv(file_path, index=None, header=None, sep='\t')

    def read(self, file_path, fps):
        self.data = {}
        print('reading elan data...')
        df = pd.read_csv(file_path, sep='\t', header=None)
        for i in range(df.shape[0]):
            record = df.iloc[i, :]
            tier_name = record[0]
            start_time = record[1]
            end_time = record[2]
            type = record[3]
            segment = [int(start_time * fps), int(end_time * fps)]
            self.data.setdefault(tier_name, {}).setdefault(type, []).append(segment)

        print('reading elan data...done')

    def get_segments(self, tier_name, type):
        '''
        This function get segments list by tier_name and type
        :param tier_name:
        :param type:
        :return:  list of segments
        '''
        try:
            segments = self.data[tier_name][type]
            return segments
        except:
            return []
