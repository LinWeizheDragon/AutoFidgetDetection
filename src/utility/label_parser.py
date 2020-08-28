import pandas as pd
import numpy as np



class Label_Parser():
    def __init__(self, file_path):
        self.file_path = file_path

    def parse(self):
        print(self.file_path)
        data = pd.read_csv(self.file_path, header=None, delimiter='\t')
        data = data.iloc[:, [0, 3, 5, 8]]
        return data.values.tolist()