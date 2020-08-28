# -*- coding: utf-8 -*-

import sys, os

import subprocess
import os.path

DATA_PATH = '...'
working_dir = '...'
BUILD_DIR = 'build'
print('Start Processing data')
#subprocess.check_call(['./cmake-build-debug/examples/openpose/openpose.bin', '--video', './examples/media/video.avi'], cwd=working_dir)

all_folders = [x[0] for x in os.walk(DATA_PATH)]
for sub_folder in all_folders:
    print(sub_folder)
    file_path = sub_folder + '/participant_video.mp4'
    if os.path.exists(file_path):
        if os.path.isfile(file_path):
            print('FOUND', file_path)
            # if the video file is not finished, start the process
            if not os.path.exists(sub_folder + '/openpose_output/output.avi'):
                try:
                    subprocess.check_call(
                        ['./' + BUILD_DIR + '/examples/openpose/openpose.bin',
                         '--video', file_path,
                         '--hand',
                         '--face',
                         '--write_video', sub_folder + '/openpose_output/output.avi',
                         '--write_json', sub_folder + '/openpose_output/',
                         ],
                        cwd=working_dir)
                except Exception as e:
                    print(e)
