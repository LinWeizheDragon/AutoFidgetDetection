import os
import json
import cv2
import math
import numpy as np
import pandas as pd
from utility.base_config import *
from utility.colors import *

from component.basic_processor import BasicProcessor



class OpticalFlowAnalyser(BasicProcessor):

    def __init__(self, name, path_data, smooth=False):
        BasicProcessor.__init__(self, name, path_data, None)
        if smooth:
            self.data = np.load(self.processed_smooth_file)
        else:
            self.data = np.load(self.processed_file)

    def run_optical_flow(self, cap, starting_time, ending_time, init_points, visualise=False):
        '''
        generate raw optical flow data from video
        :return:
        '''

        cap.set(1, starting_time)
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        data = self.data

        # params for ShiTomasi corner detection
        feature_params = dict(maxCorners=100,
                              qualityLevel=0.3,
                              minDistance=7,
                              blockSize=7)

        # Parameters for lucas kanade optical flow
        lk_params = dict(winSize=(15, 15),
                         maxLevel=2,
                         criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))


        # Take first frame
        ret, old_frame = cap.read()
        old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
        # p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
        t = starting_time
        # init p0
        p0 = init_points[t, :].reshape(-1, 2)
        p0 = np.float32(np.around(p0.reshape(-1, 1, 2)))

        # Create a mask image for drawing purposes
        mask = np.zeros_like(old_frame)

        # dict to save points data
        optical_flow_data = {}
        optical_flow_data[t] = p0
        t += 1
        try:
            while (t < ending_time):
                print('processing video...', t, 'of', data.shape[0], end='\r')
                ret, frame = cap.read()
                # cv2.imshow('frame', frame)
                frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                # cv2.imshow('frame_gray', frame_gray)

                p_ref = np.float32(np.around(init_points[t, :].reshape(-1, 1, 2)))

                # calculate optical flow
                p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, p_ref, **lk_params)

                # Select good points
                good_new = p1[st == 1]
                good_old = p0[st == 1]

                # draw the tracks
                for i, (new, old) in enumerate(zip(good_new, good_old)):
                    a, b = new.ravel()
                    c, d = old.ravel()
                    # mask = cv2.line(mask, (a, b), (c, d), color[i].tolist(), 2)
                    # frame = cv2.circle(frame, (a, b), 1, color[i].tolist(), -1)
                    frame = cv2.circle(frame, (a, b), 1, COLOR_YELLOW, -1)


                img = cv2.add(frame, mask)

                # Now update the previous frame and previous points
                old_gray = frame_gray.copy()
                p0 = good_new.reshape(-1, 1, 2)
                if p0.shape[0] != p_ref.shape[0]:
                    print('optical flow points missing.')
                optical_flow_data[t] = good_new

                if visualise:
                    cv2.imshow('frame', img)

                k = cv2.waitKey(10) & 0xff
                if k == 27:
                    break

                t += 1
                # input()
        except Exception as e:
            print(t, e)
            pass

        print('\n')

        return optical_flow_data
