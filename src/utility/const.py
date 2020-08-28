POSE_BODY_25_BODY_PARTS = [
    {0, "Nose"},
    {1, "Neck"},
    {2, "RShoulder"},
    {3, "RElbow"},
    {4, "RWrist"},
    {5, "LShoulder"},
    {6, "LElbow"},
    {7, "LWrist"},
    {8, "MidHip"},
    {9, "RHip"},
    {10, "RKnee"},
    {11, "RAnkle"},
    {12, "LHip"},
    {13, "LKnee"},
    {14, "LAnkle"},
    {15, "REye"},
    {16, "LEye"},
    {17, "REar"},
    {18, "LEar"},
    {19, "LBigToe"},
    {20, "LSmallToe"},
    {21, "LHeel"},
    {22, "RBigToe"},
    {23, "RSmallToe"},
    {24, "RHeel"},
    {25, "Background"}
]

BODY_CONNECTION = [
    (0, 15), # Face 0
    (0, 16), # Face 1
    (15, 17), # Face 2
    (16, 18), # Face 3
    (0, 1), # Upper 4
    (1, 2), # Upper 5
    (2, 3), # Upper 6
    (3, 4), # Upper 7
    (1, 5), # Upper 8
    (5, 6), # Upper 9
    (6, 7), # Upper 10
    (1, 8), # Upper 11
    (8, 9), # Lower 12
    (9, 10), # Lower 13
    (10, 11), # Lower 14
    (11, 24), # Lower 15
    (11, 22), # Lower 16
    (22, 23), # Lower 17
    (8, 12), # Lower 18
    (12, 13), # Lower 19
    (13, 14), # Lower 20
    (14, 21), # Lower 21
    (14, 19), # Lower 22
    (19, 20), # Lower 23
]
