import os
import numpy as np
import glob
import re


# Reads all people reshaped data from a directory. Returns format:
# {(person_id) int: (read_reshaped_directory output)}
def read_reshaped_directory_all_people(directory):
    assert os.path.isdir(directory), "Must be a directory to be able to read"
    person_id_re = re.compile("^person_(\d+)_\w+_\d+\.npy$")
    people_ids = set()
    for comp in glob.glob("{}/person_*.npy".format(directory)):
        match = person_id_re.match(os.path.basename(comp))
        assert match is not None, "Must be able to match person regex, {}".format(comp)
        pid = int(match.group(1))
        people_ids.add(pid)

    return {
        pid: read_reshaped_directory(directory, pid) for pid in people_ids
    }


# Reads numpy arrays from a preprocessed reshaped directory.
# Returns an array of form: {
#   frame_index_start: {
#     key (i.e. "pose|face|left|right": [(OpenPose joint positions)
#       [(position at frame) tuple of (x, y, confidence) of floats]
#     ]
#   }
# }
def read_reshaped_directory(directory, person_index):
    assert os.path.isdir(directory), "Must be a directory to be able to read"
    file_re = re.compile("^person_{}_(\w+)_(\d+)\.npy$".format(person_index))
    blocks = {}
    for comp in glob.glob("{}/person_{}_*".format(directory, person_index)):
        match = file_re.match(os.path.basename(comp))
        assert match is not None, "Must be able to match person regex, {}".format(comp)
        frame_start = int(match.group(2))
        if frame_start not in blocks:
            blocks[frame_start] = {}
        blocks[frame_start][match.group(1)] = np.load(comp)
    return blocks
