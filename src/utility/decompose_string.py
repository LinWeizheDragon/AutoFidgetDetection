
def decompose_string(descriptor):
    descriptor = descriptor.replace('.npy', '').replace('participant_video_', '')
    participant_id = int(descriptor.split('_')[0])
    if len(descriptor.split('_')[1]) == 1:
        session_id = int(descriptor.split('_')[1])
    else:
        session_id = 0
    frames = descriptor.split('<')[1].split('>')[0].split('_')
    starting = int(frames[0])
    ending = int(frames[1])
    return participant_id, session_id, starting, ending

def decompose_string_hand(descriptor):
    hand = 'None'
    if '_left' in descriptor:
        hand = 'left'
        descriptor = descriptor.replace('_left', '')
    if '_right' in descriptor:
        hand = 'right'
        descriptor = descriptor.replace('_right', '')

    participant_id, session_id, starting, ending = decompose_string(descriptor)
    return participant_id, session_id, starting, ending, hand
