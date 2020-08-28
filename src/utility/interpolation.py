import numpy as np
from scipy.interpolate import interp1d

def cubic_interpolate(collection, confidence):
    '''
    Cubic Interpolation
    :param collection:  np.array shape(n_frames,)
    :param confidence:  np.array shpae(n_frames,)
    :return: interpolated_collection: np.array shape(n_frames,)
    '''
    t_max = collection.shape[0]

    processed_collection = []
    time_axis = []
    for x in range(t_max):
        if confidence[x] != 0:
            processed_collection.append(collection[x])
            time_axis.append(x)
        #else:
        #    print('missing data point!')

    f = interp1d(np.array(time_axis), np.array(processed_collection), kind='cubic', fill_value='extrapolate')
    #plt.plot([t for t in range(t_max)], pose_x_collection[index], 'o', dense_time_axis, f(dense_time_axis), '--')
    #plt.plot(dense_time_axis, f(dense_time_axis))
    #$plt.show()
    #print(f(np.array([t for t in range(t_max)])))
    return f(np.array([t for t in range(t_max)]))
