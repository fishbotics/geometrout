import numba as nb
import numpy as np


@nb.jit(nopython=True, cache=True)
def rand_choice(arr, prob):
    """
    :param arr: A 1D numpy array of values to sample from.
    :param prob: A 1D numpy array of probabilities for the given samples.
    :return: A random sample from the given array with a given probability.
    """
    return arr[np.searchsorted(np.cumsum(prob), np.random.random(), side="right")]


@nb.jit(nopython=True, cache=True)
def transform_in_place(point_cloud, transformation_matrix):
    """

    Parameters
    ----------
    :param point_cloud: A numpy pointcloud, maybe with some addition dimensions.
        This should have shape N x [3 + M] where N is the number of points
        are some additional mask dimensions or whatever, but the 3 are x-y-z
    :param transformation_matrix: A 4x4 homography
    :param in_place: A flag which says whether to do this in place
    :return: A pointcloud that has been transformed,
        either the same as the input or a new one.
    """
    assert point_cloud.shape[1] == 3
    homogeneous_xyz = np.concatenate(
        (np.transpose(point_cloud), np.ones((1, point_cloud.shape[0]))), axis=0
    )
    transformed_xyz = np.dot(transformation_matrix, homogeneous_xyz)
    point_cloud[:, :3] = np.transpose(transformed_xyz[..., :3, :], (1, 0))
    return point_cloud
