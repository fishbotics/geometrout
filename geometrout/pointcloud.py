import numpy as np


def project(transformation_matrix, point, rotate_only=False):
    """
    :param transformation_matrix: The matrix to multiply by the point
    :param point: The point to project into the new coordinate frame
    :param rotate_only: A flag to indicate whether to only rotate the point
        instead of moving its origin
    :return: The transformed point
    """
    if rotate_only:
        return (transformation_matrix @ np.append(point, [0]))[:3]
    return (transformation_matrix @ np.append(point, [1]))[:3]


def transform(pc, transformation_matrix, in_place=True):
    """

    Parameters
    ----------
    :param pc: A numpy pointcloud, maybe with some addition dimensions.
        This should have shape N x [3 + M] where N is the number of points
        are some additional mask dimensions or whatever, but the 3 are x-y-z
    :param transformation_matrix: A 4x4 homography
    :param in_place: A flag which says whether to do this in place
    :return: A pointcloud that has been transformed,
        either the same as the input or a new one.
    """
    assert type(pc) == type(transformation_matrix)
    assert pc.ndim == transformation_matrix.ndim
    if pc.ndim == 3:
        N, M = 1, 2
    elif pc.ndim == 2:
        N, M = 0, 1
    else:
        raise Exception("Pointcloud must have dimension Nx3 or BxNx3")
    xyz = pc[..., :3]
    ones_dim = list(xyz.shape)
    ones_dim[-1] = 1
    ones_dim = tuple(ones_dim)
    homogeneous_xyz = np.concatenate((xyz, np.ones(ones_dim)), axis=M)
    transformed_xyz = np.dot(
        transformation_matrix, np.transpose(homogeneous_xyz, (M, N))
    )
    if in_place:
        pc[:, :3] = np.transpose(transformed_xyz[..., :3, :], (M, N))
        return pc
    return np.concatenate(
        (np.transpose(transformed_xyz[..., :3, :], (M, N)), pc[..., 3:]), axis=1
    )
