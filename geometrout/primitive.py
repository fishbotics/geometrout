import numba as nb
import numpy as np

from geometrout.maths import transform_in_place
from geometrout.transform import SE3, _random_rotation


class Sphere:
    def __init__(self, center, radius):
        """
        Constructs an internal sphere representation

        :param center: The center of the sphere as a list of numpy array
        :param radius: The radius of the sphere as a number
        """
        self.center = center.astype(np.double)
        assert radius >= 0
        self.radius = radius

    def __repr__(self):
        return "\n".join(
            [
                "Sphere(",
                f"    center={repr(self.center)},",
                f"    radius={self.radius},",
                ")",
            ]
        )

    def copy(self):
        return Sphere(np.copy(self.center), self.radius)

    @classmethod
    def unit(cls):
        return cls(np.zeros(3), 1.0)

    @classmethod
    def random(
        cls,
        center_range=np.zeros((2, 3)),
        radius_range=np.ones(2),
    ):
        """
        Creates a random sphere.
        :param center_range: 2x3 numpy array or list with form
          [[x_low, y_low, z_low], [x_high, y_hight, x_max]].
          Nothing passed in will default at origin
        :param radius_range: List [r_low, r_high].
          Nothing passed in will default to radius of 1
        """
        center = (center_range[1, :] - center_range[0, :]) * np.random.rand(
            3
        ) + center_range[0, :]
        mn, mx = radius_range
        radius = (mx - mn) * np.random.rand() + mn
        return cls(center, radius)

    @property
    def surface_area(self):
        return _sphere_surface_area(self.radius)

    @property
    def volume(self):
        return _sphere_volume(self.radius)

    def is_zero_volume(self, atol=1e-7):
        return self.radius < atol

    def sdf(self, point):
        """
        :param point: Point in 3D for which we want the sdf
        :return: The sdf value of that point
        """
        return _sphere_sdf(self.center, self.radius, point)

    def sample_surface(self, num_points, noise=0.0):
        """
        Samples random points on the surface of the sphere. Probabilities are
        weighed based on area of each side.

        :param num_points: The number of points to sample on the surface
        :param noise: The range of uniform noise to apply to samples

        :return: A random pointcloud sampled from the surface of the cuboid
        """
        return _sphere_sample_surface(self.center, self.radius, num_points, noise)

    def sample_volume(self, num_points):
        """
        Get a random pointcloud sampled inside the sphere (including the surface)

        :param num_points: The number of points to sample
        :return: A set of points inside the sphere
        """
        return _sphere_sample_volume(self.center, self.radius, num_points)


class Cuboid:
    def __init__(self, center, dims, quaternion):
        """
        :param center: np.array([x, y, z])
        :param dims: np.array([x, y, z])
        :param quaternion: np.array([w, x, y, z])
        """

        # Note that the input type of these is arrays but I'm still casting.
        # This is because its easier to just case to numpy arrays than it is to
        # check for type
        self.pose = SE3(center.astype(np.double), quaternion.astype(np.double))
        self.dims = dims.astype(np.double)

    def copy(self):
        return Cuboid(
            np.copy(self.pose.pos), np.copy(self.dims), np.copy(self.pose.so3.q)
        )

    def __str__(self):
        return "\n".join(
            [
                "     +----------+",
                "    /          /|",
                "  x/          / |",
                "  /          /  |",
                " /    y     /   |",
                "+----------+    |",
                "|          |    /",
                "|          |   / ",
                "|         z|  /  ",
                "|          | /   ",
                "+----------+     ",
                f"Center: {repr(self.center)}",
                f"Dimensions: {repr(self.dims)}",
                f"Orientation (wxyz): {repr(self.pose.so3.q)}",
            ]
        )

    def __repr__(self):
        return "\n".join(
            [
                "Cuboid(",
                f"    center={repr(self.center)},",
                f"    dims={repr(self.dims)},",
                f"    quaternion={repr(self.pose.so3.q)},",
                ")",
            ]
        )

    @classmethod
    def unit(cls):
        return cls(
            center=np.array([0.0, 0.0, 0.0]),
            dims=np.array([1.0, 1.0, 1.0]),
            quaternion=np.array([1.0, 0.0, 0.0, 0.0]),
        )

    @classmethod
    def random(
        cls,
        center_range=np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
        dimension_range=np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]),
        random_orientation=True,
    ):
        """
        Creates a random cuboid within the given ranges
        :param center_range: If given, represents the uniform range from which to draw a center.
            Should be np.array with dimension 2x3. First row is lower limit, second row is upper limit
            If nothing is passed, center will always be at [0, 0, 0]
        :param dimension_range: If given, represents the uniform range from which to draw a center.
            Should be np.array with dimension 2x3. First row is lower limit, second row is upper limit
            If nothing is passed, dimensions defaults to [1, 1, 1]
        :param quaternion: If True, will give a random orientation to cuboid.
            If False, will be set as the identity
            Default is True
        :return: Cuboid object drawn from specified uniform distribution
        """
        center = (center_range[1, :] - center_range[0, :]) * np.random.rand(
            3
        ) + center_range[0, :]
        dims = (dimension_range[1, :] - dimension_range[0, :]) * np.random.rand(
            3
        ) + dimension_range[0, :]
        if random_orientation:
            quaternion = _random_rotation()
        else:
            quaternion = np.array([1.0, 0.0, 0.0, 0.0])
        return cls(center, dims, quaternion)

    def is_zero_volume(self, atol=1e-7):
        for x in self.dims:
            if np.abs(x) < atol:
                return True
        return False

    def sample_surface(self, num_points, noise=0.0):
        """
        Samples random points on the surface of the cube. Probabilities are
        weighed based on area of each side.

        :param num_points: The number of points to sample on the surface
        :param noise: The range of uniform noise to apply to samples

        :return: A random pointcloud sampled from the surface of the cuboid
        """
        return _cuboid_sample_surface(self.pose.matrix, self.dims, num_points, noise)

    def sample_volume(self, num_points):
        """
        Get a random pointcloud sampled inside the cuboid (including the surface)

        :param num_points: The number of points to sample
        :return: A set of points inside the cube
        """
        return _cuboid_sample_volume(self.pose.matrix, self.dims, num_points)

    def sdf(self, point):
        """
        :param point: Point in 3D for which we want the sdf
        :return: The sdf value of that point
        """
        return _cuboid_sdf(self.pose.inverse.matrix, self.dims, point)

    @property
    def center(self):
        """
        :return: The center of the object as a list
        """
        return self.pose.pos

    @property
    def quaternion(self):
        return self.pose.so3.q

    @property
    def half_extents(self):
        """
        :return: The half-dimensions of the cuboid. This is necessary for some interfaces.
        """
        return self.dims / 2

    @property
    def corners(self):
        return _cuboid_corners(self.pose.matrix, self.dims)

    @property
    def surface_area(self):
        return _cuboid_surface_area(self.dims)


class Cylinder:
    pose: SE3

    def __init__(self, center, radius, height, quaternion):
        assert radius >= 0 and height >= 0
        self.pose = SE3(center.astype(np.double), quaternion.astype(np.double))
        self.radius = radius
        self.height = height

    def copy(self):
        return Cylinder(
            np.copy(self.pose.pos), self.radius, self.height, np.copy(self.pose.so3.q)
        )

    def __repr__(self):
        return "\n".join(
            [
                "Cylinder(",
                f"    center={repr(self.center)},",
                f"    radius={self.radius},",
                f"    height={self.height},",
                f"    quaternion={repr(self.pose.so3.q)},",
                ")",
            ]
        )

    @property
    def center(self):
        """
        :return: The center of the object as a list
        """
        return self.pose.pos

    @classmethod
    def random(
        cls,
        center_range=np.array([[0.0, 0, 0], [0.0, 0, 0]]),
        radius_range=np.array([1.0, 1.0]),
        height_range=np.array([1.0, 1.0]),
        random_orientation=True,
    ):
        """
        Creates a random Cylinder.
        :param center_range: 2x3 numpy array or list with form
          [[x_low, y_low, z_low], [x_high, y_hight, x_max]].
          If nothing is passed, center will always be at [0, 0, 0]
        :param radius_range: List [r_low, r_high]. Pass in None for
          If nothing is passed, radius will be 1.0
        :param height_range: List [h_low, h_high]. Pass in None for
          If nothing is passed, height will be 1.0
        :param random_orientation: bool Whether to have a random orientation
        """
        center = (center_range[1, :] - center_range[0, :]) * np.random.rand(
            3
        ) + center_range[0, :]
        mn, mx = radius_range
        radius = (mx - mn) * np.random.rand() + mn
        mn, mx = height_range
        height = (mx - mn) * np.random.rand() + mn
        if random_orientation:
            quaternion = _random_rotation()
        else:
            quaternion = np.array([1.0, 0.0, 0.0, 0.0])
        return cls(center, radius, height, quaternion)

    @property
    def surface_area(self):
        return _cylinder_surface_area(self.radius, self.height)

    @property
    def volume(self):
        return _cylinder_volume(self.radius, self.height)

    def is_zero_volume(self, atol=1e-7):
        return self.radius < atol or self.height < atol

    def sdf(self, point):
        """
        :param point: Point in 3D for which we want the sdf
        :return: The sdf value of that point
        """
        return _cylinder_sdf(self.pose.inverse.matrix, self.radius, self.height, point)

    def sample_surface(self, num_points, noise=0.0):
        """
        Samples random points on the surface of the cylinder. Probabilities are
        weighed based on area of each side.

        :param num_points: The number of points to sample on the surface
        :param noise: The range of uniform noise to apply to samples

        :return: A random pointcloud sampled from the surface of the cuboid
        """
        return _cylinder_sample_surface(
            self.pose.matrix, self.radius, self.height, num_points, noise
        )

    def sample_volume(self, num_points, noise=0.0):
        """
        Get a random pointcloud sampled inside the cylinder (including the surface)

        :param num_points: The number of points to sample
        :return: A set of points inside the sphere
        """
        return _cylinder_sample_volume(
            self.pose.matrix, self.radius, self.height, num_points, noise
        )


class SphereArray:
    def __init__(self, spheres):
        self.centers = np.stack([s.center for s in spheres], axis=0)
        self.radii = np.stack([s.radius for s in spheres], axis=0)

    def all_sdfs(self, points):
        if points.ndim == 1:
            return _sphere_array_all_sdfs_1d(self.centers, self.radii, points)
        elif points.ndim == 2:
            return _sphere_array_all_sdfs_2d(self.centers, self.radii, points)

    def scene_sdf(self, points):
        if points.ndim == 1:
            return _sphere_array_scene_sdf_1d(self.centers, self.radii, points)
        elif points.ndim == 2:
            return _sphere_array_scene_sdf_2d(self.centers, self.radii, points)


class CuboidArray:
    def __init__(self, cuboids):
        self.centers = np.stack([c.center for c in cuboids], axis=0)
        self.dims = np.stack([c.dims for c in cuboids], axis=0)
        self.quaternions = np.stack([c.quaternion for c in cuboids], axis=0)
        self.inverse_pose_matrices = np.stack(
            [c.pose.inverse.matrix for c in cuboids], axis=0
        )

    def all_sdfs(self, points):
        if points.ndim == 1:
            return _cuboid_array_all_sdfs_1d(
                self.inverse_pose_matrices, self.dims, points
            )
        elif points.ndim == 2:
            return _cuboid_array_all_sdfs_2d(
                self.inverse_pose_matrices, self.dims, points
            )

    def scene_sdf(self, points):
        if points.ndim == 1:
            return _cuboid_array_scene_sdf_1d(
                self.inverse_pose_matrices, self.dims, points
            )
        elif points.ndim == 2:
            return _cuboid_array_scene_sdf_2d(
                self.inverse_pose_matrices, self.dims, points
            )


class CylinderArray:
    def __init__(self, cylinders):
        self.centers = np.stack([c.center for c in cylinders], axis=0)
        self.radii = np.stack([s.radius for s in cylinders], axis=0)
        self.heights = np.stack([s.height for s in cylinders], axis=0)
        self.inverse_pose_matrices = np.stack(
            [c.pose.inverse.matrix for c in cylinders], axis=0
        )

    def all_sdfs(self, points):
        if points.ndim == 1:
            return _cylinder_array_all_sdfs_1d(
                self.inverse_pose_matrices, self.radii, self.heights, points
            )
        else:
            return _cylinder_array_all_sdfs_2d(
                self.inverse_pose_matrices, self.radii, self.heights, points
            )

    def scene_sdf(self, points):
        if points.ndim == 1:
            return _cylinder_array_scene_sdf_1d(
                self.inverse_pose_matrices, self.radii, self.heights, points
            )
        else:
            return _cylinder_array_scene_sdf_2d(
                self.inverse_pose_matrices, self.radii, self.heights, points
            )


@nb.njit
def np_apply_along_axis(func1d, axis, arr):
    assert arr.ndim == 2
    assert axis in [0, 1]
    if axis == 0:
        result = np.empty(arr.shape[1])
        for i in range(len(result)):
            result[i] = func1d(arr[:, i])
    else:
        result = np.empty(arr.shape[0])
        for i in range(len(result)):
            result[i] = func1d(arr[i, :])
    return result


@nb.njit
def np_mean(array, axis):
    return np_apply_along_axis(np.mean, axis, array)


@nb.njit
def np_std(array, axis):
    return np_apply_along_axis(np.std, axis, array)


@nb.jit(nopython=True, cache=True)
def _sphere_surface_area(radius):
    return 4 * np.pi * radius**2


@nb.jit(nopython=True, cache=True)
def _sphere_volume(radius):
    return 4 / 3 * np.pi * radius**3


@nb.jit(nopython=True, cache=True)
def _sphere_sdf(center, radius, point):
    assert point.ndim < 3
    if point.ndim == 1:
        return np.linalg.norm(point - center) - radius
    distance2 = np.power(point - center, 2)
    return np.sqrt(distance2[:, 0] + distance2[:, 1] + distance2[:, 2]) - radius


@nb.jit(nopython=True, cache=True)
def _sphere_sample_surface(center, radius, num_points, noise):
    assert (
        noise >= 0
    ), "Noise parameter should be a radius of the uniform distribution added to the random points"
    points = np.random.uniform(-1.0, 1.0, (num_points, 3))
    for i in range(points.shape[0]):
        nrm = np.linalg.norm(points[i, :])
        points[i, :] /= nrm
    points = radius * points + center
    if noise > 0.0:
        noise = np.random.uniform(-noise, noise, points.shape)
        return points + noise
    return points


@nb.jit(nopython=True, cache=True)
def _sphere_sample_volume(center, radius, num_points):
    # First produce points on the surface of the unit sphere
    points = np.random.uniform(-1.0, 1.0, (num_points, 3))
    for i in range(points.shape[0]):
        nrm = np.linalg.norm(points[i, :])
        points[i, :] /= nrm

    # Now multiply them by random radii in the range [0, radius]
    radii = np.random.uniform(0, radius, (num_points, 1))
    return points * radii + center


@nb.jit(nopython=True, cache=True)
def _sphere_array_all_sdfs_1d(centers, radii, points):
    assert points.ndim == 1
    return np_apply_along_axis(np.linalg.norm, 1, points - centers) - radii


@nb.jit(nopython=True, cache=True)
def _sphere_array_all_sdfs_2d(centers, radii, points):
    assert points.ndim == 2
    vectors = points - np.expand_dims(centers, 1)
    distances = np.sqrt(
        vectors[:, :, 0] ** 2 + vectors[:, :, 1] ** 2 + vectors[:, :, 2] ** 2
    )
    return distances - np.expand_dims(radii, 1)


@nb.jit(nopython=True, cache=True)
def _sphere_array_scene_sdf_1d(centers, radii, point):
    all_sdfs = _sphere_array_all_sdfs_1d(centers, radii, point)
    return np.min(all_sdfs)


@nb.jit(nopython=True, cache=True)
def _sphere_array_scene_sdf_2d(centers, radii, points):
    sdfs = _sphere_array_all_sdfs_2d(centers, radii, points).T
    result = np.empty(len(sdfs))
    for i in range(len(result)):
        result[i] = np.min(sdfs[i, :])
    return result


@nb.jit(nopython=True, cache=True)
def _cuboid_sample_surface(
    pose_matrix,
    dims,
    num_points,
    noise,
):
    assert (
        noise >= 0
    ), "Noise parameter should be a radius of the uniform distribution added to the random points"
    random_points = np.random.uniform(-1.0, 1.0, (num_points, 3))
    random_points = random_points * dims / 2
    probs = np.array(
        [
            dims[1] * dims[2],
            dims[1] * dims[2],
            dims[0] * dims[2],
            dims[0] * dims[2],
            dims[0] * dims[1],
            dims[0] * dims[1],
        ]
    )
    probs /= np.sum(probs)
    sides = np.searchsorted(
        np.cumsum(probs), np.random.random(num_points), side="right"
    )
    random_points[sides == 0, 0] = dims[0] / 2
    random_points[sides == 1, 0] = -dims[0] / 2
    random_points[sides == 2, 1] = dims[1] / 2
    random_points[sides == 3, 1] = -dims[1] / 2
    random_points[sides == 4, 2] = dims[2] / 2
    random_points[sides == 5, 2] = -dims[2] / 2
    transform_in_place(random_points, pose_matrix)
    noise = 2 * noise * np.random.random_sample(random_points.shape) - noise
    return random_points + noise


@nb.jit(nopython=True, cache=True)
def _cuboid_sample_volume(pose_matrix, dims, num_points):
    random_points = np.random.uniform(-1.0, 1.0, (num_points, 3))
    random_points = random_points * dims / 2
    transform_in_place(random_points, pose_matrix)
    return random_points


@nb.jit(nopython=True, cache=True)
def _cuboid_sdf(inverse_pose_matrix, dims, point):
    assert 0 < point.ndim < 3
    if point.ndim == 1:
        homog_point = np.ones(4)
        homog_point[:3] = point
        projected_point = (inverse_pose_matrix @ homog_point)[:3]
        distance = np.abs(projected_point) - (dims / 2)
        outside = np.linalg.norm(np.maximum(distance, np.zeros(3)))
        inner_max_distance = np.max(distance)
        inside = np.minimum(inner_max_distance, 0)
    else:
        homog_point = np.ones((point.shape[0], 4))
        homog_point[:, :3] = point
        projected_point = (inverse_pose_matrix @ homog_point.T)[:3, :].T
        distance = np.abs(projected_point) - (dims / 2)
        _outside = np.power(np.maximum(distance, np.zeros((1, 3))), 2)
        outside = np.sqrt(_outside[:, 0] + _outside[:, 1] + _outside[:, 2])
        inner_max_distance = np_apply_along_axis(np.max, 1, distance)
        inside = np.minimum(inner_max_distance, 0)
    return outside + inside


@nb.jit(nopython=True, cache=True)
def _cuboid_array_all_sdfs_1d(inverse_pose_matrices, dims, point):
    assert point.ndim == 1
    homog_point = np.ones(4)
    homog_point[:3] = point
    projected_point = np.empty((inverse_pose_matrices.shape[0], 3))
    for i in range(len(inverse_pose_matrices)):
        projected_point[i, :] = np.dot(inverse_pose_matrices[i], homog_point)[:3]
    distance = np.abs(projected_point) - (dims / 2)
    distance_max = np.maximum(distance, np.zeros((dims.shape[0], 3)))
    outside = np.sqrt(
        distance_max[:, 0] ** 2 + distance_max[:, 1] ** 2 + distance_max[:, 2] ** 2
    )
    inner_max_distance = np.empty(distance.shape[0])
    for i in range(len(inner_max_distance)):
        inner_max_distance[i] = np.max(distance[i, :])
    inside = np.minimum(inner_max_distance, 0)
    return outside + inside


@nb.jit(nopython=True, cache=True)
def _cuboid_array_all_sdfs_2d(inverse_pose_matrices, dims, points):
    assert points.ndim == 2
    homog_points = np.ones((points.shape[0], 4))
    homog_points[:, :3] = points
    projected_points = np.empty(
        (inverse_pose_matrices.shape[0], homog_points.shape[0], 3)
    )
    for i in range(len(inverse_pose_matrices)):
        projected_points[i] = (inverse_pose_matrices[i] @ homog_points.T)[:3, :].T
    distance = np.abs(projected_points) - (np.expand_dims(dims, 1) / 2)
    _outside = np.power(np.maximum(distance, np.zeros((1, 3))), 2)
    outside = np.sqrt(_outside[:, :, 0] + _outside[:, :, 1] + _outside[:, :, 2])
    inner_max_distance = np.empty(distance.shape[:2])
    for i in range(inner_max_distance.shape[0]):
        for j in range(inner_max_distance.shape[1]):
            inner_max_distance[i, j] = np.max(distance[i, j, :])
    inside = np.minimum(inner_max_distance, 0)
    return outside + inside


@nb.jit(nopython=True, cache=True)
def _cuboid_array_scene_sdf_1d(inverse_pose_matrices, dims, points):
    all_sdfs = _cuboid_array_all_sdfs_1d(inverse_pose_matrices, dims, points)
    return np.min(all_sdfs)


@nb.jit(nopython=True, cache=True)
def _cuboid_array_scene_sdf_2d(inverse_pose_matrices, dims, points):
    sdfs = _cuboid_array_all_sdfs_2d(inverse_pose_matrices, dims, points).T
    result = np.empty(len(sdfs))
    for i in range(len(result)):
        result[i] = np.min(sdfs[i, :])
    return result


@nb.jit(nopython=True, cache=True)
def _cuboid_corners(pose_matrix, dims):
    x_front, x_back = (
        dims[0] / 2,
        -dims[0] / 2,
    )
    y_front, y_back = (
        dims[1] / 2,
        -dims[1] / 2,
    )
    z_front, z_back = (
        dims[2] / 2,
        -dims[2] / 2,
    )
    corners = np.array(
        [
            [x_front, y_front, z_front],
            [x_back, y_front, z_front],
            [x_front, y_back, z_front],
            [x_front, y_front, z_back],
            [x_back, y_back, z_front],
            [x_back, y_front, z_back],
            [x_front, y_back, z_back],
            [x_back, y_back, z_back],
        ]
    )
    transform_in_place(corners, pose_matrix)
    return corners


@nb.jit(nopython=True, cache=True)
def _cuboid_surface_area(dims):
    return 2 * (dims[0] * dims[1] + dims[0] * dims[2] + dims[1] * dims[2])


@nb.jit(nopython=True, cache=True)
def _cylinder_surface_area(radius, height):
    return height * 2 * np.pi * radius + 2 * np.pi * radius**2


@nb.jit(nopython=True, cache=True)
def _cylinder_volume(radius, height):
    return height * np.pi * radius**2


@nb.jit(nopython=True, cache=True)
def _cylinder_sdf(inverse_pose_matrix, radius, height, point):
    assert point.ndim < 3
    if point.ndim == 1:
        homog_point = np.ones(4)
        homog_point[:3] = np.asarray(point)
        projected_point = (inverse_pose_matrix @ homog_point)[:3]
        surface_distance_xy = np.linalg.norm(projected_point[:2])
        z_distance = projected_point[2]

        # After having the z distance, we can reduce this problem to a
        # 2D box computation with size height and width 2 * radius
        half_extent_2d = np.array([radius, height / 2])
        point_2d = np.array([surface_distance_xy, z_distance])
        distance_2d = np.abs(point_2d) - half_extent_2d

        outside = np.linalg.norm(np.maximum(distance_2d, np.zeros(2)))
        inner_max_distance_2d = np.max(distance_2d)
        inside = np.minimum(inner_max_distance_2d, 0)
    else:
        homog_point = np.ones((point.shape[0], 4))
        homog_point[:, :3] = point
        projected_point = (inverse_pose_matrix @ homog_point.T)[:3, :].T
        xy2 = np.power(projected_point[:, :2], 2)
        surface_distance_xy = np.sqrt(xy2[:, 0] + xy2[:, 1])
        z_distance = projected_point[:, 2]

        half_extent_2d = np.array([radius, height / 2])
        point_2d = np.stack((surface_distance_xy, z_distance), axis=1)
        distance_2d = np.abs(point_2d) - half_extent_2d

        _outside = np.power(np.maximum(distance_2d, np.zeros(2)), 2)
        outside = np.sqrt(_outside[:, 0] + _outside[:, 1])
        inner_max_distance_2d = np_apply_along_axis(np.max, 1, distance_2d)
        inside = np.minimum(inner_max_distance_2d, 0)

    return outside + inside


@nb.jit(nopython=True, cache=True)
def _cylinder_array_all_sdfs_1d(inverse_pose_matrices, radii, heights, point):
    assert point.ndim == 1
    homog_point = np.ones(4)
    homog_point[:3] = point
    projected_point = np.empty((inverse_pose_matrices.shape[0], 3))
    for i in range(len(inverse_pose_matrices)):
        projected_point[i, :] = np.dot(inverse_pose_matrices[i], homog_point)[:3]

    _surface_distance_xy = np.power(projected_point[:, :2], 2)
    surface_distance_xy = np.sqrt(
        _surface_distance_xy[:, 0] + _surface_distance_xy[:, 1]
    )
    z_distance = projected_point[:, 2]

    # After having the z distance, we can reduce this problem to a
    # 2D box computation with size height and width 2 * radius
    half_extent_2d = np.stack((radii, heights / 2), axis=1)
    point_2d = np.stack((surface_distance_xy, z_distance), axis=1)
    distance_2d = np.abs(point_2d) - half_extent_2d

    distance_2d_max = np.maximum(distance_2d, np.zeros(2))
    _outside = np.power(distance_2d_max, 2)
    outside = np.sqrt(_outside[:, 0] + _outside[:, 1])
    inner_max_distance_2d = np.empty(distance_2d.shape[0])
    for i in range(len(inner_max_distance_2d)):
        inner_max_distance_2d[i] = np.max(distance_2d[i, :])
    inside = np.minimum(inner_max_distance_2d, 0)
    return outside + inside


@nb.jit(nopython=True, cache=True)
def _cylinder_array_all_sdfs_2d(inverse_pose_matrices, radii, heights, points):
    assert points.ndim == 2
    homog_points = np.ones((points.shape[0], 4))
    homog_points[:, :3] = points
    projected_points = np.empty(
        (inverse_pose_matrices.shape[0], homog_points.shape[0], 3)
    )
    for i in range(len(inverse_pose_matrices)):
        projected_points[i] = (inverse_pose_matrices[i] @ homog_points.T)[:3, :].T

    xy2 = np.power(projected_points[:, :, :2], 2)
    surface_distance_xy = np.sqrt(xy2[:, :, 0] + xy2[:, :, 1])
    z_distance = projected_points[:, :, 2]

    half_extent_2d = np.expand_dims(np.stack((radii, heights / 2), axis=1), 1)
    point_2d = np.stack((surface_distance_xy, z_distance), axis=2)
    distance_2d = np.abs(point_2d) - half_extent_2d

    _outside = np.power(np.maximum(distance_2d, np.zeros(2)), 2)
    outside = np.sqrt(_outside[:, :, 0] + _outside[:, :, 1])
    inner_max_distance_2d = np.empty(distance_2d.shape[:2])
    for i in range(inner_max_distance_2d.shape[0]):
        for j in range(inner_max_distance_2d.shape[1]):
            inner_max_distance_2d[i, j] = np.max(distance_2d[i, j, :])
    inside = np.minimum(inner_max_distance_2d, 0)

    return outside + inside


@nb.jit(nopython=True, cache=True)
def _cylinder_array_scene_sdf_1d(inverse_pose_matrices, radii, heights, point):
    all_sdfs = _cylinder_array_all_sdfs_1d(inverse_pose_matrices, radii, heights, point)
    return np.min(all_sdfs)


@nb.jit(nopython=True, cache=True)
def _cylinder_array_scene_sdf_2d(inverse_pose_matrices, radii, heights, points):
    sdfs = _cylinder_array_all_sdfs_2d(inverse_pose_matrices, radii, heights, points).T
    result = np.empty(len(sdfs))
    for i in range(len(result)):
        result[i] = np.min(sdfs[i, :])
    return result


@nb.jit(nopython=True, cache=True)
def _cylinder_sample_surface(pose_matrix, radius, height, num_points, noise):
    assert (
        noise >= 0
    ), "Noise parameter should be a radius of the uniform distribution added to the random points"
    angles = np.random.uniform(-np.pi, np.pi, num_points)
    circle_points = np.stack((np.cos(angles), np.sin(angles)), axis=1)
    surface_area = _cylinder_surface_area(radius, height)
    probs = np.array(
        [
            np.pi * radius**2 / surface_area,
            height * 2 * np.pi * radius / surface_area,
            np.pi * radius**2 / surface_area,
        ]
    )
    which_surface = np.searchsorted(
        np.cumsum(probs), np.random.random(num_points), side="right"
    )
    circle_points[which_surface == 0] *= np.random.uniform(
        0, radius, size=(np.count_nonzero(which_surface == 0), 1)
    )
    circle_points[which_surface == 1] *= radius
    circle_points[which_surface == 2] *= np.random.uniform(
        0, radius, size=(np.count_nonzero(which_surface == 2), 1)
    )
    z = np.ones((num_points, 1))
    z[which_surface == 0] = -height / 2
    z[which_surface == 1] = np.random.uniform(
        -height / 2,
        height / 2,
        size=(np.count_nonzero(which_surface == 1), 1),
    )
    z[which_surface == 2] = height / 2
    surface_points = np.concatenate((circle_points, z), axis=1)
    transform_in_place(surface_points, pose_matrix)
    noise = 2 * noise * np.random.random_sample(surface_points.shape) - noise
    return surface_points + noise


@nb.jit(nopython=True, cache=True)
def _cylinder_sample_volume(pose_matrix, radius, height, num_points, noise):
    assert (
        noise >= 0
    ), "Noise parameter should be a radius of the uniform distribution added to the random points"
    angles = np.random.uniform(-np.pi, np.pi, num_points)
    disc_points = np.stack((np.cos(angles), np.sin(angles)), axis=1)
    radii = np.random.uniform(0, radius, size=(num_points, 1))
    disc_points *= radii
    volume_points = np.concatenate(
        (
            disc_points,
            np.random.uniform(-height / 2, height / 2, size=(num_points, 1)),
        ),
        axis=1,
    )
    transform_in_place(volume_points, pose_matrix)
    noise = 2 * noise * np.random.random_sample(volume_points.shape) - noise
    return volume_points + noise


def test():
    cylinders = [
        Cylinder.random(
            center_range=np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]),
            radius_range=np.array([0.2, 1.0]),
            height_range=np.array([0.2, 1.0]),
            random_orientation=True,
        )
        for _ in range(10)
    ]
    array = CylinderArray(cylinders)
    point = np.array([0.3, 0.2, -5])
    points = np.random.uniform(size=(2, 3))
    print(np.allclose(array.all_sdfs(point), [c.sdf(point) for c in cylinders]))
    print(np.allclose(array.all_sdfs(points), [c.sdf(points) for c in cylinders]))
    print(array.scene_sdf(point))
    print(array.scene_sdf(points))


if __name__ == "__main__":
    test()
