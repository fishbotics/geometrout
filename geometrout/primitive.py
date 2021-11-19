from pyquaternion import Quaternion
import numpy as np

from geometrout.transform import SE3, SO3
import geometrout.pointcloud as pc


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
        self._pose = SE3(xyz=center, so3=SO3(quat=quaternion))
        self._dims = np.asarray(dims)

    @classmethod
    def random(cls, center_range=None, dimension_range=None, quaternion=False):
        """
        Creates a random cuboid within the given ranges
        :param center_range: If given, represents the uniform range from which to draw a center.
            Should be np.array with dimension 2x3. First row is lower limit, second row is upper limit
            If not given, center defaults to [0, 0, 0]
        :param dimension_range: If given, represents the uniform range from which to draw a center.
            Should be np.array with dimension 2x3. First row is lower limit, second row is upper limit
            If not given, center defaults to [1, 1, 1]
        :param quaternion: If True, will give a random orientation to cuboid.
            If not given, will be set as the identity
        :return: Cuboid object drawn from specified uniform distribution
        """
        if center_range is not None:
            center_range = np.asarray(center_range)
            assert center_range.shape == (
                2,
                3,
            ), "Center range should be passed in as numpy array with 2x3 dimension where first row is the low end of each dimension's range and second row is the high end"

            center = (center_range[1, :] - center_range[0, :]) * np.random.rand(
                3
            ) + center_range[0, :]
        else:
            center = np.array([0.0, 0.0, 0.0])
        if dimension_range is not None:
            dimension_range = np.asarray(dimension_range)
            assert dimension_range.shape == (
                2,
                3,
            ), "Dimension range should be passed in as numpy array with 2x3 dimension where first row is the low end of each dimension's range and second row is the high end"
            dims = (dimension_range[1, :] - dimension_range[0, :]) * np.random.rand(
                3
            ) + dimension_range[0, :]
        else:
            dims = np.array([1.0, 1.0, 1.0])
        if quaternion:
            quaternion = Quaternion.random()
        else:
            quaternion = Quaternion([1.0, 0.0, 0.0, 0.0])

        return cls(center, dims, quaternion)

    def is_zero_volume(self):
        return np.isclose(self._dims, 0).any()

    def sample(self, num_points, noise=0.0):
        assert (
            noise >= 0
        ), "Noise parameter should be a radius of the uniform distribution added to the random points"
        random_points = np.random.uniform(-1.0, 1.0, (num_points, 3))
        random_points = random_points * self._dims / 2
        probs = np.array(
            [
                self._dims[1] * self._dims[2],
                self._dims[1] * self._dims[2],
                self._dims[0] * self._dims[2],
                self._dims[0] * self._dims[2],
                self._dims[0] * self._dims[1],
                self._dims[0] * self._dims[1],
            ]
        )
        probs /= np.sum(probs)
        sides = np.random.choice(6, num_points, p=probs)
        random_points[sides == 0, 0] = self._dims[0] / 2
        random_points[sides == 1, 0] = -self._dims[0] / 2
        random_points[sides == 2, 1] = self._dims[1] / 2
        random_points[sides == 3, 1] = -self._dims[1] / 2
        random_points[sides == 4, 2] = self._dims[2] / 2
        random_points[sides == 5, 2] = -self._dims[2] / 2
        transform = self.pose.matrix
        pc.transform(random_points, transform, in_place=True)
        noise = 2 * noise * np.random.random_sample(random_points.shape) - noise
        return random_points + noise

    @property
    def pose(self):
        """
        :return: The pose of the cuboid as an SE3 object
        """
        return self._pose

    @property
    def center(self):
        """
        :return: The center of the object as a list
        """
        return self._pose.xyz

    @center.setter
    def center(self, val):
        """
        Set the center of the cuboid

        :param val: The new center of the cuboid
        """
        self._pose.xyz = val

    @property
    def dims(self):
        """
        :return: The dimensions of the cuboid as a list
        """
        return self._dims.tolist()

    @dims.setter
    def dims(self, value):
        """
        Set the dimensions of the cuboid

        :param value: The new desired dimensions of the cuboid
        """
        self._dims = np.asarray(value)

    @property
    def half_extents(self):
        """
        :return: The half-dimensions of the cuboid. This is necessary for some interfaces.
        """
        return (self._dims / 2).tolist()

    @property
    def surface_area(self):
        return 2 * (
            self._dims[0] * self._dims[1]
            + self._dims[0] * self._dims[2]
            + self._dims[1] * self._dims[2]
        )


class Sphere:
    def __init__(self, center, radius):
        """
        Constructs an internal sphere representation

        :param center: The center of the sphere as a list of numpy array
        :param radius: The radius of the sphere as a number
        """
        self._center = np.asarray(center)
        self._radius = radius

    @classmethod
    def random(cls, center_range=None, radius_range=None):
        if center_range is not None:
            center_range = np.asarray(center_range)
            assert center_range.shape == (
                2,
                3,
            ), "Center range should be passed in as numpy array with 2x3 dimension where first row is the low end of each dimension's range and second row is the high end"

            center = (center_range[1, :] - center_range[0, :]) * np.random.rand(
                3
            ) + center_range[0, :]
        else:
            center = np.array([0.0, 0.0, 0.0])
        if radius_range is not None:
            mn, mx = radius_range
            radius = (mx - mn) * np.random.rand() + mn
        else:
            radius = np.random.rand()
        return cls(center, radius)

    @property
    def center(self):
        """
        :return: The center of the sphere as a list
        """
        return self._center.tolist()

    @property
    def radius(self):
        """
        :return: The radius of the sphere
        """
        return self._radius

    @property
    def surface_area(self):
        return 4 * np.pi * self.radius ** 3

    def sample(self, num_points, noise=0.0):
        assert (
            noise >= 0
        ), "Noise parameter should be a radius of the uniform distribution added to the random points"
        unnormalized_points = np.random.uniform(-1.0, 1.0, (num_points, 3))
        normalized = (
            unnormalized_points / np.linalg.norm(unnormalized_points, axis=1)[:, None]
        )
        shift = np.tile(self.center, (num_points, 1))
        random_points = (normalized * self.radius + shift).astype(np.float32)
        noise = 2 * noise * np.random.random_sample(random_points.shape) - noise
        return random_points + noise
