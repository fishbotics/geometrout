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
        self._pose = SE3(xyz=center, so3=SO3(quaternion=quaternion))
        self._dims = np.asarray(dims)

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
                f"Center: {self.center}",
                f"Dimensions: {self.dims}",
                f"Orientation (wxyz): {self.pose.so3.wxyz}",
            ]
        )

    def __repr__(self):
        return "\n".join(
            [
                "Cuboid(",
                f"    center={self.center},",
                f"    dims={self.dims},",
                f"    quaternion={self.pose.so3.wxyz},",
                ")",
            ]
        )

    @classmethod
    def unit(cls):
        return cls(
            center=[0.0, 0.0, 0.0],
            dims=[1.0, 1.0, 1.0],
            quaternion=[1.0, 0.0, 0.0, 0.0],
        )

    @classmethod
    def random(
        cls,
        center_range=[[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]],
        dimension_range=[[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
        quaternion=True,
    ):
        """
        Creates a random cuboid within the given ranges
        :param center_range: If given, represents the uniform range from which to draw a center.
            Should be np.array with dimension 2x3. First row is lower limit, second row is upper limit
            If passed as None, center defaults to [0, 0, 0]
        :param dimension_range: If given, represents the uniform range from which to draw a center.
            Should be np.array with dimension 2x3. First row is lower limit, second row is upper limit
            If passed as None, dimensions defaults to [1, 1, 1]
        :param quaternion: If True, will give a random orientation to cuboid.
            If False, will be set as the identity
            Default is True
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

    def sample_surface(self, num_points, noise=0.0):
        """
        Samples random points on the surface of the cube. Probabilities are
        weighed based on area of each side.

        :param num_points: The number of points to sample on the surface
        :param noise: The range of uniform noise to apply to samples

        :return: A random pointcloud sampled from the surface of the cuboid
        """
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

    def sample_volume(self, num_points):
        """
        Get a random pointcloud sampled inside the cuboid (including the surface)

        :param num_points: The number of points to sample
        :return: A set of points inside the cube
        """
        random_points = np.random.uniform(-1.0, 1.0, (num_points, 3))
        random_points = random_points * self._dims / 2
        transform = self.pose.matrix
        pc.transform(random_points, transform, in_place=True)
        return random_points

    def sdf(self, point):
        """
        :param point: Point in 3D for which we want the sdf
        :return: The sdf value of that point
        """
        homog_point = np.ones(4)
        homog_point[:3] = np.asarray(point)
        projected_point = (self.pose.inverse.matrix @ homog_point)[:3]
        distance = np.abs(projected_point) - (self._dims / 2)
        outside = np.linalg.norm(np.maximum(distance, np.zeros(3)))
        inner_max_distance = np.max(distance)
        inside = np.minimum(inner_max_distance, 0)
        return outside + inside

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
    def corners(self):
        x_front, x_back = (
            self._dims[0] / 2,
            -self._dims[0] / 2,
        )
        y_front, y_back = (
            self._dims[1] / 2,
            -self._dims[1] / 2,
        )
        z_front, z_back = (
            self._dims[2] / 2,
            -self._dims[2] / 2,
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
        pc.transform(corners, self.pose.matrix, in_place=True)
        return corners

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

    def __str__(self):
        return "\n".join(
            [
                "         ▓▓▓▓▓▓▓▓▓▓▓▓               ",
                "      ░░████░░░░░░░░░░░░████        ",
                "    ░░██▒▒░░░░░░░░░░░░░░░░░░██      ",
                "  ░░██▒▒▒▒░░░░░░░░░░░░░░░░░░░░██    ",
                "  ██▒▒▒▒░░░░░░░░░░░░░░    ░░░░░░██  ",
                "  ██▒▒▒▒░░░░░░░░░░░░        ░░░░██  ",
                "██▓▓▒▒▒▒░░░░░░░░░░░░        ░░░░░░██",
                "██▓▓▒▒▒▒░░░░░░░░░░░░░░    ░░░░░░░░██",
                "██▓▓▒▒▒▒░░░░░░░░░░░░░░░░░░░░░░░░░░██",
                "██▓▓▒▒▒▒▒▒░░░░░░░░░░░░░░░░░░░░░░░░██",
                "██▓▓▓▓▒▒▒▒░░░░░░░░░░░░░░░░░░░░░░░░██",
                "██▓▓▓▓▒▒▒▒▒▒░░░░░░░░░░░░░░░░░░░░░░██",
                "  ▓▓▓▓▓▓▒▒▒▒▒▒░░░░░░░░░░░░░░░░░░██  ",
                "  ██▓▓▓▓▓▓▒▒▒▒▒▒▒▒░░░░░░░░░░▒▒▒▒██  ",
                "  ░░▓▓▓▓▓▓▓▓▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒██░░  ",
                "    ░░▓▓▓▓▓▓▓▓▓▓▒▒▒▒▒▒▒▒▒▒▒▒██░░    ",
                "      ░░██▓▓▓▓▓▓▓▓▓▓▓▓▓▓████░░      ",
                "        ░░░░████████████░░          ",
                "            ░░░░░░░░░░░░            ",
                f"Center: {self.center}",
                f"Radius: {self.radius}",
            ]
        )

    def __repr__(self):
        return f"Sphere(center={self.center}, radius={self.radius})"

    @classmethod
    def unit(cls):
        return cls(np.array([0.0, 0.0, 0.0]), 1.0)

    @classmethod
    def random(
        cls, center_range=[[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]], radius_range=[0.0, 1.0]
    ):
        """
        Creates a random sphere.
        :param center_range: 2x3 numpy array or list with form
          [[x_low, y_low, z_low], [x_high, y_hight, x_max]]. Pass in None to
          default at origin
        :param radius_range: List [r_low, r_high]. Pass in None for
          default radius of None
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
        if radius_range is not None:
            mn, mx = radius_range
            radius = (mx - mn) * np.random.rand() + mn
        else:
            radius = 1.0
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
        return 4 * np.pi * self.radius**3

    def is_zero_volume(self):
        return np.isclose(self.radius, 0)

    def sdf(self, point):
        """
        :param point: Point in 3D for which we want the sdf
        :return: The sdf value of that point
        """
        return np.linalg.norm(np.asarray(point) - self._center) - self.radius

    def sample_surface(self, num_points, noise=0.0):
        """
        Samples random points on the surface of the sphere. Probabilities are
        weighed based on area of each side.

        :param num_points: The number of points to sample on the surface
        :param noise: The range of uniform noise to apply to samples

        :return: A random pointcloud sampled from the surface of the cuboid
        """
        assert (
            noise >= 0
        ), "Noise parameter should be a radius of the uniform distribution added to the random points"
        unnormalized_points = np.random.uniform(-1.0, 1.0, (num_points, 3))
        normalized = (
            unnormalized_points / np.linalg.norm(unnormalized_points, axis=1)[:, None]
        )
        shift = np.tile(self._center, (num_points, 1))
        random_points = normalized * self.radius + shift
        noise = 2 * noise * np.random.random_sample(random_points.shape) - noise
        return random_points + noise

    def sample_volume(self, num_points):
        """
        Get a random pointcloud sampled inside the sphere (including the surface)

        :param num_points: The number of points to sample
        :return: A set of points inside the sphere
        """
        # First produce points on the surface of the unit sphere
        unnormalized_points = np.random.uniform(-1.0, 1.0, (num_points, 3))
        normalized = (
            unnormalized_points / np.linalg.norm(unnormalized_points, axis=1)[:, None]
        )

        # Now multiply them by random radii in the range [0, self.radius]
        radii = np.random.uniform(0, self.radius, (num_points, 1))

        shift = np.tile(self._center, (num_points, 1))
        return normalized * radii + shift


class Cylinder:
    def __init__(self, center, radius, height, quaternion):
        self._pose = SE3(xyz=center, so3=SO3(quaternion=quaternion))
        self.radius = radius
        self.height = height

    def __repr__(self):
        return "\n".join(
            [
                "Cylinder(",
                f"    center={self.center},",
                f"    radius={self.radius},",
                f"    height={self.height},",
                f"    quaternion={self.pose.so3.wxyz},",
                ")",
            ]
        )

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

    @classmethod
    def random(
        cls,
        center_range=[[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]],
        radius_range=[0.0, 1.0],
        height_range=[0.0, 1.0],
        quaternion=True,
    ):
        """
        Creates a random Cylinder.
        :param center_range: 2x3 numpy array or list with form
          [[x_low, y_low, z_low], [x_high, y_hight, x_max]]. Pass in None to
          default at origin
        :param radius_range: List [r_low, r_high]. Pass in None for
          default radius of None
        :param height_range: List [h_low, h_high]. Pass in None for
          default radius of None
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
        if radius_range is not None:
            mn, mx = radius_range
            radius = (mx - mn) * np.random.rand() + mn
        else:
            radius = 1.0
        if height_range is not None:
            mn, mx = height_range
            height = (mx - mn) * np.random.rand() + mn
        else:
            height = 1.0
        if quaternion:
            quaternion = Quaternion.random()
        else:
            quaternion = Quaternion([1.0, 0.0, 0.0, 0.0])
        return cls(center, radius, height, quaternion)

    @property
    def pose(self):
        """
        :return: The pose of the cuboid as an SE3 object
        """
        return self._pose

    @property
    def surface_area(self):
        return self.height * 2 * np.pi * self.radius + 2 * np.pi * self.radius**2

    def is_zero_volume(self):
        return np.isclose(self.radius, 0) or np.isclose(self.height, 0)

    def sdf(self, point):
        """
        :param point: Point in 3D for which we want the sdf
        :return: The sdf value of that point
        """
        homog_point = np.ones(4)
        homog_point[:3] = np.asarray(point)
        projected_point = (self.pose.inverse.matrix @ homog_point)[:3]
        surface_distance_xy = np.linalg.norm(projected_point[:2])
        z_distance = projected_point[2]

        # After having the z distance, we can reduce this problem to a
        # 2D box computation with size height and width 2 * radius
        half_extent_2d = np.array([self.radius, self.height / 2])
        point_2d = np.array([surface_distance_xy, z_distance])
        distance_2d = np.abs(point_2d) - half_extent_2d

        outside = np.linalg.norm(np.maximum(distance_2d, np.zeros(2)))
        inner_max_distance_2d = np.max(distance_2d)
        inside = np.minimum(inner_max_distance_2d, 0)
        return outside + inside

    def sample_surface(self, num_points, noise=0.0):
        """
        Samples random points on the surface of the cylinder. Probabilities are
        weighed based on area of each side.

        :param num_points: The number of points to sample on the surface
        :param noise: The range of uniform noise to apply to samples

        :return: A random pointcloud sampled from the surface of the cuboid
        """
        assert (
            noise >= 0
        ), "Noise parameter should be a radius of the uniform distribution added to the random points"
        angles = np.random.uniform(-np.pi, np.pi, num_points)
        circle_points = np.stack((np.cos(angles), np.sin(angles)), axis=1)
        surface_area = self.surface_area
        which_surface = np.random.choice(
            3,
            size=num_points,
            p=np.array(
                [
                    np.pi * self.radius**2 / surface_area,
                    self.height * 2 * np.pi * self.radius / surface_area,
                    np.pi * self.radius**2 / surface_area,
                ]
            ),
        )
        circle_points[which_surface == 0] *= np.random.uniform(
            0, self.radius, size=(np.count_nonzero(which_surface == 0), 1)
        )
        circle_points[which_surface == 1] *= self.radius
        circle_points[which_surface == 2] *= np.random.uniform(
            0, self.radius, size=(np.count_nonzero(which_surface == 2), 1)
        )
        z = np.ones((num_points, 1))
        z[which_surface == 0] = -self.height / 2
        z[which_surface == 1] = np.random.uniform(
            -self.height / 2,
            self.height / 2,
            size=(np.count_nonzero(which_surface == 1), 1),
        )
        z[which_surface == 2] = self.height / 2
        surface_points = np.concatenate((circle_points, z), axis=1)
        transform = self.pose.matrix
        pc.transform(surface_points, transform, in_place=True)
        noise = 2 * noise * np.random.random_sample(surface_points.shape) - noise
        return surface_points + noise

    def sample_volume(self, num_points, noise=0.0):
        """
        Get a random pointcloud sampled inside the cylinder (including the surface)

        :param num_points: The number of points to sample
        :return: A set of points inside the sphere
        """
        assert (
            noise >= 0
        ), "Noise parameter should be a radius of the uniform distribution added to the random points"
        angles = np.random.uniform(-np.pi, np.pi, num_points)
        disc_points = np.stack((np.cos(angles), np.sin(angles)), axis=1)
        radii = np.random.uniform(0, self.radius, size=(num_points, 1))
        disc_points *= radii
        volume_points = np.concatenate(
            (
                disc_points,
                np.random.uniform(
                    -self.height / 2, self.height / 2, size=(num_points, 1)
                ),
            ),
            axis=1,
        )
        pc.transform(volume_points, self.pose.matrix, in_place=True)
        noise = 2 * noise * np.random.random_sample(volume_points.shape) - noise
        return volume_points + noise
