import numpy as np
import numba as nb

from geometrout.transform import SE3, SO3
import geometrout.pointcloud as pc


@nb.jit
def _rand_choice(arr, prob):
    """
    :param arr: A 1D numpy array of values to sample from.
    :param prob: A 1D numpy array of probabilities for the given samples.
    :return: A random sample from the given array with a given probability.
    """
    return arr[np.searchsorted(np.cumsum(prob), np.random.random(), side="right")]


@nb.jit
def _transform(point_cloud, transformation_matrix):
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


@nb.experimental.jitclass([("dims", nb.float64[:])])
class Cuboid:
    pose: SE3

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

    @staticmethod
    def unit(cls):
        return Cuboid(
            center=np.array([0.0, 0.0, 0.0]),
            dims=np.array([1.0, 1.0, 1.0]),
            quaternion=np.array([1.0, 0.0, 0.0, 0.0]),
        )

    @staticmethod
    def random(
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
            pose = SO3.random()
        else:
            pose = SO3.unit()
        return Cuboid(center, dims, pose.q)

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
        assert (
            noise >= 0
        ), "Noise parameter should be a radius of the uniform distribution added to the random points"
        random_points = np.random.uniform(-1.0, 1.0, (num_points, 3))
        random_points = random_points * self.dims / 2
        probs = np.array(
            [
                self.dims[1] * self.dims[2],
                self.dims[1] * self.dims[2],
                self.dims[0] * self.dims[2],
                self.dims[0] * self.dims[2],
                self.dims[0] * self.dims[1],
                self.dims[0] * self.dims[1],
            ]
        )
        probs /= np.sum(probs)
        sides = np.searchsorted(
            np.cumsum(probs), np.random.random(num_points), side="right"
        )
        random_points[sides == 0, 0] = self.dims[0] / 2
        random_points[sides == 1, 0] = -self.dims[0] / 2
        random_points[sides == 2, 1] = self.dims[1] / 2
        random_points[sides == 3, 1] = -self.dims[1] / 2
        random_points[sides == 4, 2] = self.dims[2] / 2
        random_points[sides == 5, 2] = -self.dims[2] / 2
        transform = self.pose.matrix
        _transform(random_points, transform)
        noise = 2 * noise * np.random.random_sample(random_points.shape) - noise
        return random_points + noise

    def sample_volume(self, num_points):
        """
        Get a random pointcloud sampled inside the cuboid (including the surface)

        :param num_points: The number of points to sample
        :return: A set of points inside the cube
        """
        random_points = np.random.uniform(-1.0, 1.0, (num_points, 3))
        random_points = random_points * self.dims / 2
        transform = self.pose.matrix
        _transform(random_points, transform)
        return random_points

    def sdf(self, point):
        """
        :param point: Point in 3D for which we want the sdf
        :return: The sdf value of that point
        """
        homog_point = np.ones(4)
        homog_point[:3] = point
        projected_point = (self.pose.inverse.matrix @ homog_point)[:3]
        distance = np.abs(projected_point) - (self.dims / 2)
        outside = np.linalg.norm(np.maximum(distance, np.zeros(3)))
        inner_max_distance = np.max(distance)
        inside = np.minimum(inner_max_distance, 0)
        return outside + inside

    @property
    def center(self):
        """
        :return: The center of the object as a list
        """
        return self.pose.pos

    @property
    def half_extents(self):
        """
        :return: The half-dimensions of the cuboid. This is necessary for some interfaces.
        """
        return self.dims / 2

    @property
    def corners(self):
        x_front, x_back = (
            self.dims[0] / 2,
            -self.dims[0] / 2,
        )
        y_front, y_back = (
            self.dims[1] / 2,
            -self.dims[1] / 2,
        )
        z_front, z_back = (
            self.dims[2] / 2,
            -self.dims[2] / 2,
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
        _transform(corners, self.pose.matrix)
        return corners

    @property
    def surface_area(self):
        return 2 * (
            self.dims[0] * self.dims[1]
            + self.dims[0] * self.dims[2]
            + self.dims[1] * self.dims[2]
        )


@nb.experimental.jitclass([("center", nb.float64[:]), ("radius", nb.float64)])
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

    @staticmethod
    def unit():
        return Sphere(np.array([0.0, 0.0, 0.0]), 1.0)

    @staticmethod
    def random(
        center_range=np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
        radius_range=np.array([1.0, 1.0]),
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
        return Sphere(center, radius)

    @property
    def surface_area(self):
        return 4 * np.pi * self.radius**2

    @property
    def volume(self):
        return 4 / 3 * np.pi * self.radius**3

    def is_zero_volume(self, atol=1e-7):
        return self.radius < atol

    def sdf(self, point):
        """
        :param point: Point in 3D for which we want the sdf
        :return: The sdf value of that point
        """
        return np.linalg.norm(point - self.center) - self.radius

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
        points = np.random.uniform(-1.0, 1.0, (num_points, 3))
        for i in range(points.shape[0]):
            nrm = np.linalg.norm(points[i, :])
            points[i, :] /= nrm
        points = self.radius * points + self.center
        if noise > 0.0:
            noise = np.random.uniform(-noise, noise, points.shape)
            return points + noise
        return points

    def sample_volume(self, num_points):
        """
        Get a random pointcloud sampled inside the sphere (including the surface)

        :param num_points: The number of points to sample
        :return: A set of points inside the sphere
        """
        # First produce points on the surface of the unit sphere
        points = np.random.uniform(-1.0, 1.0, (num_points, 3))
        for i in range(points.shape[0]):
            nrm = np.linalg.norm(points[i, :])
            points[i, :] /= nrm

        # Now multiply them by random radii in the range [0, self.radius]
        radii = np.random.uniform(0, self.radius, (num_points, 1))
        return points * radii + self.center


@nb.experimental.jitclass([("radius", nb.float64), ("height", nb.float64)])
class Cylinder:
    pose: SE3

    def __init__(self, center, radius, height, quaternion):
        assert radius >= 0 and height >= 0
        self.pose = SE3(center.astype(np.double), quaternion.astype(np.double))
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
        return self.pose.pos

    @staticmethod
    def random(
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
            pose = SO3.random()
        else:
            pose = SO3.unit()
        return Cylinder(center, radius, height, pose.q)

    @property
    def surface_area(self):
        return self.height * 2 * np.pi * self.radius + 2 * np.pi * self.radius**2

    @property
    def volume(self):
        return self.height * np.pi * self.radius**2

    def is_zero_volume(self, atol=1e-7):
        return self.radius < atol or self.height < atol

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
        probs = np.array(
            [
                np.pi * self.radius**2 / surface_area,
                self.height * 2 * np.pi * self.radius / surface_area,
                np.pi * self.radius**2 / surface_area,
            ]
        )
        which_surface = np.searchsorted(
            np.cumsum(probs), np.random.random(num_points), side="right"
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
        _transform(surface_points, transform)
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
        _transform(volume_points, self.pose.matrix)
        noise = 2 * noise * np.random.random_sample(volume_points.shape) - noise
        return volume_points + noise
