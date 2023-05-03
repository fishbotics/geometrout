import numba as nb
import numpy as np


@nb.jit(nopython=True, cache=True)
def _quaternion_trace_method(matrix, rtol=1e-7, atol=1e-7):
    """
    This code uses a modification of the algorithm described in:
    https://d3cw3dd2w32x2b.cloudfront.net/wp-content/uploads/2015/01/matrix-to-quat.pdf
    which is itself based on the method described here:
    http://www.euclideanspace.com/maths/geometry/rotations/conversions/matrixToQuaternion/
    Altered to work with the column vector convention instead of row vectors
    """
    assert matrix.shape == (3, 3)
    if not np.allclose(
        np.dot(matrix, matrix.conj().transpose()),
        np.eye(3),
        rtol=rtol,
        atol=atol,
        equal_nan=False,
    ):
        raise ValueError(
            "Matrix must be orthogonal, i.e. its transpose should be its inverse"
        )
    # Re-implemented `np.isclose` for Numba
    if np.abs(np.linalg.det(matrix) - 1.0) > atol + rtol:
        raise ValueError(
            "Matrix must be special orthogonal i.e. its determinant must be +1.0"
        )
    m = (
        matrix.conj().transpose()
    )  # This method assumes row-vector and postmultiplication of that vector
    if m[2, 2] < 0:
        if m[0, 0] > m[1, 1]:
            t = 1 + m[0, 0] - m[1, 1] - m[2, 2]
            q = [m[1, 2] - m[2, 1], t, m[0, 1] + m[1, 0], m[2, 0] + m[0, 2]]
        else:
            t = 1 - m[0, 0] + m[1, 1] - m[2, 2]
            q = [m[2, 0] - m[0, 2], m[0, 1] + m[1, 0], t, m[1, 2] + m[2, 1]]
    else:
        if m[0, 0] < -m[1, 1]:
            t = 1 - m[0, 0] - m[1, 1] + m[2, 2]
            q = [m[0, 1] - m[1, 0], m[2, 0] + m[0, 2], m[1, 2] + m[2, 1], t]
        else:
            t = 1 + m[0, 0] + m[1, 1] + m[2, 2]
            q = [t, m[1, 2] - m[2, 1], m[2, 0] - m[0, 2], m[0, 1] - m[1, 0]]

    q = np.array(q).astype("float64")
    q *= 0.5 / np.sqrt(t)
    return q


@nb.jit(nopython=True, cache=True)
def _random_rotation():
    r1, r2, r3 = np.random.random(3)
    return np.array(
        [
            np.sqrt(1.0 - r1) * (np.sin(2 * np.pi * r2)),
            np.sqrt(1.0 - r1) * (np.cos(2 * np.pi * r2)),
            np.sqrt(r1) * (np.sin(2 * np.pi * r3)),
            np.sqrt(r1) * (np.cos(2 * np.pi * r3)),
        ]
    )


@nb.jit(nopython=True, cache=True)
def _unit_axes_to_quaternion(x, y, z, rtol, atol):
    assert np.abs(np.linalg.norm(x) - 1) <= atol + rtol
    assert np.abs(np.linalg.norm(y) - 1) <= atol + rtol
    assert np.abs(np.linalg.norm(z) - 1) <= atol + rtol
    assert np.dot(x, y) < atol
    assert np.dot(x, z) < atol
    assert np.dot(z, y) < atol

    m = np.eye(3)
    m[:3, 0] = x
    m[:3, 1] = y
    m[:3, 2] = z
    return _quaternion_trace_method(m, rtol, atol)


@nb.jit(nopython=True, cache=True)
def _normalize(v):
    return v / np.linalg.norm(v)


@nb.jit(nopython=True, cache=True)
def _quaternion_from_rpy(r, p, y):
    c3, c2, c1 = np.cos(np.array([r, p, y]))
    s3, s2, s1 = np.sin(np.array([r, p, y]))

    matrix = np.array(
        [
            [c1 * c2, (c1 * s2 * s3) - (c3 * s1), (s1 * s3) + (c1 * c3 * s2)],
            [c2 * s1, (c1 * c3) + (s1 * s2 * s3), (c3 * s1 * s2) - (c1 * s3)],
            [-s2, c2 * s3, c2 * c3],
        ],
        dtype=np.float64,
    )
    return _quaternion_trace_method(matrix)


@nb.jit(nopython=True, cache=True)
def _quaternion_from_axis_angle(axis, angle):
    mag = np.linalg.norm(axis)
    if mag == 0.0:
        raise ZeroDivisionError("Provided rotation axis has no length")
    # Ensure axis is in unit vector form
    if np.abs(1.0 - mag) > 1e-12:
        axis = axis / mag
    theta = angle / 2.0
    r = np.cos(theta)
    i = axis * np.sin(theta)
    return np.array([r, i[0], i[1], i[2]])


@nb.jit(nopython=True, cache=True)
def _rotation_multiply(q1, q2):
    w0, x0, y0, z0 = q1
    w1, x1, y1, z1 = q2
    return np.array(
        [
            w0 * w1 + -x0 * x1 + -y0 * y1 + -z0 * z1,
            x0 * w1 + w0 * x1 + -z0 * y1 + y0 * z1,
            y0 * w1 + z0 * x1 + w0 * y1 + -x0 * z1,
            z0 * w1 + -y0 * x1 + x0 * y1 + w0 * z1,
        ]
    )


@nb.jit(nopython=True, cache=True)
def _quaternion_inverse(q):
    qinv = np.copy(q)
    qinv[1:] *= -1
    return qinv


@nb.jit(nopython=True, cache=True)
def _quaternion_to_radians(q):
    theta = 2.0 * np.arctan2(np.linalg.norm(q[1:4]), q[0])
    result = ((theta + np.pi) % (2 * np.pi)) - np.pi
    if np.abs(result + np.pi) < 1e-12:
        return np.pi
    return result


@nb.jit(nopython=True, cache=True)
def _quaternion_to_matrix(q):
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y**2 + z**2), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x**2 + z**2), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x**2 + y**2)],
        ],
    )


@nb.jit(nopython=True, cache=True)
def _quaternion_to_rpy(q):
    matrix = _quaternion_to_matrix(q)
    yaw = np.arctan2(matrix[1, 0], matrix[0, 0])
    pitch = np.arctan2(-matrix[2, 0], np.sqrt(matrix[2, 1] ** 2 + matrix[2, 2] ** 2))
    roll = np.arctan2(matrix[2, 1], matrix[2, 2])
    return roll, pitch, yaw


class SO3:
    """
    A generic class defining a 3D orientation. Mostly a wrapper around quaternions
    """

    def __init__(self, quaternion: np.ndarray):
        """
        :param quaternion: np.ndarray
        """
        assert quaternion.shape == (4,)
        self.q = _normalize(quaternion)

    @staticmethod
    def from_matrix(matrix, rtol=1e-7, atol=1e-7):
        q = _quaternion_trace_method(matrix, rtol, atol)
        return SO3(q)

    def __repr__(self):
        return f"SO3(quaternion={self.q})"

    @staticmethod
    def unit():
        return SO3(np.array([1.0, 0.0, 0.0, 0.0]))

    @staticmethod
    def random():
        return SO3(_random_rotation())

    @staticmethod
    def from_rpy(r, p, y):
        """
        Convert roll-pitch-yaw coordinates to a 3x3 homogenous rotation matrix.

        The roll-pitch-yaw axes in a typical URDF are defined as a
        rotation of ``r`` radians around the x-axis followed by a rotation of
        ``p`` radians around the y-axis followed by a rotation of ``y`` radians
        around the z-axis. These are the Z1-Y2-X3 Tait-Bryan angles. See
        Wikipedia_ for more information.

        .. _Wikipedia: https://en.wikipedia.org/wiki/Euler_angles#Rotation_matrix

        :param rpy: The roll-pitch-yaw coordinates in order (x-rot, y-rot, z-rot).
        :return: An SO3 object
        """
        return SO3(_quaternion_from_rpy(r, p, y))

    @staticmethod
    def from_axis_angle(axis, angle):
        return SO3(_quaternion_from_axis_angle(axis, angle))

    def __mul__(self, other):
        return SO3(_rotation_multiply(self.q, other.q))

    @staticmethod
    def from_unit_axes(x, y, z, rtol=1e-7, atol=1e-7):
        return SO3(_unit_axes_to_quaternion(x, y, z, atol, rtol))

    @property
    def inverse(self):
        """
        :return: The inverse of the orientation
        """
        return SO3(_quaternion_inverse(self.q))

    @property
    def radians(self):
        return _quaternion_to_radians(self.q)

    @property
    def degrees(self):
        return np.degrees(self.radians)

    @property
    def conjugate(self):
        return self.inverse

    @property
    def rpy(self):
        """
        This might not be the most numerically stable and should probably be replaced
        by whatever Eigen has
        """
        return _quaternion_to_rpy(self.q)

    @property
    def transformation_matrix(self):
        mat = np.eye(4)
        mat[:3, :3] = self.matrix
        return mat

    @property
    def xyzw(self):
        """
        :return: A list representation of the quaternion as xyzw
        """
        w, x, y, z = self.q
        return [x, y, z, w]

    @property
    def wxyz(self):
        """
        :return: A list representation of the quaternion as wxyz
        """
        w, x, y, z = self.q
        return [w, x, y, z]

    @property
    def matrix(self):
        """
        :return: The matrix representation of the orientation
        """
        return _quaternion_to_matrix(self.q)


@nb.jit(nopython=True, cache=True)
def _pose_multiply(pos1, q1, pos2, q2):
    q = _rotation_multiply(q1, q2)
    pos = np.dot(_quaternion_to_matrix(q1), pos2) + pos1
    return pos, q


@nb.jit(nopython=True, cache=True)
def _pose_inverse(pos, q):
    qinv = _quaternion_inverse(q)
    posinv = np.dot(-_quaternion_to_matrix(qinv), pos)
    return posinv, qinv


class SE3:
    """
    A generic class defining a 3D pose with some helper functions for easy conversions
    """

    so3: SO3

    def __init__(self, pos: np.ndarray, quaternion: np.ndarray):
        assert pos.shape == (3,)
        self.pos = pos
        self.so3 = SO3(quaternion)

    def __mul__(self, other):
        """
        Allows for numpy-style matrix multiplication using `@`
        """
        pos, q = _pose_multiply(self.pos, self.so3.q, other.pos, other.so3.q)
        return SE3(pos, q)

    @property
    def inverse(self):
        """
        :return: The inverse transformation
        """
        pos, q = _pose_inverse(self.pos, self.so3.q)
        return SE3(pos, q)

    @property
    def matrix(self):
        """
        :return: The internal matrix representation
        """
        m = self.so3.transformation_matrix
        m[:3, 3] = self.pos
        return m

    @property
    def xyz(self):
        """
        :return: The translation vector
        """
        x, y, z = self.pos
        return [x, y, z]

    @staticmethod
    def from_matrix(matrix, rtol=1e-7, atol=1e-7):
        q = _quaternion_trace_method(np.copy(matrix[:3, :3]), rtol, atol)
        return SE3(np.copy(matrix[:3, 3]), q)

    def __repr__(self):
        return "\n".join(
            [
                "SE3(",
                f"    pos={repr(self.pos)},",
                f"    quaternion={repr(self.so3.q)},",
                ")",
            ]
        )

    @staticmethod
    def from_unit_axes(origin, x, y, z, rtol=1e-7, atol=1e-7):
        """
        Constructs SE3 object from unit axes indicating direction and an origin

        :param origin: np.array indicating the placement of the origin
        :param x: A unit axis indicating the direction of the x axis
        :param y: A unit axis indicating the direction of the y axis
        :param z: A unit axis indicating the direction of the z axis
        :return: SE3 object
        """
        q = _unit_axes_to_quaternion(x, y, z, atol, rtol)
        return SE3(origin, q)
