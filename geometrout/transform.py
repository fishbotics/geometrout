from pyquaternion import Quaternion
import numpy as np


class SO3:
    """
    A generic class defining a 3D orientation. Mostly a wrapper around quaternions
    """

    def __init__(self, quat):
        """
        :param quat: Quaternion
        """
        if isinstance(quat, Quaternion):
            self._quat = quat
        else:
            self._quat = Quaternion(np.asarray(quaternion))

    @classmethod
    def from_rpy(cls, rpy):
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
        coords = np.asanyarray(rpy, dtype=np.float64)
        c3, c2, c1 = np.cos(rpy)
        s3, s2, s1 = np.sin(rpy)

        matrix = np.array(
            [
                [c1 * c2, (c1 * s2 * s3) - (c3 * s1), (s1 * s3) + (c1 * c3 * s2)],
                [c2 * s1, (c1 * c3) + (s1 * s2 * s3), (c3 * s1 * s2) - (c1 * s3)],
                [-s2, c2 * s3, c2 * c3],
            ],
            dtype=np.float64,
        )
        return SO3(Quaternion(matrix=matrix))

    @property
    def inverse(self):
        """
        :return: The inverse of the orientation
        """
        return SO3(self._quat.inverse)

    @property
    def transformation_matrix(self):
        return self._quat.transformation_matrix

    @property
    def xyzw(self):
        """
        :return: A list representation of the quaternion as xyzw
        """
        return self._quat.vector.tolist() + [self._quat.scalar]

    @property
    def wxyz(self):
        """
        :return: A list representation of the quaternion as wxyz
        """
        return [self._quat.scalar] + self._quat.vector.tolist()

    @property
    def matrix(self):
        """
        :return: The matrix representation of the orientation
        """
        return self._quat.rotation_matrix


class SE3:
    """
    A generic class defining a 3D pose with some helper functions for easy conversions
    """

    def __init__(self, matrix=None, xyz=None, quat=None, so3=None, rpy=None):
        assert bool(matrix is None) != bool(
            xyz is None and (bool(quat is None) ^ bool(so3 is None) ^ bool(rpy is None))
        )
        if matrix is not None:
            self._xyz = matrix[:3, 3]
            self._so3 = SO3(Quaternion(matrix=matrix))
        else:
            self._xyz = np.asarray(xyz)
            if quat is not None:
                assert isinstance(quat, Quaternion)
                self._so3 = SO3(quat)
            elif rpy is not None:
                self._so3 = SO3.from_rpy(rpy)
            else:
                self._so3 = so3

    def __matmul__(self, other):
        """
        Allows for numpy-style matrix multiplication using `@`
        """
        return SE3(matrix=self.matrix @ other.matrix)

    @property
    def inverse(self):
        """
        :return: The inverse transformation
        """
        return SE3(xyz=-self._xyz, so3=self._so3.inverse)

    @property
    def matrix(self):
        """
        :return: The internal matrix representation
        """
        m = self._so3.transformation_matrix
        m[:3, 3] = self.xyz
        return m

    @property
    def so3(self):
        """
        :return: The representation of orientation
        """
        return self._so3

    @so3.setter
    def so3(self, val):
        """
        :param val: A pose object
        """
        assert isinstance(val, SO3)
        self._so3 = val

    @property
    def xyz(self):
        """
        :return: The translation vector
        """
        return self._xyz.tolist()

    @xyz.setter
    def xyz(self, val):
        """
        :return: The translation vector
        """
        self._xyz = np.asarray(val)

    @classmethod
    def from_unit_axes(cls, origin, x, y, z):
        """
        Constructs SE3 object from unit axes indicating direction and an origin

        :param origin: np.array indicating the placement of the origin
        :param x: A unit axis indicating the direction of the x axis
        :param y: A unit axis indicating the direction of the y axis
        :param z: A unit axis indicating the direction of the z axis
        :return: SE3 object
        """
        assert np.isclose(np.dot(x, y), 0)
        assert np.isclose(np.dot(x, z), 0)
        assert np.isclose(np.dot(y, z), 0)
        assert np.isclose(np.linalg.norm(x), 1)
        assert np.isclose(np.linalg.norm(y), 1)
        assert np.isclose(np.linalg.norm(z), 1)
        m = np.eye(4)
        m[:3, 0] = x
        m[:3, 1] = y
        m[:3, 2] = z
        m[:3, 2] = origin
        return cls(matrix=m)