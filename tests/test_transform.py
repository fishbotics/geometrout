from geometrout.transform import SE3, SO3
import numpy as np
from math import pi, sin, cos
from pyquaternion import Quaternion


ALMOST_EQUAL_TOLERANCE = 8


def test_init_from_explicit_matrix():
    def R_z(theta):
        """
        Generate a rotation matrix describing a rotation of theta degrees about the z-axis
        """
        c = cos(theta)
        s = sin(theta)
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

    v = np.array([1, 0, 0])
    for angle, quat in [
        (0, np.array([1.0, 0, 0, 0])),
        (pi / 6, np.array([0.96592583, 0.0, 0.0, 0.25881905])),
        (pi / 4, np.array([0.92387953, 0.0, 0.0, 0.38268343])),
        (pi / 2, np.array([0.70710678, 0.0, 0.0, 0.70710678])),
        (pi, np.array([0, 0, 0, 1.0])),
        (4 * pi / 3, np.array([-0.5, 0.0, 0.0, 0.8660254])),
        (3 * pi / 2, np.array([-0.70710678, 0.0, 0.0, 0.70710678])),
        (2 * pi, np.array([1.0, 0.0, 0.0, -1.2246468e-16])),
    ]:
        R = R_z(angle)  # rotation matrix describing rotation of 90 about +z
        np.testing.assert_almost_equal(
            SO3.from_matrix(R).q, quat, decimal=ALMOST_EQUAL_TOLERANCE
        )

    R = np.matrix(np.eye(3))
    np.testing.assert_almost_equal(
        SO3.from_matrix(np.eye(3)).q, np.array([1.0, 0, 0, 0])
    )
