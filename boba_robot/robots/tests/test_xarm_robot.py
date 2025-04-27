import numpy as np
import pytest

from boba_robot.robots.xarm_robot import _aa_from_quat, _quat_from_aa


def test_aa_from_quat_simple():
    quat = np.array([1, 0, 0, 0])
    expected_aa = np.array([0, 0, 0])
    np.testing.assert_almost_equal(_aa_from_quat(quat), expected_aa, decimal=6)

    quat = np.array([1, 0, 0, 1])
    expected_aa = np.array([0, 0, np.pi / 2])
    np.testing.assert_almost_equal(_aa_from_quat(quat), expected_aa, decimal=6)

    quat = np.array([1, 0, 1, 0])
    expected_aa = np.array([0, np.pi / 2, 0])
    np.testing.assert_almost_equal(_aa_from_quat(quat), expected_aa, decimal=6)

    quat = np.array([1, 1, 0, 0])
    expected_aa = np.array([np.pi / 2, 0, 0])
    np.testing.assert_almost_equal(_aa_from_quat(quat), expected_aa, decimal=6)


def test_aa_from_quat():
    quat = np.array([np.cos(np.pi / 4), 0, 0, np.sin(np.pi / 4)])
    expected_aa = np.array([0, 0, np.pi / 2])
    np.testing.assert_almost_equal(_aa_from_quat(quat), expected_aa, decimal=6)


def test_quat_from_aa():
    aa = np.array([0, 0, np.pi / 2])
    expected_quat = np.array([np.cos(np.pi / 4), 0, 0, np.sin(np.pi / 4)])
    np.testing.assert_almost_equal(_quat_from_aa(aa), expected_quat, decimal=6)


def test_round_trip():
    original_quat = np.array([0, 0, np.sin(np.pi / 4), np.cos(np.pi / 4)])
    aa = _aa_from_quat(original_quat)
    quat = _quat_from_aa(aa)
    np.testing.assert_almost_equal(quat, original_quat, decimal=6)


def test_invalid_input():
    with pytest.raises(AssertionError):
        _aa_from_quat(np.array([0, 0, 0, 0]))  # zero quaternion
    with pytest.raises(AssertionError):
        _quat_from_aa(np.array([0, 0, 0, 0]))  # wrong shape
