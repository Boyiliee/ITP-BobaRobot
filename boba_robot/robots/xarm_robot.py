import time
from typing import Optional

import numpy as np
from pyquaternion import Quaternion

from boba_robot.robots.base import RobotDriver, RobotState


def _aa_from_quat(quat: np.ndarray) -> np.ndarray:
    """Convert a quaternion to an axis-angle representation.

    Args:
        quat (np.ndarray): The quaternion to convert.

    Returns:
        np.ndarray: The axis-angle representation of the quaternion.
    """
    assert quat.shape == (4,), "Input quaternion must be a 4D vector."
    norm = np.linalg.norm(quat)
    assert norm != 0, "Input quaternion must not be a zero vector."
    quat = quat / norm  # Normalize the quaternion

    Q = Quaternion(w=quat[0], x=quat[1], y=quat[2], z=quat[3])
    angle = Q.angle
    axis = Q.axis
    aa = axis * angle
    return aa


def _quat_from_aa(aa: np.ndarray) -> np.ndarray:
    """Convert an axis-angle representation to a quaternion.

    Args:
        aa (np.ndarray): The axis-angle representation to convert.

    Returns:
        np.ndarray: The quaternion representation of the axis-angle.
    """
    assert aa.shape == (3,), "Input axis-angle must be a 3D vector."
    if np.linalg.norm(aa) == 0:
        return np.array([0, 0, 0, 1])
    else:
        axis = aa / np.linalg.norm(aa)
        angle = np.linalg.norm(aa)
        Q = Quaternion(axis=axis, angle=angle)
        return np.array([Q.w, Q.x, Q.y, Q.z])


class XArmDriver(RobotDriver):
    GRIPPER_OPEN = 800
    GRIPPER_CLOSE = 0

    def __init__(
        self, ip: str, real: bool = True, speed: float = 500.0, joint_speed: float = 4.0
    ):
        self.real = real
        if real:
            from xarm.wrapper import XArmAPI

            print(f"connecting to {ip}")
            self.robot = XArmAPI(ip, is_radian=True)
        else:
            self.robot = None

        self._mode = 0
        self._clear_error_states()
        self._speed = speed
        self._joint_speed = joint_speed
        self._set_gripper_position(self.GRIPPER_OPEN)

    def num_joints(self) -> int:
        return 7

    def command_qpos(self, joints: np.ndarray) -> None:
        """Command the leader robot to a given state.

        Args:
            joints (np.ndarray): The state to command the leader robot to.
        """
        if self.robot is None:
            return
        code  = self.robot.set_servo_angle(angle=joints, wait=True, speed=self._joint_speed)
        if code!=0:
            print(code)
            self._clear_error_states()
            self.command_qpos(joints)

    def command_ee_pos(
        self,
        x: Optional[float] = None,
        y: Optional[float] = None,
        z: Optional[float] = None,
        quat: Optional[np.ndarray] = None,
    ) -> None:
        """Command the leader robot to a given state.

        Args:
            x (Optional[float], optional): The x position of the end-effector. Defaults to None.
            y (Optional[float], optional): The y position of the end-effector. Defaults to None.
            z (Optional[float], optional): The z position of the end-effector. Defaults to None.
            quat (Optional[np.ndarray], optional): The quaternion of the end-effector. Defaults to None.
        """
        if self.robot is None:
            return

        if 0 != self._mode:
            self.robot.set_mode(0)
            self._mode = 0
            time.sleep(0.5)

        if quat is None:
            code = self.robot.set_position(
                x=None if x is None else x * 1000,
                y=None if y is None else y * 1000,
                z=None if z is None else z * 1000,
                wait=True,
                speed=self._speed,
            )
        else:
            assert (
                x is not None and y is not None and z is not None
            ), "Must specify x, y, and z if quat is specified."
            aa = _aa_from_quat(quat)
            axis_angle_pose = [x * 1000, y * 1000, z * 1000, aa[0], aa[1], aa[2]]
            code = self.robot.set_position_aa(axis_angle_pose, wait=True, speed=self._speed)
        if code!=0:
            print(code)
            self._clear_error_states()
            self.command_ee_pos(x, y, z, quat)


    def command_gripper_pos(self, pos: float) -> None:
        """Command the leader robot to a given state.

        Args:
            pos (float): The position of the gripper.
        """
        if self.robot is None:
            return
        if 0 != self._mode:
            self.robot.set_mode(0)
            self._mode = 0
            time.sleep(0.5)
        pos = int(pos * (self.GRIPPER_CLOSE - self.GRIPPER_OPEN) + self.GRIPPER_OPEN)
        self._set_gripper_position(pos)

    def command_servo_ee_pos(
        self, x: float, y: float, z: float, quat: np.ndarray
    ) -> None:
        """Command the leader robot to a given state.

        This is a servo command, meaning that the robot will move to the commanded state with no delay.

        Args:
            x (float): The x position of the end-effector.
            y (float): The y position of the end-effector.
            z (float): The z position of the end-effector.
            quat (np.ndarray): The quaternion of the end-effector.
        """
        if self.robot is None:
            return
        if 1 != self._mode:
            self.robot.set_mode(1)
            self._mode = 1
            time.sleep(0.5)
            self.robot.set_mode(1)
            time.sleep(0.5)

        aa = _aa_from_quat(quat)
        axis_angle_pose = [x * 1000, y * 1000, z * 1000, aa[0], aa[1], aa[2]]
        code = self.robot.set_servo_cartesian_aa(axis_angle_pose, wait=False)

        print(code)

    def get_state(self) -> RobotState:
        """Get the current state of the leader robot.

        Returns:
            T: The current state of the leader robot.
        """
        if self.robot is None:
            return RobotState(
                qpos=np.zeros(7),
                ee_pos=np.zeros(3),
                ee_quat=np.zeros(4),
                gripper_pos=0.0,
            )

        gripper_pos = self._get_gripper_pos()
        code, servo_angle = self.robot.get_servo_angle(is_radian=True)
        while code != 0:
            print(f"Error code {code} in get_servo_angle().")
            self._clear_error_states()
            code, servo_angle = self.robot.get_servo_angle(is_radian=True)

        code, cart_pos = self.robot.get_position_aa(is_radian=True)
        while code != 0:
            print(f"Error code {code} in get_position().")
            self._clear_error_states()
            code, cart_pos = self.robot.get_position_aa(is_radian=True)

        cart_pos = np.array(cart_pos)
        aa = cart_pos[3:]
        cart_pos[:3] /= 1000
        return RobotState(
            qpos=np.array(servo_angle),
            ee_pos=cart_pos[:3],
            ee_quat=_quat_from_aa(aa),
            gripper_pos=gripper_pos,
        )

    def _clear_error_states(self):
        if self.robot is None:
            return
        self.robot.clean_error()
        self.robot.clean_warn()
        self.robot.motion_enable(True)
        time.sleep(1)
        self.robot.set_mode(self._mode)
        time.sleep(1)
        self.robot.set_collision_sensitivity(0)
        time.sleep(1)
        self.robot.set_state(state=0)
        time.sleep(1)
        self.robot.set_gripper_enable(True)
        time.sleep(1)
        self.robot.set_gripper_mode(0)
        time.sleep(1)
        self.robot.set_gripper_speed(3000)
        time.sleep(1)

    def _get_gripper_pos(self) -> float:
        if self.robot is None:
            return 0.0
        code, gripper_pos = self.robot.get_gripper_position()
        while code != 0 or gripper_pos is None:
            print(f"Error code {code} in get_gripper_position(). {gripper_pos}")
            time.sleep(0.001)
            code, gripper_pos = self.robot.get_gripper_position()
            if code == 22:
                self._clear_error_states()

        normalized_gripper_pos = (gripper_pos - self.GRIPPER_CLOSE) / (
            self.GRIPPER_OPEN - self.GRIPPER_CLOSE
        )
        return normalized_gripper_pos

    def _set_gripper_position(self, pos: int) -> None:
        if self.robot is None:
            return
        self.robot.set_gripper_position(pos, wait=True)


if __name__ == "__main__":
    # driver = XArmDriver("192.168.1.233")
    # for _ in range(2):
    #     state = driver.get_state()
    #     print(state)
    # driver.command_ee_pos(0.3, 0, 0.05, quat=np.array([1, 0, 1, 0]))

    import time

    from xarm.wrapper import XArmAPI

    ip = "192.168.1.233"
    robot = XArmAPI(ip, is_radian=True)
    robot.clean_error()
    robot.clean_warn()
    robot.motion_enable(True)
    time.sleep(1)
    robot.set_mode(0)
    time.sleep(1)
    robot.set_collision_sensitivity(0)
    time.sleep(1)
    robot.set_mode(0)
    time.sleep(1)

    _, servo_angle = robot.get_servo_angle(is_radian=True)

    servo_angle[-1] += 0.5
    robot.set_servo_angle(angle=servo_angle, wait=True, speed=1, mvacc=1)
    servo_angle[-1] -= 0.5
    robot.set_servo_angle(angle=servo_angle, wait=True, speed=10, mvacc=1)
    servo_angle[-1] += 0.5
    robot.set_servo_angle(angle=servo_angle, wait=True, speed=10, mvacc=10)
    servo_angle[-1] -= 0.5
    robot.set_servo_angle(angle=servo_angle, wait=True, speed=100, mvacc=100)
