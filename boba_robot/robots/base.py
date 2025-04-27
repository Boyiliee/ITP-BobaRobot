import time
from dataclasses import dataclass
from typing import Optional, Protocol

import numpy as np


@dataclass
class RobotState:
    """A class representing the joint state of a robot at a given time."""

    # joint information
    qpos: np.ndarray
    # qvel: np.ndarray
    # qeff: np.ndarray

    # end-effector information
    ee_pos: np.ndarray
    ee_quat: np.ndarray

    # gripper information
    gripper_pos: float


class RobotDriver(Protocol):
    """Robot protocol.

    A protocol for a robot that can be controlled.
    """

    def num_joints(self) -> int:
        """Get the number of joints of the robot.

        Returns:
            int: The number of joints of the robot.
        """
        ...

    def get_state(self) -> RobotState:
        """Get the current state of the leader robot.

        Returns:
            T: The current state of the leader robot.
        """
        ...

    def command_qpos(self, joints: np.ndarray) -> None:
        """Command the leader robot to a given state.

        Args:
            joints (np.ndarray): The state to command the leader robot to.
        """
        ...

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
        ...

    def command_gripper_pos(self, pos: float) -> None:
        """Command the leader robot to a given state.

        Args:
            pos (float): The position of the gripper.
        """
        ...

    def command_servo_ee_pos(
        self, x: float, y: float, z: float, quat: np.ndarray
    ) -> None:
        """Command the leader robot to a given state.

        Args:
            x (float): The x position of the end-effector.
            y (float): The y position of the end-effector.
            z (float): The z position of the end-effector.
            quat (np.ndarray): The quaternion of the end-effector.
        """
        ...


class Rate:
    def __init__(self, *, duration):
        self.duration = duration
        self.last = time.time()

    def sleep(self, duration=None) -> None:
        duration = self.duration if duration is None else duration
        assert duration >= 0
        now = time.time()
        passed = now - self.last
        remaining = duration - passed
        assert passed >= 0
        if remaining > 0.0001:
            time.sleep(remaining)
        self.last = time.time()
