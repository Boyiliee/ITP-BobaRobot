from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

# TODO rewrite this file to follow the RobotDriver interface, like the xarm


@dataclass
class RobotState:
    """A class representing the state of a robot at a given time."""

    qpos: Optional[np.ndarray] = None
    """Joint position."""

    pose: Optional[np.ndarray] = None
    """End effector pose as a length 7 array (x, y, z, qw, qx, qy, qz)"""

    gripper: Optional[float] = None
    """Gripper position."""


class URRobot:
    """A class representing a UR robot."""

    def __init__(self, robot_ip: str = "192.168.1.10"):
        import rtde_control
        import rtde_receive

        from boba_robot.robots.robotiq_gripper import RobotiqGripper

        [print("in ur robot") for _ in range(4)]
        self.robot = rtde_control.RTDEControlInterface(robot_ip)
        self.r_inter = rtde_receive.RTDEReceiveInterface(robot_ip)

        self.gripper = RobotiqGripper()
        self.gripper.connect(hostname=robot_ip, port=63352)
        # gripper.activate()

        [print("connected") for _ in range(4)]

        self._free_drive = False
        self.robot.endFreedriveMode()

    def _get_gripper_pos(self) -> float:
        import time

        time.sleep(0.01)
        gripper_pos = self.gripper.get_current_position()
        gripper_pos = 0
        assert 0 <= gripper_pos <= 255, "Gripper position must be between 0 and 255"
        return gripper_pos / 255

    def get_state(self) -> RobotState:
        """Get the current state of the robot.

        Returns:
            RobotState: The current state of the robot.
        """
        robot_joints = self.r_inter.getActualQ()
        robot_pose = self.r_inter.getActualTCPPose()
        gripper_pos = self._get_gripper_pos()
        return RobotState(qpos=robot_joints, gripper=gripper_pos, pose=robot_pose)

    def command_state(self, robot_state: RobotState) -> None:
        """Command the robot to a given state.

        Args:
            robot_state (RobotState): The state to command the robot to.
        """
        # assert that qpos and pose are not both used
        assert (
            robot_state.qpos is None or robot_state.pose is None
        ), "Cannot command both qpos and pose"

        velocity = 0.5
        acceleration = 0.5
        dt = 1.0 / 500  # 2ms
        lookahead_time = 0.2
        gain = 100

        t_start = self.robot.initPeriod()
        if robot_state.pose is not None:
            robot_pose = robot_state.pose
            raise NotImplementedError
            self.robot.moveL(
                robot_pose, velocity, acceleration, dt, lookahead_time, gain
            )

        if robot_state.qpos is not None:
            robot_joints = robot_state.pos[:6]
            self.robot.moveJ(
                robot_joints, velocity, acceleration, dt, lookahead_time, gain
            )
        if robot_state.gripper is not None:
            gripper_pos = robot_state.gripper * 255
            self.gripper.move(gripper_pos, 255, 10)
        self.robot.waitPeriod(t_start)

    def freedrive_enabled(self) -> bool:
        """Check if the robot is in freedrive mode.

        Returns:
            bool: True if the robot is in freedrive mode, False otherwise.
        """
        return self._free_drive

    def set_freedrive_mode(self, enable: bool) -> None:
        """Set the freedrive mode of the robot.

        Args:
            enable (bool): True to enable freedrive mode, False to disable it.
        """
        if enable and not self._free_drive:
            self._free_drive = True
            self.robot.FreedriveMode()
        elif not enable and self._free_drive:
            self._free_drive = False
            self.robot.endFreedriveMode()

    def get_observations(self) -> Dict[str, np.ndarray]:
        """Get the observations of the robot.

        Returns:
            Dict[str, np.ndarray]: The observations of the robot.
        """
        state = self.get_state()
        return {
            "qpos": state.qpos,
            "gripper": state.gripper,
        }


def main():
    robot_ip = "192.168.1.10"
    ur = URRobot(robot_ip, no_gripper=True)
    print(ur)
    ur.set_freedrive_mode(True)


if __name__ == "__main__":
    main()
