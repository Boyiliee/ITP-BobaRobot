from typing import Dict, Optional, Tuple

import numpy as np
from pyquaternion import Quaternion

from boba_robot.cameras.base import CameraDriver
from boba_robot.cameras.realsense import RealSenseCamera
from boba_robot.robots.base import RobotDriver

Z_GRASP = 0.06
Z_TRANSPORT = 0.2

BASE_QUAT = Quaternion(x=1, y=0, z=1, w=0)


class FeedBackPolicy:
    def __init__(self, camera: CameraDriver):
        from boba_robot.grounded_dino import get_ovod_model

        self._grounded_dino = get_ovod_model()
        self._camera = camera

    def grasp(
        self,
        robot: RobotDriver,
        language: str,
        x: int,
        y: int,
        z: int,
        rotz: bool = False,
    ):
        BOX_THRESHOLD = 0.3
        TEXT_THRESHOLD = 0.3
        # warm up the camera
        for _ in range(5):
            self._camera.read()

        error = 1e10
        k = 0.0002
        r = np.sqrt(x**2 + y**2)

        import cv2
        import supervision as sv

        cv2.namedWindow("image")
        while error > 0.001:
            ee_pos = robot.get_state().ee_pos
            x_curr, y_curr = ee_pos[0], ee_pos[1]
            r_curr = np.sqrt(x_curr**2 + y_curr**2)
            image, _ = self._camera.read()
            detections = self._grounded_dino.predict_with_classes(
                image=image[:, :, ::-1],
                classes=[language],
                box_threshold=BOX_THRESHOLD,
                text_threshold=TEXT_THRESHOLD,
            )
            if len(detections) == 0:
                raise RuntimeError(f"No detections of {language} found")
            box_annotator = sv.BoxAnnotator()
            annotated_image = box_annotator.annotate(
                scene=np.ascontiguousarray(image), detections=detections
            )
            cv2.imshow("image", annotated_image[:, :, ::-1])
            cv2.moveWindow("image", 50, 50)
            cv2.waitKey(1)

            centers_px = []
            areas = []
            for xyxy in detections.xyxy:
                centers_px.append((xyxy[0] + xyxy[2]) / 2)
                areas.append((xyxy[2] - xyxy[0]) * (xyxy[3] - xyxy[1]))
            H, W, C = image.shape

            # pick the center of the largest detection
            center_px = centers_px[np.argmax(areas)]
            error = center_px - W / 2
            control_action = error * k

            angle = np.arctan2(x_curr, y_curr)
            desired_angle = angle + control_action
            y_command = np.cos(desired_angle) * r_curr
            x_command = np.sin(desired_angle) * r_curr
            quat = get_quat(x_command, y_command)

            # print(
            #     f"(x, y, angle, control_actions, desired_angle, x_new, y_new): {x_curr:.3}, {y_curr:.3}, {angle:.3}, {control_action:.3}, {desired_angle:.3}, {x_command:.3}, {y_command:.3}"
            # )
            # robot.command_servo_ee_pos(x_command, y_command, z, quat)
            robot.command_ee_pos(x_command, y_command, z, quat)

        if rotz:
            joint_state = robot.get_state().qpos
            joint_state[-1] -= np.pi
            robot.command_qpos(joint_state)

        # object should be centered so we can approach
        extra_delta = 0.011
        y_command = np.cos(desired_angle) * (r + extra_delta)
        x_command = np.sin(desired_angle) * (r + extra_delta)
        robot.command_ee_pos(x_command, y_command, z)


def format_message(message: str, role: str = "user") -> Dict[str, str]:
    return {
        "role": role,
        "content": message,
    }


def get_quat(x: float, y: float) -> np.ndarray:
    """Get the quaternion of the vector pointing at the given (x, y) direction.

    The quaternion is computed with respect to the [1, 0] vector.
    """
    angle = np.arctan2(y, x)

    # rotate around z axis by the angle angle
    quat = Quaternion(axis=[0, 0, 1], angle=angle)
    quat = quat * BASE_QUAT
    return quat.elements


def get_x_y_offset(x: float, y: float, offset: float = 0.06) -> Tuple[float, float]:
    norm: float = np.linalg.norm([x, y])  # type: ignore
    x_sub = x / norm * offset
    y_sub = y / norm * offset

    if x > 0:
        new_x = max(x - x_sub, 0)
    else:
        new_x = min(x - x_sub, 0)

    if y > 0:
        new_y = max(y - y_sub, 0)
    else:
        new_y = min(y - y_sub, 0)
    return new_x, new_y


class RobotLanguageInterface:
    def __init__(self, robot: Optional[RobotDriver], use_feedback_policy: bool = False):
        """Initialize the robot language interface.

        Args:
            robot (Optional[RobotDriver]): The robot driver to use.
            use_feedback_policy (bool): Whether to use the feedback policy.
        """
        self._robot = robot
        self._gripper_offset = 0.18  # Offset from the center of the tcp to the center of the end of the gripper
        self._use_feedback_policy = use_feedback_policy
        if self._use_feedback_policy:
            wrist_camera_ids = "918512072239"
            camera = RealSenseCamera(device_id=wrist_camera_ids)
            self._feedback_policy = FeedBackPolicy(camera)

    def get_qpos(self):
        if self._robot is None:
            return np.zeros(7)
        return self._robot.get_state().qpos

    def set_qpos(self, qpos: np.ndarray):
        if self._robot is None:
            return
        self._robot.command_qpos(qpos)

    def set_ee(
        self,
        x: Optional[float] = None,
        y: Optional[float] = None,
        z: Optional[float] = None,
        quat: Optional[np.ndarray] = None,
    ) -> None:
        if self._robot is None:
            return
        self._robot.command_ee_pos(x, y, z, quat)

    def grasp_cup(self, x: float, y: float, description: str = "") -> Tuple[bool, str]:
        """Grasp a cup at the given x, y coordinates. Should ONLY be used on cups (not bowls or stack of cups, other objects).

        Args:
            x (float): The x coordinate of the cup.
            y (float): The y coordinate of the cup.
            description (str): The description of the cup to grasp.

        Returns:
            Tuple[bool, str]: Whether the grasp was successful and a message.
        """
        print(f"in RobotInterface.grasp_cup: {x}, {y}, {description}")
        additional_tolerance = 0.095
        x, y = get_x_y_offset(x, y, self._gripper_offset)  # back project to tcp
        if self._robot is None:
            return True, "Grasp successful"

        quat = get_quat(x, y)

        x_app, y_app = get_x_y_offset(x, y, offset=additional_tolerance)
        self._robot.command_ee_pos(z=Z_TRANSPORT)
        self._robot.command_ee_pos(x_app, y_app, Z_TRANSPORT, quat)
        self._robot.command_ee_pos(z=Z_GRASP)

        if self._use_feedback_policy:
            try:
                description = "cup"
                self._feedback_policy.grasp(self._robot, description, x, y, Z_GRASP)
            except Exception as e:
                print(e)
                return False, f"Grasp failed: {e}"
        else:
            self._robot.command_ee_pos(x, y, Z_GRASP, quat)

        self._robot.command_gripper_pos(0.28)
        self._robot.command_ee_pos(z=Z_TRANSPORT)
        return True, "Grasp successful"

    def pour(self, x: float, y: float, description: str) -> Tuple[bool, str]:
        """Execute a pouring action at the given x, y coordinates.

        The pouring action should only be executed to a location where the cup is exactly known.
        This is achived by firs placing the target cup at a specific location.

        Args:
            x (float): The x coordinate of where to pour.
            y (float): The y coordinate of where to pour.
            description (str): The description of what is being poured.

        Returns:
            Tuple[bool, str]: Whether the pour was successful and a message.
        """
        print(f"in RobotInterface.pour: {x}, {y}")
        x, y = get_x_y_offset(x, y, self._gripper_offset)  # back project to tcp
        if self._robot is None:
            return True, "Pour successful"

        # Get the current position of the robot
        current_pos = self._robot.get_state().ee_pos[:2]

        # Calculate the distances to the left and right sides of the target position
        offset = 0.067
        angle = np.arctan2(y, x)
        x_left = x + offset * np.cos(angle + np.pi / 2)  # Left side
        y_left = y + offset * np.sin(angle + np.pi / 2)
        x_right = x + offset * np.cos(angle - np.pi / 2)  # Right side
        y_right = y + offset * np.sin(angle - np.pi / 2)

        dist_left = np.linalg.norm(np.array([x_left, y_left]) - current_pos)
        dist_right = np.linalg.norm(np.array([x_right, y_right]) - current_pos)

        # Choose the side with the shorter distance
        if "milk" in description.lower():
            pour_angle = np.pi * 12 / 27
        else:
            pour_angle = np.pi * 8.8 / 27

        if dist_left < dist_right:
            x_pour, y_pour = x_left, y_left
            joint_delta = pour_angle
        else:
            x_pour, y_pour = x_right, y_right
            joint_delta = -pour_angle

        # Move to the pouring location
        quat = get_quat(x, y)
        self._robot.command_ee_pos(x_pour, y_pour, Z_GRASP + 0.12, quat)

        # Rotate the wrist to pour
        joint_state = self._robot.get_state().qpos
        og_joint_state = np.copy(joint_state)
        joint_state[-1] += joint_delta

        self._robot.command_qpos(joint_state)
        self._robot.command_qpos(og_joint_state)

        # Return to the transport position
        self._robot.command_ee_pos(z=Z_TRANSPORT)
        return True, "Pour successful"

    def place_cup(self, x: float, y: float) -> Tuple[bool, str]:
        """Place a cup to the given x, y coordinates. The cup should be grasped first using "grasp" function before being placed.

        Args:
            x (float): The x coordinate of the final location.
            y (float): The y coordinate of the final location.

        Returns:
            Tuple[bool, str]: Whether the place was successful and a message.
        """
        print(f"in RobotInterface.place_cup: {x}, {y}")
        x, y = get_x_y_offset(x, y, self._gripper_offset)  # back project to tcp
        if self._robot is None:
            return True, "Place successful"
        self._robot.command_ee_pos(z=Z_TRANSPORT)
        quat = get_quat(x, y)
        self._robot.command_ee_pos(x, y, Z_TRANSPORT, quat)

        if x < 0.2:
            self._robot.command_ee_pos(z=Z_GRASP + 0.01)
        else:
            self._robot.command_ee_pos(z=Z_GRASP)
        self._robot.command_gripper_pos(0.0)

        # retract backwards
        x_app, y_app = get_x_y_offset(x, y)
        if x < 0.2:
            self._robot.command_ee_pos(x_app, y_app, Z_GRASP + 0.01, quat)
        else:
            self._robot.command_ee_pos(x_app, y_app, Z_GRASP, quat)
        self._robot.command_ee_pos(z=Z_TRANSPORT)
        return True, "Place successful"

    def stir(self, x: float, y: float) -> Tuple[bool, str]:
        """Stir an object at the given x, y coordinates.

        Args:
            x (float): The x coordinate of the object.
            y (float): The y coordinate of the object.

        Returns:
            Tuple[bool, str]: Whether the grasp was successful and a message.
        """
        print(f"in RobotInterface.stir: {x}, {y}")
        x, y = get_x_y_offset(x, y, self._gripper_offset)  # back project to tcp
        if self._robot is None:
            return True, "Stir successful"
        raise NotImplementedError  # TODO

    def grasp_empty_cup_from_stack(self, x: float, y: float) -> Tuple[bool, str]:
        """Grab an empty cup at the given x, y coordinates from the stack. Should ONLY be used on a cup stack.

        Args:
            x (float): The x coordinate of the object.
            y (float): The y coordinate of the object.

        Returns:
            Tuple[bool, str]: Whether the grasp was successful and a message.
        """
        print(f"in RobotInterface.grasp_empty_cup_from_stack: {x}, {y}")
        additional_tolerance = 0.095
        x, y = get_x_y_offset(x, y, self._gripper_offset)  # back project to tcp
        if self._robot is None:
            return True, "Grasp successful"

        quat = get_quat(x, y)
        Z_STACK = 0.13

        x_app, y_app = get_x_y_offset(x, y, offset=additional_tolerance)
        self._robot.command_ee_pos(z=Z_STACK)
        self._robot.command_ee_pos(x_app, y_app, Z_STACK, quat)
        start_joints = self._robot.get_state().qpos

        if self._use_feedback_policy:
            try:
                description = "cup"
                self._feedback_policy.grasp(
                    self._robot, description, x, y, Z_STACK, rotz=True
                )
            except Exception as e:
                print(e)
                return False, f"Grasp failed: {e}"
        else:
            self._robot.command_ee_pos(x, y, Z_STACK, quat)

        self._robot.command_gripper_pos(0.194)
        self._robot.command_ee_pos(z=Z_STACK + 0.15)

        # backup
        curr_x, curr_y = self._robot.get_state().ee_pos[:2]
        x_app, y_app = get_x_y_offset(curr_x, curr_y, offset=additional_tolerance)
        self._robot.command_ee_pos(x_app, y_app, Z_STACK + 0.15)

        # rotate wrist back
        self._robot.command_qpos(start_joints)
        return True, "Grasp successful"

    def scoop_boba_to_location(self, x: float, y: float) -> Tuple[bool, str]:
        """Scoop an boba into the given x, y. The provided x, y coordinate should be the final location of the cup.

        Args:
            x (float): The x coordinate of the object.
            y (float): The y coordinate of the object.

        Returns:
            Tuple[bool, str]: Whether the grasp was successful and a message.
        """
        print(f"in RobotInterface.scoop_boba_to_location: {x}, {y}")
        x, y = get_x_y_offset(x, y, self._gripper_offset)  # back project to tcp
        if self._robot is None:
            return True, "Scoop successful"

        current_pose = self._robot.get_state().ee_pos
        # move to z = 0.3
        import time

        time.sleep(0.1)
        self._robot.command_ee_pos(z=0.28)
        time.sleep(0.1)
        if self._robot.get_state().ee_pos[2] < 0.27:
            return False, "Scoop failed: try again"

        reset_pos = np.array([0.31568179, -0.22398379, 0.19772279])
        reset_quat = np.array([0.264044, 0.66234467, -0.28171151, 0.64204278])
        self._robot.command_ee_pos(*reset_pos, quat=reset_quat)

        #  reset_joints = np.deg2rad([67, 95, -103, 48, -150, -19, -122])
        reset_joints = np.deg2rad([-1.3, -7.5, -7.5, 42, -95, 80, 121])
        self._robot.command_qpos(reset_joints)
        reset_state = self._robot.get_state()

        grasp_pos = np.array([0.388, -0.408, 0.3])
        grasp_quat = np.array([0.5, 0.5, -0.5, 0.5])

        pre_grasp_pos = np.array([0.388, -0.358, 0.3])
        pre_grasp_quat = np.array([0.5, 0.5, -0.5, 0.5])
        #  pre_grasp_joints = np.array(
        #      [0.334306, 1.536916, -1.740129, 1.489871, -2.608869, -0.296189, -2.172175]
        #  )

        # grasp ladel
        #  self._robot.command_qpos(pre_grasp_joints)
        self._robot.command_ee_pos(*pre_grasp_pos, quat=pre_grasp_quat)
        self._robot.command_ee_pos(*grasp_pos, quat=grasp_quat)
        # close gripper and move back
        self._robot.command_gripper_pos(1)
        self._robot.command_ee_pos(x=0.420, y=-0.37, z=0.4)
        prior_state = self._robot.get_state()

        # scoop
        ee_frame_to_scoop_transform = np.array(
            [
                [1, 0, 0, -0.165],
                [0, 1, 0, -0.03],
                [0, 0, 1, 0.0],
                [0, 0, 0, 1],
            ]
        )

        # move to boba location and scoop
        boba_x, boba_y = 0.45, -0.25
        self._robot.command_ee_pos(x=boba_x, y=boba_y, z=0.35)

        curr_state = self._robot.get_state()
        current_pose = np.eye(4)
        current_pose[:3, :3] = Quaternion(curr_state.ee_quat).rotation_matrix
        current_pose[:3, 3] = curr_state.ee_pos
        prep_angle = -np.pi / 7
        rotz = np.array(
            [
                [np.cos(prep_angle), -np.sin(prep_angle), 0, 0],
                [np.sin(prep_angle), np.cos(prep_angle), 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
        )
        final_ee_frame = (
            current_pose
            @ ee_frame_to_scoop_transform
            @ rotz
            @ np.linalg.inv(ee_frame_to_scoop_transform)
        )
        Q = Quaternion(matrix=final_ee_frame[:3, :3])
        self._robot.command_ee_pos(*final_ee_frame[:3, 3], quat=Q.elements)

        self._robot.command_ee_pos(z=0.21)
        curr_state = self._robot.get_state()
        # rotate in the local scoop frame by 30 degrees in the z axis

        current_pose = np.eye(4)
        current_pose[:3, :3] = Quaternion(curr_state.ee_quat).rotation_matrix
        current_pose[:3, 3] = curr_state.ee_pos
        angle = np.pi / 7 - prep_angle
        rotz = np.array(
            [
                [np.cos(angle), -np.sin(angle), 0, 0],
                [np.sin(angle), np.cos(angle), 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
        )
        final_ee_frame = (
            current_pose
            @ ee_frame_to_scoop_transform
            @ rotz
            @ np.linalg.inv(ee_frame_to_scoop_transform)
        )

        save_speed = self._robot._speed
        self._robot._speed = 100
        Q = Quaternion(matrix=final_ee_frame[:3, :3])
        self._robot.command_ee_pos(*final_ee_frame[:3, 3], quat=Q.elements)

        # lift
        self._robot.command_ee_pos(z=0.42)
        self._robot.command_ee_pos(z=0.38)
        self._robot.command_ee_pos(z=0.42)
        self._robot.command_ee_pos(z=0.40)
        self._robot.command_ee_pos(z=0.42)
        self._robot.command_ee_pos(x=x + 0.048, y=y + self._gripper_offset - 0.009)
        self._robot.command_ee_pos(z=0.29)

        current_state = self._robot.get_state()
        current_pose = np.eye(4)
        current_pose[:3, :3] = Quaternion(current_state.ee_quat).rotation_matrix
        current_pose[:3, 3] = current_state.ee_pos
        # dump
        angle = -np.pi * 4 / 9
        rotz = np.array(
            [
                [np.cos(angle), -np.sin(angle), 0, 0],
                [np.sin(angle), np.cos(angle), 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
        )
        rotated_frame_scoop = (
            current_pose
            @ ee_frame_to_scoop_transform
            @ rotz
            @ np.linalg.inv(ee_frame_to_scoop_transform)
        )
        Q = Quaternion(matrix=rotated_frame_scoop[:3, :3])
        self._robot.command_ee_pos(*rotated_frame_scoop[:3, 3], quat=Q.elements)
        # move up and down to shake
        self._robot._speed = save_speed
        self._robot.command_ee_pos(z=0.34)
        self._robot.command_ee_pos(z=0.29)

        # move back
        # return
        self._robot.command_ee_pos(z=0.42)
        self._robot.command_ee_pos(*prior_state.ee_pos, quat=prior_state.ee_quat)
        self._robot.command_ee_pos(*grasp_pos, quat=grasp_quat)
        # open gripper
        self._robot.command_gripper_pos(0)
        self._robot.command_ee_pos(*pre_grasp_pos, quat=pre_grasp_quat)
        self._robot.command_ee_pos(*reset_state.ee_pos, quat=reset_state.ee_quat)

        return True, "Scoop successful"
