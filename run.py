from dataclasses import dataclass
from typing import Optional

import numpy as np
import tyro

from boba_robot.cameras.base import DummyCamera, SavedCamera
from boba_robot.code_as_policies.cap_interface import CAPCoordinator
from boba_robot.language_interface import LanguageCoordinator
from boba_robot.robot_interface import RobotLanguageInterface
from boba_robot.utils import print_color


@dataclass
class Args:
    user_prompt: Optional[str] = None
    real_robot: bool = False
    real_camera: bool = False
    use_feedback_policy: bool = False
    verbose: bool = True
    speed: float = 250
    executor_type: str = "default"
    coordinator_type: str = "default"

    # used for evaluation and debugging purposes only
    interupt_step: Optional[int] = None
    interupt_instruction: Optional[str] = None
    name: Optional[str] = None


def main(args: Args):
    if args.interupt_step is None:
        replan_config = None
    else:
        assert args.interupt_instruction is not None
        replan_config = (args.interupt_step, args.interupt_instruction)

    if args.real_robot:
        from boba_robot.robots.xarm_robot import XArmDriver

        reset_joints = np.array(
            [
                -0.112512,
                0.377531,
                0.128068,
                1.880434,
                -3.040666,
                0.06573,
                -3.289401 + 2 * np.pi,
            ]
        )

        start_joints = np.array(
            [
                -0.409026,
                0.013359,
                0.304852,
                0.338141,
                -3.257623,
                1.191721,
                -3.104624 + 2 * np.pi,
            ]
        )
        _driver = XArmDriver("192.168.1.233", speed=args.speed)
        # _driver.command_ee_pos(z=0.4)
        _driver.command_ee_pos(0.6, 0, 0.5, quat=np.array([0, 1, 0, 1]))
        _driver.command_qpos(reset_joints)
        _driver.command_ee_pos(0.3, 0, 0.3, quat=np.array([0, 1, 0, 1]))
        _driver.command_qpos(start_joints)
        robot_language_interface = RobotLanguageInterface(
            _driver, use_feedback_policy=args.use_feedback_policy
        )
    else:
        robot_language_interface = RobotLanguageInterface(None)

    if args.real_camera:
        try:
            from boba_robot.cameras.realsense import RealSenseCamera, get_device_ids

            device_ids = get_device_ids()
            top_camera_id = "941322071164"
            assert top_camera_id in device_ids
            camera = RealSenseCamera(flip=True, device_id=top_camera_id)
        except Exception as e:
            print(e)
            print(
                "Encountered exception when using RealSenseCamera, using saved camera"
            )
            camera = SavedCamera()
    else:
        camera = DummyCamera()

    if args.user_prompt is None:
        user_prompt = input("Enter your request: ")
    else:
        user_prompt = args.user_prompt

    if args.coordinator_type == "default":
        coordinator = LanguageCoordinator(
            robot_lang_interface=robot_language_interface,
            camera=camera,
            verbose=args.verbose,
            executor_type=args.executor_type,
            replan_timeout=3,
            replan_config=replan_config,
            name=args.name,
        )

        plan = coordinator.get_plan(user_prompt)
        if plan is None:
            print_color(
                "Failed to get plan, give a different user_prompt",
                color="red",
                attrs=("bold",),
            )
            return
        coordinator.execute(plan)
    elif args.coordinator_type == "code_as_policies":
        coordinator = CAPCoordinator(
            robot_lang_interface=robot_language_interface,
            camera=camera,
        )
        coordinator.execute_request(user_prompt)
    else:
        raise NotImplementedError(f"coordinator_type {args.coordinator_type} not found")

    print_color("Completed! ✅", color="green", attrs=("bold",))


if __name__ == "__main__":
    main(tyro.cli(Args))
