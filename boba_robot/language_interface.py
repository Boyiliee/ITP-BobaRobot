import datetime
import json
import logging
import os
import re
import select
import sys
import time
from logging import Logger
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import openai
import torch

from boba_robot.cameras.base import CameraDriver, DummyCamera
from boba_robot.code_as_policies.cap_interface import CAPExecutor
from boba_robot.robot_interface import RobotLanguageInterface
from boba_robot.step_executor import StepExecutor
from boba_robot.utils import GPTFunctionAssistant, print_color
from boba_robot.vision import SCENE_KEYS, get_location, getkb

_PROJECT_ROOT: Path = Path(__file__).parent.parent


def section_str(s: str, section: str, indent: int = 0) -> str:
    # section at the start and end of the string with indent
    s = "\n".join([f"{' ' * indent}{line}" for line in s.split("\n")])
    return f"{' ' * indent}{'#' * 150}\n{' ' * indent}{section}\n{' ' * indent}{'#' * 150}\n{s}\n{' ' * indent}{'#' * 150}\n"


def init_logger(
    file_name: Optional[str] = None, logger_name="Logger", format: bool = False
) -> Logger:
    if file_name is None:
        now = datetime.datetime.now()
        current_time = now.strftime("%Y-%m-%d-%H-%M-%S")
        file_name = f"log_{current_time}.txt"

    # create logger
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)

    # create file handler
    fh = logging.FileHandler(file_name)
    fh.setLevel(logging.DEBUG)

    # create formatter
    if format:
        formatter = logging.Formatter(
            "%(asctime)-24s %(name)-12s %(levelname)-8s >>>  %(message)s"
        )
    else:
        formatter = logging.Formatter("%(message)s")
    fh.setFormatter(formatter)

    # add file handler to logger
    logger.addHandler(fh)
    return logger


def format_message(message: str, role: str = "user") -> Dict[str, str]:
    return {
        "role": role,
        "content": message,
    }


# create formulation


class GPTFuncitonExecutor(StepExecutor):
    def __init__(
        self,
        functions: List[StepExecutor],
        model: str,
        verbose: bool = False,
        logger: Optional[Logger] = None,
    ):
        self._assistant = GPTFunctionAssistant(
            functions,
            model=model,
            verbose=verbose,
            logger=logger,
        )

    def _format_step_message(
        self, completed_steps: List[str], step: str, scene_description: str
    ) -> str:
        return f"""So far the robot has completed these steps: {completed_steps}. The next step is to {step}.
The current scene looks like this: {scene_description}
Can you use the robot functions to complete: {step}?
After completion, respond with a summary of the execution. Make sure to put things back after using them :)
"""

    def execute(
        self,
        completed_steps: List[str],
        current_step: str,
        scene_description: Dict[str, Tuple[int, int]],
    ) -> str:
        step_request = self._format_step_message(
            completed_steps, current_step, json.dumps(scene_description)
        )
        response, _ = self._assistant.chat(step_request)
        return response["content"]


class LanguageCoordinator:
    """Takes the output of the language plan and outputs a robot plan."""

    def __init__(
        self,
        robot_lang_interface: RobotLanguageInterface,
        camera: CameraDriver,
        # model: str = "gpt-3.5-turbo-0613",
        model: str = "gpt-4-0613",
        verbose: bool = False,
        user_guildlines_file: str = "simple_user_guidelines.txt",
        replan_timeout: float = 3.0,
        replan_config: Optional[Tuple[int, str]] = None,
        executor_type: str = "default",
        name: Optional[str] = None,
    ):
        with open(os.path.join(_PROJECT_ROOT, "text", user_guildlines_file)) as f:
            self._user_guidelines = f.read()
        self._camera = camera
        self._robot_lang_interface = robot_lang_interface
        self._model = model
        self._verbose = verbose
        self._replan_timeout = replan_timeout
        self._scene = None
        self._executor_type = executor_type
        self._name = name

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self.object_detection_model = None
        x = [60, 50, 35]
        y = [10, 0, -10]
        xi = [256, 316, 376]
        yi = [230, 282, 372]
        self.kx, self.bx = getkb(x, yi)
        self.ky, self.by = getkb(y, xi)
        self._replan_config = replan_config

    def log(self, message: str):
        if self._verbose:
            print(message)

    def get_plan(self, user_prompt: str) -> Optional[List[str]]:
        """Get a plan from the language model given a user prompt.

        Args:
            user_prompt: the user prompt

        Returns:
            plan: the plan, if it exists, which is a list of steps (strings). if the plan does not exist, return None.
        """
        date_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        if self._replan_config is None:
            file_name = f"logs/{date_time}_{user_prompt[:20].replace(' ', '_')}.txt"
        else:
            file_name = f"logs/{date_time}_{user_prompt[:20].replace(' ', '_')}_{self._replan_config[0]}_{self._replan_config[1][:20].replace(' ', '_')}.txt"
        if self._name is not None:
            file_name = file_name.replace(".txt", f"_{self._name}.txt")

        os.makedirs(os.path.dirname(file_name), exist_ok=True)
        self._logger = init_logger(file_name, format=False)

        message_list = []
        # user_prompt = "I want a boba mango milk"
        # prompt = f"""The user has requested:
        # ---
        # {user_prompt}
        # ---
        # The the guidelines are:
        # ---
        # {self._user_guidelines}
        # ---
        # Can you give me a step by step plan to complete the user's request?
        # The format of the response should be a list of steps, where each step is in a new line. The steps should exactly match one of the guidelines. Respond with only the steps"""
        prompt = f"""The user has requested:
---
{user_prompt}
---
The the guidelines are:
---
{self._user_guidelines}
---
Set A = all the materials we have now.
Set B = all the materials we need.
Print Set A in the first line.
Print Set B in the second line.
Print Set C in the third line where Set C are the items in Set B that are not in Set A.
If Set C is not empty, provide unique element and respond with "Set C is not empty";
else, respond with a numbered list of steps, where each step is in a new line (the steps should closely match one of the guidelines).
"""
        self._logger.info(section_str(prompt, "Prompt"))

        message_list.append(format_message(prompt))
        self.log(f"Calling OpenAI with message list:\n{message_list}\n")
        response = openai.ChatCompletion.create(
            model=self._model,
            messages=message_list,
            temperature=0.0,
        )
        response_message = response["choices"][0]["message"]
        self.log(f"OpenAI response:\n{response_message}\n")

        content = response_message["content"]
        self._logger.info(section_str(content, "Step Response"))
        message_list.append(response_message.to_dict_recursive())
        self._message_list = message_list

        if "not empty" in content:
            return None
#
        # content = content.split("\n\n")[-1]
        _steps = content.split("\n")
        # check that the first character of each step is a number
        steps = []
        for step in _steps:
            if not re.match(r"^\d+\)", step):
                pass
                # print_color(
                #     f"Invalid step: {step}. Steps should start with a number followed by a parenthesis.",
                #     color="red",
                # )
            else:
                print(step)
                steps.append(step)

        # strip the number from the beginning of each step
        # steps = [re.sub(r"^\d+\)", "", step).strip() for step in steps]
        return steps

    def execute(
        self, plan: List[str], completed_steps: Optional[List[str]] = None
    ) -> bool:
        self.log("Executing plan:")

        if completed_steps is None:
            completed_steps: List[str] = []
        for idx, step in enumerate(plan):
            self.log("-" * 150 + f"\n\tPlanning Step {idx}: {step}")

            if self._executor_type == "default":
                executor = GPTFuncitonExecutor(
                    functions=[
                        self._robot_lang_interface.grasp_cup,
                        self._robot_lang_interface.place_cup,
                        self._robot_lang_interface.pour,
                        self._robot_lang_interface.scoop_boba_to_location,
                        self._robot_lang_interface.grasp_empty_cup_from_stack,
                    ],
                    model=self._model,
                    verbose=self._verbose,
                    logger=self._logger,
                )
            elif self._executor_type == "code_as_policies":
                executor = CAPExecutor(
                    self._robot_lang_interface,
                    scene_keys=SCENE_KEYS,
                    verbose=self._verbose,
                )
            else:
                raise NotImplementedError

            scene_description = self.image_to_scene_description()
            self.log(f"\tScene Description: {scene_description}\n")

            self._logger.info(scene_description)
            response = executor.execute(completed_steps, step, scene_description)
            completed_steps.append(f"{step}: {response}")
            self.log(f"\tResponse: {response}\n")
            new_plan = self.replan(completed_steps)
            if new_plan is not None:
                return self.execute(new_plan, completed_steps)

        return True

    def image_to_scene_description(self) -> str:
        if self._scene is not None:
            self._scene["working cup"] = (0.55, 0.0)
            return self._scene
        if isinstance(self._camera, DummyCamera):
            scene = {
                "cup with milk": (0.667, -0.266),
                "bowl with boba": (0.535, -0.47),
                "cup with taro": (0.23, -0.46),
                "cup with strawberry": (0.37, -0.865),
                "cup stack": (0.4, 0.0),
                "finished location": (0.15, 0.5),
                "trash_location": (0.6, -0.4),
            }
            return scene

        self._robot_lang_interface.set_ee(z=0.15)
        current_joints = self._robot_lang_interface.get_qpos()
        cam_capture_joints = np.deg2rad(np.array([38, -11, 15, 6, -187, 70, +175]))
        self._robot_lang_interface.set_qpos(cam_capture_joints)
        (
            rgb_image,
            depth_monocular,
        ) = self._camera.read()  # shape: (480, 640, 3), (480, 640, 1)

        if self.object_detection_model is None:
            from boba_robot.grounded_dino import get_ovod_model

            self.object_detection_model = get_ovod_model(self._device)

        scene = get_location(
            rgb_image,
            self.object_detection_model,
            self.kx,
            self.bx,
            self.ky,
            self.by,
            show_and_wait_for_confirm=True,
        )
        scene["finished location"] = (0.15, 0.5)
        scene["trash location"] = (0.6, -0.4)
        self._robot_lang_interface.set_qpos(current_joints)
        # convert to scene description
        self._scene = scene
        return scene

    def _format_replan_message(
        self, completed_steps: List[str], new_request: str
    ) -> str:
        return f"""So far the robot has completed these steps: {completed_steps}.

The user has requested some feedback now:
---
{new_request}
---
If the user wants to add something, directly add one step and keep the original steps unchanged.
For other requests, can you first print a summary of the current users request after their feedback? Then like before
Set A = all the materials we have now.
Set B = all the materials we need.
Print Set A in the first line.
Print Set B in the second line.
Print Set C in the third line where Set C are the items in Set B that are not in Set A.
If Set C is not empty, provide unique element and respond with "Set C is not empty";
else, respond with a new numbered list of steps, where each step is in a new line (the steps should closely match one of the guidelines).
steps already completed should be excluded from the new list. If you need to start from scratch, then put the existing cup in the trash location and get a new empty cup."""

    def replan(self, completed_steps: List[str]) -> Optional[List[str]]:
        """Replan the remaining steps.

        Args:
            completed_steps: the steps that have been completed

        Returns:
            new_plan: the new plan, if it exists
        """
        if self._replan_config is None:
            str_input = input_with_timeout(self._replan_timeout)
        else:
            if self._replan_config[0] == len(completed_steps):
                str_input = self._replan_config[1]
            else:
                str_input = None

        if str_input is not None:
            new_request = str_input
            # task current plan, completed steps, and the new request. ask LLM if it can complete the task still or if it needs start from scratch
            # ...
            # new plan
            new_request = self._format_replan_message(completed_steps, new_request)
            self._message_list.append(format_message(new_request))
            self._logger.info(section_str(new_request, "Replan Request"))
            self.log(f"Calling OpenAI with message list:\n{self._message_list}\n")

            response = openai.ChatCompletion.create(
                model=self._model,
                messages=self._message_list,
                temperature=0.0,
            )

            response_message = response["choices"][0]["message"]
            self._message_list.append(response_message.to_dict_recursive())
            self.log(f"OpenAI response:\n{response_message}\n")
            content = response_message["content"]
            self._logger.info(section_str(content, "Replan Response"))

            if "not empty" in content:
                raise RuntimeError(f"The user's request cannot be completed: {content}")

            # content = content.split("\n\n")[-1]
            _steps = content.split("\n")
            # check that the first character of each step is a number
            steps = []
            for step in _steps:
                if not re.match(r"^\d+\)", step):
                    pass
                    # print_color(
                    #     f"Invalid step: {step}. Steps should start with a number followed by a parenthesis.",
                    #     color="red",
                    # )
                else:
                    print(step)
                    steps.append(step)
            return steps
        else:
            return None


def input_with_timeout(timeout: float) -> Optional[str]:
    if timeout <= 0:
        return None
    prompt = "Press enter to provide feedback 🎤:"
    print_color(prompt, color="yellow", end="", flush=True)

    start = time.time()
    rem_time = timeout

    while rem_time > 0:
        ready, _, _ = select.select([sys.stdin], [], [], 0.1)
        rem_time = timeout - (time.time() - start)
        print_color(
            f"\r{prompt} | ⏰ Time left: {rem_time:.2} sec",
            color="yellow",
            attrs=("bold",),
            end="",
            flush=True,
        )
        if ready:
            _ = input()
            return input("\nEnter Feedback:\n")

    print_color("\nTime's up! You didn't enter any feedback.", color="yellow")
    return None


if __name__ == "__main__":
    res = input_with_timeout(5)
    print(res)
