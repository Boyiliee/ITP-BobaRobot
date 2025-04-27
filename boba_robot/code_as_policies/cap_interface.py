import os
from collections.abc import Callable
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

from boba_robot.cameras.base import CameraDriver, DummyCamera
from boba_robot.code_as_policies.lmp_core import lmp_fgen
from boba_robot.robot_interface import RobotLanguageInterface
from boba_robot.step_executor import StepExecutor
from boba_robot.vision import get_location, getkb

_PROJECT_ROOT: Path = Path(__file__).parent.parent.parent

prompt_f_gen_imports = '''
from typing import Dict, Tuple
import robot

def grasp_cup(x: float, y: float, cup_description: str):
    """Grasp a cup at the given x, y coordinates. Should ONLY be used on cups (not bowls or stack of cups, other objects).

    Args:
        x (float): The x coordinate of the cup.
        y (float): The y coordinate of the cup.
        description (str): The description of the cup to grasp.

    Returns:
        Tuple[bool, str]: Whether the grasp was successful and a message.
    """
    return robot.grasp_cup(x, y, cup_description)

def place_cup(x: float, y: float):
    """Place a cup to the given x, y coordinates. The cup should be grasped first using "grasp" function before being placed.

    Args:
        x (float): The x coordinate of the final location.
        y (float): The y coordinate of the final location.

    Returns:
        Tuple[bool, str]: Whether the place was successful and a message.
    """
    return robot.place_cup(x, y)

def pour(x: float, y: float):
    """Execute a pouring action at the given x, y coordinates.

    The pouring action should only be executed to a location where the cup is exactly known.
    This is achived by firs placing the target cup at a specific location.

    Args:
        x (float): The x coordinate of where to pour.
        y (float): The y coordinate of where to pour.

    Returns:
        Tuple[bool, str]: Whether the pour was successful and a message.
    """
    return robot.pour(x, y)

def scoop_boba_to_location(x: float, y: float):
    """Scoop an boba into the given x, y. The provided x, y coordinate should be the final location of the cup.

    Args:
        x (float): The x coordinate of the object.
        y (float): The y coordinate of the object.

    Returns:
        Tuple[bool, str]: Whether the grasp was successful and a message.
    """
    return robot.scoop_boba_to_location(x, y)

def grasp_empty_cup_from_stack(x, y):
    """Grab an empty cup at the given x, y coordinates from the stack. Should ONLY be used on a cup stack.

    Args:
        x (float): The x coordinate of the object.
        y (float): The y coordinate of the object.

    Returns:
        Tuple[bool, str]: Whether the grasp was successful and a message.
    """
    return robot.grasp_empty_cup_from_stack(x, y)
'''

prompt_f_gen_examples = """
# define function: [example_complete_step0(scene_description)] that "get an working cup and bring to the workspace of the table (0.50, 0.0)"
# info: the completed steps are []
def example_complete_step0(scene_description: Dict[str, Tuple[int, int]]):
    stack_x, stack_y = scene_description["cup stack"]
    grasp_empty_cup_from_stack(stack_x, stack_y)
    place_cup(0.50, 0.0)
    return

# define function: [example_complete_step1] that "add strawberry jam into the working cup"
# info: the completed steps are ["get an working cup and bring to the workspace of the table (0.50, 0.0)"]
def example_complete_step1(scene_description: Dict[str, Tuple[int, int]]):
    # grab the cup with strawberry
    strawberry_x, strawberry_y = scene_description["cup with strawberry"]
    grasp_cup(strawberry_x, strawberry_y, "cup with strawberry")

    # pour the strawberry jam into the working cup
    cup_x, cup_y = scene_description["working cup"]
    pour_cup(cup_x, cup_y)

    # place the cup with strawberry back to its original position
    place_cup(cup_x, cup_y)


# define function: [example_complete_step2(scene_description)] that "add boba into the working cup"
# info: the completed steps are ["get an working cup and bring to the workspace of the table (0.50, 0.0)", "add strawberry jam into the working cup"]
def example_complete_step2(scene_description: Dict[str, Tuple[int, int]]):
    scoop_boba_to_location(0.50, 0.0)
    return


# define function: [example_complete_step3(scene_description)] that "put the cup in the finished location"
# info: the completed steps are ["get an working cup and bring to the workspace of the table (0.50, 0.0)", "add strawberry jam into the working cup", "add boba into the working cup"]
def example_complete_step3(scene_description: Dict[str, Tuple[int, int]]):
    cup_x, cup_y = scene_description["working cup"]
    grasp_cup(cup_x, cup_y, "working cup")
    place_cup(cup_x, cup_y)
    return
""".strip()


# def get_code_block_from_function(functions: List[Callable]) -> str:
#     _functions_str = []
#     for function in functions:
#         source_code = inspect.getsource(function)
#         _functions_str.append(source_code)
#     return "\n\n".join(_functions_str)


class CAPExecutor(StepExecutor):
    def __init__(
        self, robot: List[Callable], scene_keys: List[str], verbose: bool = False
    ):
        self._robot = robot
        self._scene_keys = scene_keys

        scene_description = """# scene_description: Dict[str, Tuple[int, int]], maps object name to (x, y) position
# possible keys are:
#""" + "\n# ".join(
            ['"' + s + '"' for s in scene_keys]
        )

        self._prompt = (
            prompt_f_gen_imports
            + "\n\n"
            + scene_description
            + "\n\n"
            + prompt_f_gen_examples
        )

    def execute(
        self,
        completed_steps: List[str],
        current_step: str,
        scene_description: Dict[str, Tuple[int, int]],
    ) -> str:
        info = f"the completed steps are {completed_steps}"
        context_vars = {"robot": self._robot}
        prompt_f_gen_exec = """
from typing import Dict, Tuple

def grasp_cup(x: float, y: float, cup_description: str):
    return robot.grasp_cup(x, y, cup_description)

def place_cup(x: float, y: float):
    return robot.place_cup(x, y)

def pour(x: float, y: float):
    return robot.pour(x, y)

def scoop_boba_to_location(x: float, y: float):
    return robot.scoop_boba_to_location(x, y)

def grasp_empty_cup_from_stack(x, y):
    return robot.grasp_empty_cup_from_stack(x, y)
""".strip()
        exec(prompt_f_gen_exec, context_vars)
        print("generate function, completed steps are", completed_steps)
        function_name = f"complete_step{len(completed_steps)}"
        policy = lmp_fgen(
            self._prompt,
            function_name,
            f'[{function_name}(scene_description)] that "{current_step}"',
            recurse=False,
            context_vars=context_vars,
            info=info,
        )
        policy(scene_description)
        return "success"


coordinator_prompt_f_gen_examples = """
# define function: [example_complete_request(scene_description)] that satisfies the user request "i want a milk"
def example_complete_request(scene_description: Dict[str, Tuple[int, int]]):
    # get an working cup and bring to the workspace of the table (0.50, 0.0)
    stack_x, stack_y = scene_description["cup stack"]
    grasp_empty_cup_from_stack(stack_x, stack_y)
    place_cup(0.50, 0.0)

    # add milk into the working cup
    milk_x, milk_y = scene_description["cup with milk"]
    grasp_cup(milk_x, milk_y, "cup with milk")
    pour(0.5, 0)
    place_cup(milk_x, milk_y)

    # put the working cup in the finished location
    grasp_cup(0.5, 0, "working cup")
    final_x, final_y = scene_description["finished location"]
    place_cup(final_x, final_y)
    return
""".strip()



class CAPCoordinator:
    def __init__(
        self,
        robot_lang_interface: RobotLanguageInterface,
        camera: CameraDriver,
        user_guildlines_file: str = "simple_user_guidelines.txt",
    ):
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._robot = robot_lang_interface
        self._camera = camera
        with open(os.path.join(_PROJECT_ROOT, "text", user_guildlines_file)) as f:
            self._user_guidelines = f.read()
        self.object_detection_model = None
        self._scene = None
        x = [60, 50, 35]
        y = [10, 0, -10]
        xi = [256, 316, 376]
        yi = [230, 282, 372]
        self.kx, self.bx = getkb(x, yi)
        self.ky, self.by = getkb(y, xi)

    def image_to_scene_description(self) -> str:
        if self._scene is not None:
            self._scene["empty cup"] = (0.5, 0.0)
            return self._scene
        if isinstance(self._camera, DummyCamera):
            scene = {
                "cup with milk": (0.667, -0.266),
                "bowl with boba": (0.535, -0.47),
                "cup with taro": (0.23, -0.46),
                "cup with strawberry": (0.37, -0.865),
                "cup stack": (0.4, 0.0),
                "finished location": (0.4, 0.5),
                "trash_location": (0.6, -0.4),
            }
            return scene

        self._robot.set_ee(z=0.15)
        current_joints = self._robot.get_qpos()
        cam_capture_joints = np.deg2rad(np.array([38, -11, 15, 6, -187, 70, +175]))
        self._robot.set_qpos(cam_capture_joints)
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
        scene["finished location"] = (0.4, 0.5)
        scene["trash location"] = (0.6, -0.4)
        self._robot.set_qpos(current_joints)
        # convert to scene description
        self._scene = scene
        return scene

    def execute_request(self, user_prompt: str):
        scene = self.image_to_scene_description()

        scene_str = (
            "{\n" + ",\n".join([f'    "{k}": {v}' for k, v in scene.items()]) + "\n}"
        )
        scene_description = (
            """# scene_description: Dict[str, Tuple[int, int]], maps object name to (x, y) position
scene_description = """
            + scene_str
        )

        user_guildlines_prompt = (
            "# user_guildlines: str, these are the user guildines\n"
            + f'user_guildlines = """{self._user_guidelines}"""'
        )
        user_guildlines_prompt += "\n# the code should make sure that the user_guildlines are followed\n# no new robot functions should be used of defined. use the defined primatives.\n"

        prompt = (
            prompt_f_gen_imports
            + "\n\n"
            + user_guildlines_prompt
            + "\n\n"
            + scene_description
            + "\n\n"
            + coordinator_prompt_f_gen_examples
        )
        print()
        print("*" * 80)
        print("Code As Policy Prompt:")
        print(prompt)
        print("*" * 80)
        print()

        context_vars = {"robot": self._robot}
        prompt_f_gen_exec = """
from typing import Dict, Tuple

def grasp_cup(x: float, y: float, cup_description: str):
    return robot.grasp_cup(x, y, cup_description)

def place_cup(x: float, y: float):
    return robot.place_cup(x, y)

def pour(x: float, y: float):
    return robot.pour(x, y)

def scoop_boba_to_location(x: float, y: float):
    return robot.scoop_boba_to_location(x, y)

def grasp_empty_cup_from_stack(x, y):
    return robot.grasp_empty_cup_from_stack(x, y)
""".strip()
        exec(prompt_f_gen_exec, context_vars)
        function_name = "complete_request"
        policy = lmp_fgen(
            prompt,
            function_name,
            f'[{function_name}(scene_description)] that satisfies the user request "{user_prompt}"',
            recurse=False,
            context_vars=context_vars,
        )
        policy(scene)
        return
