import json
from inspect import Parameter, getdoc, signature
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
    get_args,
    get_origin,
)

import openai


def section_str(s: str, section: str, indent: int = 0) -> str:
    # section at the start and end of the string with indent
    s = "\n".join([f"{' ' * indent}{line}" for line in s.split("\n")])
    return f"{' ' * indent}{'#' * 150}\n{' ' * indent}{section}\n{' ' * indent}{'#' * 150}\n{s}\n{' ' * indent}{'#' * 150}\n"


def print_color(*args, color, attrs=(), **kwargs):
    import termcolor

    if len(args) > 0:
        args = tuple(termcolor.colored(arg, color=color, attrs=attrs) for arg in args)
    print(*args, **kwargs)


def python_type_to_json_type(parameter: Parameter) -> Dict[str, Any]:
    """Convert a Python type to a JSON type.

    For example, the Python type `str` should be converted to the JSON type
    `string`.

    List[str] should be converted to the JSON type array of strings. This would look like
    ```
    {
        "type": "array",
        "items": {
            "type": "string",
        }
    }

    Tuple[int, str] should be converted to the JSON type array with two elements, the first
    of which is an integer and the second of which is a string. This would look like
    ```
    {
        "type": "array",
        "items": [
            {
                "type": "integer",
            },
            {
                "type": "string",
            }
        ]
    }

    This works recursively, so List[Tuple[int, str]] should be converted to
    ```
    {
        "type": "array",
        "items": {
            "type": "array",
            "items": [
                {
                    "type": "integer",
                },
                {
                    "type": "string",
                }
            ]
        }
    }

    Args:
        parameter: The parameter to convert.

    Returns:
        A dictionary containing the JSON type.
    """

    def convert_type(python_type: Any) -> Dict[str, Any]:
        origin = get_origin(python_type)
        type_candidate = python_type

        if origin is None:
            if type_candidate == str:
                return {"type": "string"}
            elif type_candidate in (int, float):  # noqa: E721
                return {"type": "number"}
            elif type_candidate == bool:
                return {"type": "boolean"}
            elif type_candidate == type(None):  # noqa: E721
                return {"type": "null"}
            else:
                raise ValueError(
                    f"Unsupported type {type_candidate} for parameter {parameter.name}"
                )
        elif origin in (list, Sequence):  # noqa: E721
            item_type = get_args(python_type)[0]
            return {"type": "array", "items": convert_type(item_type)}
        elif origin == tuple:
            item_types = get_args(python_type)
            return {
                "type": "array",
                "items": [convert_type(item_type) for item_type in item_types],
            }
        elif origin == Union:
            union_types = get_args(python_type)
            return {"anyOf": [convert_type(t) for t in union_types]}
        else:
            raise ValueError(
                f"Unsupported type {type_candidate} for parameter {parameter.name} with origin {origin}"
            )

    assert parameter.annotation is not Parameter.empty
    json_type = convert_type(parameter.annotation)
    return json_type


def extract_api_from_docstring(
    function: Callable,
) -> Dict[str, Any]:
    """Extract the API from a function's docstring.

    For example, the resulting API foro this function
    ```
    def get_current_weather(location: str, use_celsius: bool = False, verbose: bool = False):
        '''Get the current weather in a given location.

        Args:
            location (str): The city and state, e.g. San Francisco, CA
            use_celsius: (bool) Whether to return the temperature in celsius or fahrenheit.

        Returns:
            A JSON string containing the current weather information.
        '''
    ```
    Should be
    {
        "name": "get_current_weather",
        "description": "Get the current weather in a given location.",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city and state, e.g. San Francisco, CA",
                },
                "unit": {
                    "type": "bool",
                    "description": "Whether to return the temperature in celsius or fahrenheit.",
                    "default": False,
                },
                "verbose": {
                    "type": "bool",
                    "default": False,
                }
            },
            "required": ["location"],
        },
    }
    """  # noqa: D214
    docstring = getdoc(function)
    sig = signature(function)

    if docstring is None:
        raise ValueError(f"Function {function.__name__} does not have a docstring.")

    lines = docstring.split("\n")

    arg_descs = {}
    if "Args:" not in lines and len(sig.parameters) > 0:
        raise ValueError("The function docstring must contain an 'Args:' section")
    elif "Args:" not in lines and len(sig.parameters) == 0:
        if "Returns:" in lines:
            desc_end = lines.index("Returns:")
        else:
            desc_end = len(lines)
        desc = " ".join(line.strip() for line in lines[0:desc_end]).strip()
    else:
        desc_end = lines.index("Args:")
        desc = " ".join(line.strip() for line in lines[0:desc_end]).strip()

        args_start = desc_end + 1
        for line in lines[args_start:]:
            if line.strip() == "":
                continue
            arg_name, arg_desc = [s.strip() for s in line.split(":", 1)]
            arg_name = arg_name.split(" ")[0]  # remove type annotation if present
            arg_descs[arg_name] = arg_desc.strip()

    required_arguments = []
    arguments = {}
    for parameter in sig.parameters.values():
        if parameter.annotation is Parameter.empty:
            raise ValueError(
                f"Function {function.__name__} has an argument {parameter.name} "
                "without a type annotation."
            )

        arguments[parameter.name] = python_type_to_json_type(parameter)
        if parameter.name in arg_descs:
            arguments[parameter.name]["description"] = arg_descs[parameter.name]

        if parameter.default is Parameter.empty:
            required_arguments.append(parameter.name)
        else:
            arguments[parameter.name]["default"] = parameter.default

    return {
        "name": function.__name__,
        "description": desc,
        "parameters": {
            "type": "object",
            "properties": arguments,
            "required": required_arguments,
        },
    }


class GPTFunctionAssistant:
    def __init__(
        self,
        functions: List[Callable],
        model: str = "gpt-4-0613",
        max_interactions: int = 10,
        verbose: bool = False,
        logger: Optional[Any] = None,
    ):
        self.max_interactions = max_interactions
        self.verbose = verbose
        self.functions = functions
        self.function_apis = [
            extract_api_from_docstring(function) for function in functions
        ]
        self.function_names = [function.__name__ for function in functions]
        self.model = model

        self.message_list: List[Dict[str, Any]] = []
        self._logger = logger

    def reset(self):
        """Reset the assistant's state."""
        self.message_list = []

    def _handle_function_call(self, message) -> Tuple[str, List[Dict]]:
        if len(self.message_list) > self.max_interactions:
            return "I'm tired. Let's talk later.", self.message_list

        self._logger.info(
            section_str(str(message["function_call"]), "response", indent=8)
        )
        function_name = message["function_call"]["name"]
        if function_name not in self.function_names:
            return (
                f"I don't know how to do that. I only know how to do {', '.join(self.function_names)}",
                self.message_list,
            )

        function_index = self.function_names.index(function_name)
        function = self.functions[function_index]

        args = message["function_call"]["arguments"]
        args_json = json.loads(args)

        kwargs = {}
        for key, value in args_json.items():
            kwargs[key] = value

        function_response = function(**kwargs)
        function_response = str(function_response)  # TODO: handle non-string responses

        if self.verbose:
            print("Assistant:", message["function_call"], "\n")
        self.message_list.append(message.to_dict_recursive())
        self.message_list.append(
            {"role": "function", "content": function_response, "name": function_name}
        )
        if self.verbose:
            print("Function:", function_response, "\n")
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=self.message_list,
            functions=self.function_apis,
            function_call="auto",
        )
        response_message = response["choices"][0]["message"]

        if response_message["content"] is not None:
            self._logger.info(
                section_str(str(response_message["content"]), "response", indent=8)
            )

        if response_message.get("function_call"):
            return self._handle_function_call(response_message)
        else:
            if self.verbose:
                print("Assistant:", response_message["content"], "\n")
            self.message_list.append(response_message.to_dict_recursive())
            return response["choices"][0]["message"], self.message_list

    def chat(self, message: str) -> Tuple[str, List[Dict]]:
        """Chat with the assistant.

        Args:
            message (str): The message to send to the assistant.

        Returns:
            Tuple[str, List[Dict]]: The assistant's response and the full message history.
        """
        if len(self.message_list) > self.max_interactions:
            return "I'm tired. Let's talk later.", self.message_list

        if self.verbose:
            print("User:", message, "\n")

        self._logger.info(section_str(message, "user", indent=8))
        self.message_list.append({"role": "user", "content": message})
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=self.message_list,
            functions=self.function_apis,
            function_call="auto",
        )
        response_message = response["choices"][0]["message"]
        if response_message["content"] is not None:
            self._logger.info(
                section_str(str(response_message["content"]), "response", indent=8)
            )

        if response_message.get("function_call"):
            return self._handle_function_call(response_message)
        else:
            if self.verbose:
                print("Assistant:", response_message["content"], "\n")
            self.message_list.append(response_message.to_dict_recursive())
            return response_message, self.message_list


def main():
    print_color("Hello world!", "testing", color="red", attrs=("bold",))
    print_color("Hello world!", color="magenta")


if __name__ == "__main__":
    main()
