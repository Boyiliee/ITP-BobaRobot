from typing import Dict, List, Optional, Protocol, Tuple

class StepExecutor(Protocol):
    def execute(
        self,
        completed_steps: List[str],
        current_step: str,
        scene_description: Dict[str, Tuple[int, int]],
    ) -> str:
        raise NotImplementedError
        ...
