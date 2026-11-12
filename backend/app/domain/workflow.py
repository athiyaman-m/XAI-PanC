from backend.app.domain.enums import WORKFLOW_ORDER, WorkflowStep


def step_index(step: WorkflowStep) -> int:
    return WORKFLOW_ORDER.index(step)


def next_step(current: WorkflowStep) -> WorkflowStep | None:
    idx = step_index(current)
    if idx + 1 >= len(WORKFLOW_ORDER):
        return None
    return WORKFLOW_ORDER[idx + 1]


def previous_step(current: WorkflowStep) -> WorkflowStep | None:
    idx = step_index(current)
    if idx <= 0:
        return None
    return WORKFLOW_ORDER[idx - 1]


def is_step_accessible(target: WorkflowStep, current: WorkflowStep, completed: set[WorkflowStep]) -> bool:
    """A step is accessible if it is the current step or already completed."""
    target_idx = step_index(target)
    current_idx = step_index(current)
    if target in completed:
        return True
    return target_idx <= current_idx


def can_advance(current: WorkflowStep, completed: set[WorkflowStep]) -> bool:
    return current not in completed


def advance_workflow(current: WorkflowStep, completed: set[WorkflowStep]) -> tuple[WorkflowStep, set[WorkflowStep]]:
    if not can_advance(current, completed):
        raise ValueError(f"Step {current} is already completed")
    completed = set(completed)
    completed.add(current)
    nxt = next_step(current)
    new_current = nxt if nxt is not None else current
    return new_current, completed
