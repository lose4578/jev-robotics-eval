"""Small RoboTwin task registry. Coordinates are supplied by the adapter."""

from dataclasses import dataclass


@dataclass(frozen=True)
class RoboTwinTask:
    actor: str
    contact_point: int | None
    goal_actor: str | None
    instruction: str


TASKS = {
    "click_bell": RoboTwinTask(
        "bell", 0, None, "Press the top of the bell with the active arm."),
    "press_stapler": RoboTwinTask(
        "stapler", 2, None, "Press the top of the stapler with the active arm."),
    "move_pillbottle_pad": RoboTwinTask(
        "pillbottle", None, "pad", "Pick up the pill bottle and place it upright on the pad."),
}
