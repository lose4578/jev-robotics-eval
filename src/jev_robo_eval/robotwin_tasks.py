"""Small RoboTwin task registry. Coordinates are supplied by the adapter."""

from dataclasses import dataclass


@dataclass(frozen=True)
class RoboTwinTask:
    actor: str
    contact_point: int | None
    goal_actor: str | None
    instruction: str
    goal_functional_point: int | None = None


TASKS = {
    "click_bell": RoboTwinTask(
        "bell", 0, None, "Press the top of the bell with the active arm."),
    "press_stapler": RoboTwinTask(
        "stapler", 2, None, "Press the top of the stapler with the active arm."),
    "move_pillbottle_pad": RoboTwinTask(
        "pillbottle", None, "pad", "Pick up the pill bottle and place it upright on the pad."),
    "click_alarmclock": RoboTwinTask(
        "alarm", 0, None, "Press the top button of the alarm clock with the active arm."),
    "place_container_plate": RoboTwinTask(
        "container", None, "plate", "Pick up the cup or bowl and place it on the plate, then release it."),
    "place_object_scale": RoboTwinTask(
        "object", None, "scale", "Pick up the small object and place it on top of the electronic scale, then release it.",
        goal_functional_point=0),
}
