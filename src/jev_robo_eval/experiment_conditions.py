"""Shared, explicit experimental conditions for all evaluation entry points."""

from .observation_access import validate_plan_only


CONDITION_FIELDS = (
    "plan_only", "stuck_recovery", "recovery_seed", "recovery_window",
    "recovery_displacement_m", "recovery_cooldown", "recovery_max_interventions",
    "recovery_steps", "action_granularity",
)


def add_condition_arguments(parser):
    parser.add_argument("--plan-only", action="store_true",
                        help="L3-P: keep oracle waypoints but remove extra scene truth")
    parser.add_argument("--stuck-recovery", choices=("none", "jitter"), default="none")
    parser.add_argument("--recovery-seed", type=int, default=1000,
                        help="Independent recovery RNG base seed; episode index is added")
    parser.add_argument("--recovery-window", type=int, default=6)
    parser.add_argument("--recovery-displacement-m", type=float, default=0.003)
    parser.add_argument("--recovery-cooldown", type=int, default=6)
    parser.add_argument("--recovery-max-interventions", type=int, default=3)
    parser.add_argument("--recovery-steps", type=int, choices=(1, 2), default=2)
    parser.add_argument("--action-granularity", choices=("fixed", "phase-fixed", "adaptive"), default="fixed",
                        help="Adaptive: JEV chooses phase-specific motion direction and step size")


def condition_config(args):
    return {key: getattr(args, key) for key in CONDITION_FIELDS}


def recovery_for_episode(args, episode_index=1):
    if args.stuck_recovery == "none":
        return None
    from .recovery import StuckRecovery
    return StuckRecovery(
        seed=args.recovery_seed + episode_index - 1,
        window=args.recovery_window,
        displacement_threshold_m=args.recovery_displacement_m,
        cooldown=args.recovery_cooldown,
        max_interventions=args.recovery_max_interventions,
        perturb_steps=args.recovery_steps,
    )


def validate_condition_arguments(args):
    validate_plan_only(args.privilege_level, args.plan_only)
    if args.action_granularity in {"phase-fixed", "adaptive"} and (
            args.mode != "vision" or args.sensor_policy != "staged"
            or args.action_space != "primitive" or args.privilege_level == 3):
        raise ValueError("Adaptive granularity requires vision, staged, primitive, and L0/L1/L2")
    # Validate even inactive settings: saved configs must be usable verbatim.
    from .recovery import StuckRecovery
    StuckRecovery(seed=args.recovery_seed, window=args.recovery_window,
                  displacement_threshold_m=args.recovery_displacement_m,
                  cooldown=args.recovery_cooldown,
                  max_interventions=args.recovery_max_interventions,
                  perturb_steps=args.recovery_steps)


def condition_cli_arguments(args):
    result = ["--plan-only"] if args.plan_only else []
    for key in CONDITION_FIELDS[1:]:
        result.extend(["--" + key.replace("_", "-"), str(getattr(args, key))])
    return result
