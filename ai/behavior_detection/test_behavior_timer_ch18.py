
from types import SimpleNamespace

from behavior_timer import BehaviorTimer


def make_state(status, behavior=None):
    return SimpleNamespace(
        status=status,
        behavior=behavior
    )


def run_test():

    timer = BehaviorTimer(warning_seconds=5.0)

    print("=== HTTM CH-18 TIMER TEST ===")

    # Track 1: Confirmed using_phone at t=0
    states = {
        1: make_state("CONFIRMED", "using_phone")
    }

    for now in [0.0, 1.0, 3.0, 5.0, 6.0]:

        timers, warnings = timer.update(
            states,
            now=now
        )

        current = timers.get(1)

        print(
            f"t={now:.1f}s | "
            f"Track 1 | "
            f"Elapsed={current.elapsed:.1f}s | "
            f"Warned={current.warned}"
        )

        for warning in warnings:
            print(
                "WARNING:",
                warning
            )

    # Reset when behavior becomes NORMAL
    states = {
        1: make_state("NORMAL")
    }

    timers, warnings = timer.update(
        states,
        now=7.0
    )

    print(
        "\nAfter NORMAL:",
        "Timers =", len(timers)
    )

    # New confirmed behavior starts new timer
    states = {
        1: make_state("CONFIRMED", "sleeping")
    }

    timers, warnings = timer.update(
        states,
        now=8.0
    )

    print(
        "New behavior:",
        timers[1].behavior,
        "| Elapsed:",
        timers[1].elapsed
    )

    print("\nCH-18 test finished")


if __name__ == "__main__":
    run_test()
