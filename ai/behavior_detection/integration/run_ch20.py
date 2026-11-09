"""CH-20 entrypoint using the shared asynchronous video runner."""

import sys

from .run_video import main as run_video


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    if "--ch20" not in arguments:
        arguments.append("--ch20")
    return run_video(arguments)


if __name__ == "__main__":
    main()
