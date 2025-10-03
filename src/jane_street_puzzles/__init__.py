# TODO: waiting on <https://github.com/CPMpy/cpmpy/issues/709>
import warnings
import sys
warnings.filterwarnings("ignore", category=UserWarning, message=".*pkg_resources.*")

from jane_street_puzzles.puzzle_2025_09 import main as puzzle_2025_09_main  # noqa: E402

def main() -> None:
    puzzles = {
        "2025-09": puzzle_2025_09_main
    }

    if len(sys.argv) == 2 and sys.argv[1] in puzzles:
        puzzles[sys.argv[1]]()
    else:
        print("jane-street-puzzles <puzzle-id>")
        print("  where <puzzle-id> is one of: " + ", ".join(puzzles.keys()))


if __name__ == "__main__":
    main()
