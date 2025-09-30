# TODO: waiting on <https://github.com/CPMpy/cpmpy/issues/709>
import warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*pkg_resources.*")

from jane_street_puzzles.puzzle_2025_09 import main as puzzle_2025_09_main  # noqa: E402

def main() -> None:
    puzzle_2025_09_main()

if __name__ == "__main__":
    main()
