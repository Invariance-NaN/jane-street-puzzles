import cpmpy as cp
import math
import numpy as np
import rich
from jane_street_puzzles.utils.grids import BooleanGrid, IntGrid
from jane_street_puzzles.utils.polyominos import PENTOMINOS

class Hook:
    """
    An L-shaped hook of given size and any orientation, placed somewhere in a grid of given dimensions.
    """
    def __init__(self, model: cp.Model, hook_size: int, grid_width: int, grid_height: int) -> None:
        self.model = model
        self.grid = BooleanGrid(model, grid_height, grid_width)
        self.hook_size = hook_size

        upper_not_lower = cp.boolvar()
        left_not_right = cp.boolvar()

        # The hook is placed with its upper-left cell these coordinates.
        hook_i = cp.intvar(0, grid_height - self.hook_size)
        hook_j = cp.intvar(0, grid_width - self.hook_size)

        for i in range(grid_height):
            for j in range(grid_width):
                horizontal_bar_upper = [
                    upper_not_lower,
                    i == hook_i, hook_j <= j, j < hook_j + self.hook_size,
                ]

                horizontal_bar_lower = [
                    ~upper_not_lower,
                    i == hook_i + self.hook_size - 1, hook_j <= j, j < hook_j + self.hook_size,
                ]

                vertical_bar_left = [
                    left_not_right,
                    j == hook_j, hook_i <= i, i < hook_i + self.hook_size,
                ]

                vertical_bar_right = [
                    ~left_not_right,
                    j == hook_j + self.hook_size - 1, hook_i <= i, i < hook_i + self.hook_size,
                ]

                # Each cell is covered iff it is in one of the bars.
                self.model += self.grid[i, j] == cp.any([
                    cp.all(horizontal_bar_upper),
                    cp.all(horizontal_bar_lower),
                    cp.all(vertical_bar_left),
                    cp.all(vertical_bar_right)
                ])

class MarkedHook:
    """
    A `Hook` with `self.marked_count` of its cells "marked".
    """
    def __init__(self, model: cp.Model, hook_size: int, grid_width: int, grid_height: int, *, lb: int, ub: int) -> None:
        self.model = model
        self.hook = Hook(model, hook_size, grid_width, grid_height)
        self.grid = self.hook.grid
        self.marked_grid = BooleanGrid(model, grid_height, grid_width)
        self.marked_count = self.marked_grid.popcount
        model += self.grid.covers(self.marked_grid)

class OptionalPentomino:
    """
    A pentomino of a given type, possibly placed somewhere in a grid of given dimensions.
    """

    def __init__(self, model: cp.Model, pentomino_name: str, grid_width: int, grid_height: int) -> None:
        assert pentomino_name in PENTOMINOS, f"Unknown pentomino name: {pentomino_name}"

        self.model = model
        self.pentomino_name = pentomino_name

        self.grid = BooleanGrid(model, grid_height, grid_width)
        self.is_placed = cp.boolvar()

        orientations = PENTOMINOS[pentomino_name]

        orientation_idx = cp.intvar(0, len(orientations) - 1)
        pos_i = cp.intvar(0, grid_width - 1)
        pos_j = cp.intvar(0, grid_height - 1)

        model += (~self.is_placed).implies(self.grid.is_empty)

        for idx, coords in enumerate(orientations):
            orientation_placed = self.is_placed & (orientation_idx == idx)

            max_i = max(i for i, _ in coords)
            max_j = max(j for _, j in coords)

            fits_in_grid = (pos_i + max_i < grid_height) & (pos_j + max_j < grid_width)
            model += orientation_placed.implies(fits_in_grid)

            def cell_is_filled(i: int, j: int):
                return cp.any([
                    (pos_i + coord_1 == i) & (pos_j + coord_2 == j)
                    for coord_1, coord_2 in coords
                ])

            model += orientation_placed.implies(self.grid.each_eq(cell_is_filled))

def no_2x2_block(grid: BooleanGrid):
    return cp.all(
        [
            cp.sum([grid[i, j], grid[i, j + 1], grid[i + 1, j], grid[i + 1, j + 1]]) <= 3
            for i in range(grid.width - 1)
            for j in range(grid.height - 1)
        ]
    )

class Puzzle:
    def __init__(self, grid_size: int):
        model = cp.Model()

        # Empty cells are represented by 0
        numbers = IntGrid(model, grid_size, grid_size, lb=0, ub=grid_size)

        # Create valid hooks
        hooks = [MarkedHook(model, hook_size, grid_size, grid_size, lb=1, ub=grid_size) for hook_size in range(1, grid_size + 1)]
        model += BooleanGrid.are_disjoint([hook.grid for hook in hooks])
        model += cp.AllDifferent(hook.marked_count for hook in hooks)

        # Link hooks to numbers grid
        model += cp.all(
            (
                hook.grid.excluding(hook.marked_grid).each_implies(lambda i, j: numbers[i, j] == 0) &
                hook.marked_grid.each_implies(lambda i, j: numbers[i, j] == hook.marked_count)
            )
            for hook in hooks
        )

        # Constraints on the numbers grid
        model += no_2x2_block(numbers.nonzero_view)
        model += numbers.nonzero_view.is_connected

        pentominos = { name: OptionalPentomino(model, name, grid_size, grid_size) for name in PENTOMINOS.keys() }
        model += BooleanGrid.are_disjoint([pentomino.grid for pentomino in pentominos.values()])
        model += BooleanGrid.equals(
            BooleanGrid.union([pentomino.grid for pentomino in pentominos.values()]),
            numbers.nonzero_view
        )

        for pentomino in pentominos.values():
            # Ensure the numbers in the pentomino sum to a multiple of 5.
            # (If the pentomino isn't placed, then the sum will be 0 and hence a multiple of 5.)
            pentomino_sum = cp.sum(
                [
                    pentomino.grid[i, j] * numbers[i, j]
                    for i in range(grid_size)
                    for j in range(grid_size)
                ]
            )
            model += pentomino_sum % 5 == 0

        self.grid_size = grid_size
        self.model = model
        self.hooks = hooks
        self.numbers = numbers
        self.pentominos = pentominos

    def digit_hint(self, *, i: int, j: int, value: int):
        self.model += self.numbers[i, j] == value

    def _edge_hint(self, *, idx: int, is_row: bool, is_first: bool, hint: int | str):
        is_number = isinstance(hint, int)
        assert is_number or hint in self.pentominos

        is_filled = self.numbers.nonzero_view

        grid = self.numbers if is_number else self.pentominos[hint].grid
        value = hint if is_number else 1

        match (is_row, is_first):
            case (True, True):
                i, j = idx, is_filled.first_row_idx(idx)
            case (True, False):
                i, j = idx, is_filled.last_row_idx(idx)
            case (False, True):
                i, j = is_filled.last_col_idx(idx), idx
            case (False, False):
                i, j = is_filled.first_col_idx(idx), idx

        self.model += grid[i, j] == value

    def edge_hint_left(self, *, i: int, hint: int | str):
        self._edge_hint(idx=i, is_row=True, is_first=True, hint=hint)
    def edge_hint_right(self, *, i: int, hint: int | str):
        self._edge_hint(idx=i, is_row=True, is_first=False, hint=hint)
    def edge_hint_top(self, *, j: int, hint: int | str):
        self._edge_hint(idx=j, is_row=False, is_first=True, hint=hint)
    def edge_hint_bottom(self, *, j: int, hint: int | str):
        self._edge_hint(idx=j, is_row=False, is_first=False, hint=hint)

    def solution(self):
        if not self.model.solve():
            return None

        # The solution is the product of the areas of the empty (zero) regions in self.numbers

        filled_grid = self.numbers.nonzero_view.value()
        visited = np.zeros_like(filled_grid, dtype=bool)

        def flood_fill(start_i, start_j):
            if visited[start_i, start_j] or filled_grid[start_i, start_j]:
                return 0

            area = 0
            to_check = [(start_i, start_j)]

            while to_check:
                i, j = to_check.pop()
                if i < 0 or i >= self.grid_size or j < 0 or j >= self.grid_size or visited[i, j] or filled_grid[i, j]:
                    continue

                visited[i, j] = True
                area += 1

                for di, dj in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    to_check.append((i + di, j + dj))

            return area

        areas = [
            flood_fill(i, j)
            for i in range(self.grid_size)
            for j in range(self.grid_size)
        ]

        return math.prod(x for x in areas if x != 0)

    def print_solution(self):
        """
        Just for fun, a pretty-printer that uses constraint programming to format the grid.
        """

        BOX_CHARS = {
            # (Up, Down, Left, Right)
            (False, False, False, False): " ",
            (False, False, False, True ): "╶",
            (False, False, True , False): "╴",
            (False, False, True , True ): "─",
            (False, True , False, False): "╷",
            (False, True , False, True ): "┌",
            (False, True , True , False): "┐",
            (False, True , True , True ): "┬",
            (True , False, False, False): "╵",
            (True , False, False, True ): "└",
            (True , False, True , False): "┘",
            (True , False, True , True ): "┴",
            (True , True , False, False): "│",
            (True , True , False, True ): "├",
            (True , True , True , False): "┤",
            (True , True , True , True ): "┼",
        }


        solution = self.solution()

        if solution is None:
            rich.print("[red]No solution found[/red]")
            return

        printing_model = cp.Model()
        connects_up    = BooleanGrid(printing_model, self.grid_size * 2 + 1, self.grid_size * 4 + 1)
        connects_down  = BooleanGrid(printing_model, self.grid_size * 2 + 1, self.grid_size * 4 + 1)
        connects_left  = BooleanGrid(printing_model, self.grid_size * 2 + 1, self.grid_size * 4 + 1)
        connects_right = BooleanGrid(printing_model, self.grid_size * 2 + 1, self.grid_size * 4 + 1)
        printing_model.minimize(
            connects_up.popcount +
            connects_down.popcount +
            connects_left.popcount +
            connects_right.popcount
        )

        vertical = BooleanGrid.intersection([connects_up, connects_down])
        horizontal = BooleanGrid.intersection([connects_left, connects_right])
        has_box_char = BooleanGrid.union([connects_up, connects_down, connects_left, connects_right])

        printing_model += cp.all([
            connects_down[i, j] == connects_up[i + 1, j]
            for i in range(self.grid_size * 2)
            for j in range(self.grid_size * 4 + 1)
        ])

        printing_model += cp.all([
            connects_right[i, j] == connects_left[i, j + 1]
            for i in range(self.grid_size * 2 + 1)
            for j in range(self.grid_size * 4)
        ])

        output_grid = np.array([
            [" " for _ in range(self.grid_size * 4 + 1)]
            for _ in range(self.grid_size * 2 + 1)
        ], dtype='U50')

        pentomino_colors = {
            "F": "red",
            "I": "green",
            "L": "yellow",
            "N": "blue",
            "P": "magenta",
            "T": "cyan",
            "U": "slate_blue1",
            "V": "cyan2",
            "W": "medium_purple1",
            "X": "honeydew2",
            "Y": "magenta3",
            "Z": "dark_orange3"
        }

        def get_pentomino_for_cell(i: int, j: int) -> str | None:
            for name, pentomino in self.pentominos.items():
                if pentomino.grid[i, j].value():
                    return name
            return None

        for i, row in enumerate(self.numbers.value()):
            for j, x in enumerate(row):
                char = "·" if x == 0 else str(x)
                assert len(char) == 1, "Multi-digit numbers not supported in pretty printer"

                pentomino_name = get_pentomino_for_cell(i, j)
                if pentomino_name is not None:
                    color = pentomino_colors[pentomino_name]
                    output_grid[i * 2 + 1][j * 4 + 2] = f"[{color}]{char}[/{color}]"
                else:
                    output_grid[i * 2 + 1][j * 4 + 2] = char

        for i in range(1, self.grid_size * 2):
            printing_model += vertical[i, 0]
            printing_model += vertical[i, -1]

        for j in range(1, self.grid_size * 4):
            printing_model += horizontal[0, j]
            printing_model += horizontal[-1, j]


        for hook in self.hooks:
            for x in range(self.grid_size):
                for y in range(self.grid_size - 1):
                    if hook.grid[x, y].value() ^ hook.grid[x, y + 1].value():
                        printing_model += vertical[x * 2 + 1, y * 4 + 4]

                    if hook.grid[y, x].value() ^ hook.grid[y + 1, x].value():
                        printing_model += horizontal[y * 2 + 2, x * 4 + 1]
                        printing_model += horizontal[y * 2 + 2, x * 4 + 2]
                        printing_model += horizontal[y * 2 + 2, x * 4 + 3]

        assert printing_model.solve()

        for i, row in enumerate(has_box_char.cells):
            for j, has_char in enumerate(row):
                if not has_char.value():
                    continue
                output_grid[i, j] = BOX_CHARS[
                    connects_up[i, j].value(),
                    connects_down[i, j].value(),
                    connects_left[i, j].value(),
                    connects_right[i, j].value()
                ]


        for row in output_grid:
            rich.print("".join(row))

        print("Final answer (product of areas of empty regions):", solution)


# Puzzle instances

example = Puzzle(5)
example.digit_hint(i=0, j=4, value=4)
example.digit_hint(i=1, j=2, value=3)
example.digit_hint(i=2, j=0, value=3)
example.digit_hint(i=2, j=1, value=2)
example.digit_hint(i=3, j=2, value=1)
example.digit_hint(i=4, j=4, value=4)
example.edge_hint_left(i=0, hint="U")
example.edge_hint_right(i=1, hint="U")
example.edge_hint_right(i=3, hint="F")
example.edge_hint_left(i=4, hint="Y")

hooks_11 = Puzzle(9)
hooks_11.digit_hint(i=0, j=4, value=5)
hooks_11.digit_hint(i=1, j=3, value=4)
hooks_11.digit_hint(i=4, j=4, value=1)
hooks_11.digit_hint(i=7, j=5, value=8)
hooks_11.digit_hint(i=8, j=4, value=9)
hooks_11.edge_hint_left(i=0, hint="I")
hooks_11.edge_hint_right(i=0, hint="U")
hooks_11.edge_hint_left(i=3, hint=6)
hooks_11.edge_hint_right(i=3, hint="X")
hooks_11.edge_hint_left(i=5, hint="N")
hooks_11.edge_hint_right(i=5, hint=2)
hooks_11.edge_hint_left(i=8, hint="Z")
hooks_11.edge_hint_right(i=8, hint="V")
hooks_11.edge_hint_top(j=2, hint=3)
hooks_11.edge_hint_bottom(j=6, hint=7)


def main():
    print("Solving jane-street puzzle 2025-09 (Hooks 11)...")
    print("Example solution:")
    example.print_solution()
    print()
    print("Puzzle solution:")
    hooks_11.print_solution()

if __name__ == "__main__":
    main()
