import cpmpy as cp
import numpy as np
import typing
from cpmpy.expressions.variables import NDVarArray
from functools import cache
from functools import cached_property


class IntGrid:
    @typing.overload
    def __init__(
        self, model: cp.Model, height: int, width: int, *, lb: int, ub: int
    ): ...
    @typing.overload
    def __init__(self, model: cp.Model, height: int, width: int, *, _bools=False): ...

    def __init__(
        self,
        model: cp.Model,
        height: int,
        width: int,
        *,
        lb: int | None = None,
        ub: int | None = None,
        _bools=False,
    ):
        self.model = model
        self.width = width
        self.height = height

        assert (lb is not None and ub is not None) or _bools, (
            "Bounds must be provided for non-booleans"
        )

        self.cells: NDVarArray = (
            cp.boolvar(shape=(height, width))  # type: ignore
            if _bools
            else cp.intvar(lb, ub, shape=(height, width))  # type: ignore
        )

    def __getitem__(self, idx) -> typing.Any:
        return self.cells[idx]

    @staticmethod
    def _grid_list_helper(
        grids: typing.Sequence["IntGrid"],
    ) -> tuple[cp.Model, int, int]:
        assert len(grids) > 0, "At least one grid is required"

        model = grids[0].model
        height = grids[0].height
        width = grids[0].width

        assert all(grid.model == model for grid in grids), (
            "All grids must belong to the same model"
        )
        assert all(grid.height == height and grid.width == width for grid in grids), (
            "All grids must have the same dimensions"
        )

        return model, height, width

    @cached_property
    def nonzero_view(self) -> "BooleanGrid":
        """
        Returns a BooleanGrid where each cell is true iff the corresponding cell in this grid is non-zero.
        """

        nonzero_grid = BooleanGrid(self.model, self.height, self.width)
        self.model += cp.all(
            nonzero_grid[i, j] == (self[i, j] > 0)
            for i in range(self.height)
            for j in range(self.width)
        )
        return nonzero_grid

    _vectorized_value = np.vectorize(lambda cell: cell.value())
    def value(self):
        return IntGrid._vectorized_value(self.cells)

    def each_eq(self, fn: typing.Callable[[int, int], typing.Any]):
        """
        Returns a decision variable that is true iff the value returned by the given function
        is equal to the value in every cell of the grid.

        The function is passed the coordinates of the cell.
        """

        return cp.all(
            self[i, j] == fn(i, j)
            for i in range(self.height)
            for j in range(self.width)
        )


class BooleanGrid(IntGrid):
    def __init__(self, model: cp.Model, height: int, width: int):
        super().__init__(model, height, width, _bools=True)
        self.cells = cp.boolvar(shape=(height, width))  # type: ignore

    @staticmethod
    def are_disjoint(grids: list["BooleanGrid"]):
        """
        Returns a decision variable that is true iff all given boolean grids are disjoint.
        The grids must all have the same dimensions.
        """

        _, height, width = BooleanGrid._grid_list_helper(grids)

        return cp.all(
            cp.sum(grid[i, j] for grid in grids) <= 1
            for i in range(height)
            for j in range(width)
        )

    @staticmethod
    def union(grids: list["BooleanGrid"]) -> "BooleanGrid":
        """
        Returns a new BooleanGrid that is the union of the given grids (cellwise logical or).
        The grids must all have the same dimensions and belong to the same model.
        """

        model, height, width = BooleanGrid._grid_list_helper(grids)

        union_grid = BooleanGrid(model, height, width)

        model += cp.all(
            union_grid[i, j] == cp.any(grid[i, j] for grid in grids)
            for i in range(height)
            for j in range(width)
        )

        return union_grid

    @staticmethod
    def intersection(grids: list["BooleanGrid"]) -> "BooleanGrid":
        """
        Returns a new BooleanGrid that is the intersection of the given grids (cellwise logical and).
        The grids must all have the same dimensions and belong to the same model.
        """

        model, height, width = BooleanGrid._grid_list_helper(grids)

        intersection_grid = BooleanGrid(model, height, width)

        model += cp.all(
            intersection_grid[i, j] == cp.all(grid[i, j] for grid in grids)
            for i in range(height)
            for j in range(width)
        )

        return intersection_grid

    @staticmethod
    def equals(lhs, rhs):
        """
        Returns a decision variable that is true iff the two given boolean grids are equal.
        The grids must have the same dimensions and belong to the same model.
        """

        _, height, width = BooleanGrid._grid_list_helper([lhs, rhs])

        return cp.all(
            lhs[i, j] == rhs[i, j]
            for i in range(height)
            for j in range(width)
        )

    def excluding(self, other: "BooleanGrid") -> "BooleanGrid":
        """
        Returns a new BooleanGrid that is this grid excluding the other grid (cellwise logical and not).
        The grids must have the same dimensions and belong to the same model.
        """

        _, height, width = BooleanGrid._grid_list_helper([self, other])

        excluding_grid = BooleanGrid(self.model, height, width)

        self.model += cp.all(
            excluding_grid[i, j] == (self[i, j] & ~other[i, j])
            for i in range(height)
            for j in range(width)
        )

        return excluding_grid

    def covers(self, other: "BooleanGrid"):
        """
        Returns a decision variable that is true iff this grid covers the other grid.
        The grids must have the same dimensions and belong to the same model.
        """

        _, height, width = BooleanGrid._grid_list_helper([self, other])

        return cp.all(
            other[i, j].implies(self[i, j]) for i in range(height) for j in range(width)
        )

    def covered_by(self, other: "BooleanGrid"):
        """
        Returns a decision variable that is true iff this grid is covered by the other grid.
        The grids must have the same dimensions and belong to the same model.
        """
        return other.covers(self)

    def is_empty(self):
        """
        Returns a decision variable that is true iff this grid is empty (all cells false).
        """

        return cp.all(
            ~self[i, j] for i in range(self.height) for j in range(self.width)
        )

    def each_implies(self, fn: typing.Callable[[int, int], typing.Any]):
        """
        Returns a decision variable that is true iff the given expression is true
        for every true cell in the grid.

        The function is passed the coordinates of the cell.
        """

        return cp.all(
            self[i, j].implies(fn(i, j))
            for i in range(self.height)
            for j in range(self.width)
        )

    def is_connected(self):
        """
        Ensures that all filled cells form an orthogonally-connected region
        using a flow-based approach.
        """
        constraints = []

        def nieghbors(i, j):
            for di, dj in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                i2, j2 = i + di, j + dj
                if 0 <= i2 < self.height and 0 <= j2 < self.width:
                    yield (i2, j2)

        # We pick one true cell to be the root.
        is_root: "NDVarArray" = cp.boolvar(shape=(self.height, self.width))  # type: ignore
        constraints.append(cp.sum(is_root) == 1)

        for i1 in range(self.height):
            for j1 in range(self.width):
                constraints.append(is_root[i1][j1].implies(self[i1][j1]))

        # flow[i1, j1, i2, j2] is the amount of flow sent from cell (i1, j1) to (i2, j2)
        flow = {
            (i1, j1, i2, j2): cp.intvar(0, self.height * self.width)
            for i1 in range(self.height)
            for j1 in range(self.width)
            for i2, j2 in nieghbors(i1, j1)
        }

        true_cell_count = cp.sum(self.cells)

        for i1 in range(self.height):
            for j1 in range(self.width):
                inflow = cp.sum([
                    flow.get((i2, j2, i1, j1), 0)
                    for i2, j2 in nieghbors(i1, j1)
                ])

                outflow = cp.sum([
                    flow.get((i1, j1, i2, j2), 0)
                    for i2, j2 in nieghbors(i1, j1)
                ])

                # Only true cells can have flow
                constraints.append(((inflow > 0) | (outflow > 0)).implies(self[i1, j1])) # type: ignore

                # Every filled cell has one more unit of inflow than outflow,
                # with the exception of the root cell (which is the source of all flow).
                constraints.append(
                    self[i1, j1].implies(
                        (is_root[i1][j1] & (outflow - inflow == true_cell_count - 1))
                        | (~is_root[i1][j1] & (inflow - outflow == 1))
                    )
                )

        return cp.all(constraints)

    def _first_true_idx(self, boolvars):
        """
        Returns an decision variable that is the index of the first true cell in the given list,
        or -1 if there are no true variables in the list.
        """

        first = cp.intvar(-1, self.width - 1)

        self.model += cp.any([
            (first == -1) & (~cp.any(boolvars)),
            (first != -1) & (boolvars[first] > 0) & cp.all((j < first).implies(~boolvars[j]) for j in range(len(boolvars)) )
        ])

        return first

    @cache
    def first_row_idx(self, i: int):
        """
        Returns an decision variable that is the index of the first true cell in row `i`,
        or -1 if there are no true cells in that row.
        """
        return self._first_true_idx(self[i, :])

    @cache
    def last_row_idx(self, i: int):
        """
        Returns an decision variable that is the index of the last true cell in row `i`,
        or -1 if there are no true cells in that row.
        """
        return self.width - 1 - self._first_true_idx(self[i, :][::-1])

    @cache
    def first_col_idx(self, j: int):
        """
        Returns an decision variable that is the index of the first true cell in column `j`,
        or -1 if there are no true cells in that column.
        """
        return self._first_true_idx(self[:, j])

    @cache
    def last_col_idx(self, j: int):
        """
        Returns an decision variable that is the index of the last true cell in column `j`,
        or -1 if there are no true cells in that column.
        """
        return self.height - 1 - self._first_true_idx(self[:, j][::-1])
