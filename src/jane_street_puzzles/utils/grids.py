import cpmpy as cp
import numpy as np
import typing
from cpmpy.expressions.variables import NDVarArray
from functools import cache, cached_property
import jane_street_puzzles.utils.cp as cpx

class IntGrid:
    @typing.overload
    def __init__(self, model: cp.Model, height: int, width: int, *, lb: int, ub: int): ...
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
    def _grid_list_helper(grids: typing.Sequence["IntGrid"]) -> 'IntGrid':
        """
        Ensures that a given non-empty list of grids are all compatible (same model and dimensions),
        and returns one of the grids.
        """

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

        return grids[0]

    @cached_property
    def nonzero_view(self) -> "BoolGrid":
        """
        Returns a `BoolGrid` where each cell is true iff the corresponding cell in this grid is non-zero.
        """

        nonzero_grid = BoolGrid(self.model, self.height, self.width)
        self.model += cpx.all(
            nonzero_grid[i, j] == (self[i, j] > 0)
            for i, j in self.indices()
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

        return cpx.all(
            self[i, j] == fn(i, j)
            for i, j in self.indices()
        )

    @cached_property
    def total_sum(self):
        """
        Returns a decision variable that is the sum of the values of all cells in the grid.
        """
        return cpx.sum(
            self[i, j]
            for i, j in self.indices()
        )

    def indices(self):
        return ((i, j) for i in range(self.height) for j in range(self.width))

    def neighbors(self, i: int, j: int):
        for di, dj in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            i2, j2 = i + di, j + dj
            if 0 <= i2 < self.height and 0 <= j2 < self.width:
                yield (i2, j2)

class BoolGrid(IntGrid):
    def __init__(self, model: cp.Model, height: int, width: int):
        super().__init__(model, height, width, _bools=True)

    @staticmethod
    def are_disjoint(grids: list["BoolGrid"]):
        """
        Returns a decision variable that is true iff all given boolean grids are disjoint.
        The grids must all have the same dimensions.
        """

        proto = BoolGrid._grid_list_helper(grids)

        return cpx.all(
            cpx.sum(grid[i, j] for grid in grids) <= 1
            for i, j in proto.indices()
        )

    @staticmethod
    def union(grids: list["BoolGrid"]) -> "BoolGrid":
        """
        Returns a new `BoolGrid` that is the union of the given grids (cellwise logical or).
        The grids must all have the same dimensions and belong to the same model.
        """

        proto = BoolGrid._grid_list_helper(grids)

        union_grid = BoolGrid(proto.model, proto.height, proto.width)

        proto.model += cpx.all(
            union_grid[i, j] == cpx.any(grid[i, j] for grid in grids)
            for i, j in proto.indices()
        )

        return union_grid

    @staticmethod
    def intersection(grids: list["BoolGrid"]) -> "BoolGrid":
        """
        Returns a new `BoolGrid` that is the intersection of the given grids (cellwise logical and).
        The grids must all have the same dimensions and belong to the same model.
        """

        proto = BoolGrid._grid_list_helper(grids)

        intersection_grid = BoolGrid(proto.model, proto.height, proto.width)

        proto.model += cpx.all(
            intersection_grid[i, j] == cpx.all(grid[i, j] for grid in grids)
            for i, j in proto.indices()
        )

        return intersection_grid

    @staticmethod
    def equals(lhs, rhs):
        """
        Returns a decision variable that is true iff the two given boolean grids are equal.
        The grids must have the same dimensions and belong to the same model.
        """

        proto = BoolGrid._grid_list_helper([lhs, rhs])

        return cpx.all(
            lhs[i, j] == rhs[i, j]
            for i, j in proto.indices()
        )

    def excluding(self, other: "BoolGrid") -> "BoolGrid":
        """
        Returns a new `BoolGrid` that is this grid excluding the other grid (cellwise logical and not).
        The grids must have the same dimensions and belong to the same model.
        """

        proto = BoolGrid._grid_list_helper([self, other])

        excluding_grid = BoolGrid(proto.model, proto.height, proto.width)

        proto.model += cpx.all(
            excluding_grid[i, j] == (self[i, j] & ~other[i, j])
            for i, j in proto.indices()
        )

        return excluding_grid

    def covers(self, other: "BoolGrid"):
        """
        Returns a decision variable that is true iff this grid covers the other grid.
        The grids must have the same dimensions and belong to the same model.
        """

        proto = BoolGrid._grid_list_helper([self, other])

        return cpx.all(
            other[i, j].implies(self[i, j]) for i, j in proto.indices()
        )

    def covered_by(self, other: "BoolGrid"):
        """
        Returns a decision variable that is true iff this grid is covered by the other grid.
        The grids must have the same dimensions and belong to the same model.
        """
        return other.covers(self)

    def each_implies(self, fn: typing.Callable[[int, int], typing.Any]):
        """
        Returns a decision variable that is true iff the given expression is true
        for every true cell in the grid.

        The function is passed the coordinates of the cell.
        """

        return cpx.all(
            self[i, j].implies(fn(i, j))
            for i, j in self.indices()
        )

    @cached_property
    def popcount(self):
        """
        Alias for `self.total_sum`, to reflect that the sum of all of the cells
        in the grid is the number of true cells in the grid.
        """
        return self.total_sum

    @cached_property
    def is_empty(self):
        """
        Returns a decision variable that is true iff this grid is empty (all cells false).
        """
        return self.popcount == 0

    @cached_property
    def _ensure_connected(self):
        """
        Returns a decision variable that is true only if the true cells in
        this grid are orthogonally connected.

        The variable can be false even if the grid is connected.
        """

        # We use the notion of flow to check connectivity.
        # We put one unit of flow for each true cell into the system (at the "root"),
        # and require that every other cell has one more unit of inflow than outflow,
        # which can only be the case if the flow stemming from the root can reach every true cell.

        # We pick one true cell to be the root, unless the grid is empty.
        is_root = BoolGrid(self.model, self.height, self.width)
        self.model += (is_root.popcount <= 1) & is_root.is_empty.implies(self.is_empty)
        self.model += is_root.each_implies(lambda i, j: self[i, j])

        # The amount of flow sent from cell (i1, j1) to (i2, j2) is denoted by `flow[i1, j1, i2, j2]`.
        flow = {
            (i1, j1, i2, j2): cp.intvar(0, self.height * self.width)
            for i1, j1 in self.indices()
            for i2, j2 in self.neighbors(i1, j1)
        }

        @cache
        def inflow(i, j):
            return cpx.sum(flow.get((i2, j2, i, j), cpx.zero) for i2, j2 in self.neighbors(i, j))

        @cache
        def outflow(i, j):
            return cpx.sum(flow.get((i, j, i2, j2), cpx.zero) for i2, j2 in self.neighbors(i, j))

        flow_only_on_true_cells = cpx.all(
            ((inflow(i, j) != 0) | (outflow(i, j) != 0)).implies(self[i, j])
            for i, j in self.indices()
        )

        self.model += flow_only_on_true_cells

        flow_preserved = cpx.all(
            self[i, j].implies(
                ( is_root[i, j] & (outflow(i, j) - inflow(i, j) == self.popcount - 1)) |
                (~is_root[i, j] & (inflow(i, j) - outflow(i, j) == 1))
            )
            for i, j in self.indices()
        )

        return flow_preserved

    @cached_property
    def _ensure_disconnected(self):
        """
        Returns a decision variable that is true only if the true cells in
        this grid are NOT orthogonally connected.

        The variable can be false even if the grid is connected.
        """

        # A graph is disconnected iff it can be partitioned into two non-empty sets
        # such that any two adjacent vertices are in the same set.

        # Partition B is implicitly the cells that are true in `self` but not in `partition_A`
        partition_A = BoolGrid(self.model, self.height, self.width)
        self.model += self.covers(partition_A)

        partition_A_nonempty = partition_A.popcount >= 1
        partition_B_nonempty = (self.popcount - partition_A.popcount) >= 1

        no_cross_edges = cpx.all(
            partition_A[i1, j1].implies(
                cpx.all(
                    (~self[i2, j2]) | partition_A[i2, j2]
                    for i2, j2 in self.neighbors(i1, j1)
                )
            )
            for i1, j1 in self.indices()
        )

        return partition_A_nonempty & partition_B_nonempty & no_cross_edges


    @cached_property
    def is_connected(self):
        """
        Returns a decision variable that is true if and only if the true cells in
        this grid are orthogonally connected.
        """
        result = cp.boolvar()
        self.model += result.implies(self._ensure_connected) & (~result).implies(self._ensure_disconnected)

        return result



    def _first_true_idx(self, boolvars):
        """
        Returns an decision variable that is the index of the first true cell in the given list,
        or -1 if there are no true variables in the list.
        """

        first = cp.intvar(-1, self.width - 1)

        self.model += (
            ((first == -1) & (~cpx.any(boolvars))) |
            ((first != -1) & (boolvars[first] > 0) & cpx.all((j < first).implies(~boolvars[j]) for j in range(len(boolvars))))
        )

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
