import pytest
import cpmpy as cp
from jane_street_puzzles.utils.grids import BooleanGrid
import hypothesis
import hypothesis.strategies as st
import hypothesis.extra.numpy as hnp

def make_grid(model, xss):
    height = len(xss)
    width = len(xss[0]) if height > 0 else 0

    grid = BooleanGrid(model, height=height, width=width)
    model += grid.each_eq(lambda i, j: xss[i][j])

    return grid

def is_grid_connected(xss):
    xss = [list(row) for row in xss]
    start = None
    for i, row in enumerate(xss):
        for j, val in enumerate(row):
            if val:
                start = (i, j)
                break
        else:
            continue
        break

    if start is None:
        return True

    def propogate_false(i, j):
        if i < 0 or i >= len(xss) or j < 0 or j >= len(xss[0]) or not xss[i][j]:
            return
        xss[i][j] = False
        propogate_false(i+1, j)
        propogate_false(i-1, j)
        propogate_false(i, j+1)
        propogate_false(i, j-1)

    propogate_false(start[0], start[1])

    return not any(any(row) for row in xss)


@hypothesis.given(hnp.arrays(dtype=bool, shape=st.tuples(st.integers(0, 10), st.integers(0, 10))))
@hypothesis.settings(deadline=None)
def test_is_connected(xss):
    # Make sure that `is_connected` can have the correct answer...
    model_a = cp.Model()
    grid_a = make_grid(model_a, xss)
    model_a += (grid_a.is_connected == is_grid_connected(xss))
    assert model_a.solve()

    # ... and that it cannot have the incorrect answer.
    model_b = cp.Model()
    grid_b = make_grid(model_b, xss)
    model_b += (grid_b.is_connected != is_grid_connected(xss))
    assert not model_b.solve()

if __name__ == "__main__":
    import numpy as np
    xss = np.array([[True, True]])

    model_a = cp.Model()
    grid_a = make_grid(model_a, xss)
    model_a += (grid_a.is_connected == is_grid_connected(xss))
    assert model_a.solve()

    # ... and that it cannot have the incorrect answer.
    model_b = cp.Model()
    grid_b = make_grid(model_b, xss)
    model_b += (grid_b.is_connected != is_grid_connected(xss))
    assert not model_b.solve()

    pytest.main([__file__])
