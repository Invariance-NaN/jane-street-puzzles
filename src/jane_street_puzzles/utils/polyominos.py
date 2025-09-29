type Coord = tuple[int, int]
type PentominoCoords = tuple[Coord, Coord, Coord, Coord, Coord]

def _pentominos() -> dict[str, list[PentominoCoords]]:
    base = {
        "F": ((0, 1), (0, 2), (1, 0), (1, 1), (2, 1)),
        "I": ((0, 0), (1, 0), (2, 0), (3, 0), (4, 0)),
        "L": ((0, 0), (1, 0), (2, 0), (3, 0), (3, 1)),
        "N": ((0, 0), (0, 1), (1, 1), (1, 2), (1, 3)),
        "P": ((0, 0), (0, 1), (1, 0), (1, 1), (2, 0)),
        "T": ((0, 0), (0, 1), (0, 2), (1, 1), (2, 1)),
        "U": ((0, 0), (0, 2), (1, 0), (1, 1), (1, 2)),
        "V": ((0, 0), (1, 0), (2, 0), (2, 1), (2, 2)),
        "W": ((0, 0), (1, 0), (1, 1), (2, 1), (2, 2)),
        "X": ((0, 1), (1, 0), (1, 1), (1, 2), (2, 1)),
        "Y": ((0, 0), (0, 1), (0, 2), (0, 3), (1, 2)),
        "Z": ((0, 0), (0, 1), (1, 1), (2, 1), (2, 2)),
    }

    def normalize(coords: PentominoCoords) -> PentominoCoords:
        min_x = min(coord[0] for coord in coords)
        min_y = min(coord[1] for coord in coords)
        return tuple(sorted((x - min_x, y - min_y) for x, y in coords)) # type: ignore

    def rotations_and_reflections(coords: PentominoCoords) -> list[PentominoCoords]:
        result = set()
        current = normalize(coords)
        for _ in range(4):
            # Add the current rotation and its reflection
            result.add(current)
            result.add(normalize(tuple((x, -y) for x, y in current))) # type: ignore
            # Rotate 90 degrees
            current = normalize(tuple((y, -x) for x, y in current)) # type: ignore
        return sorted(result)

    return { name: rotations_and_reflections(coords) for name, coords in base.items() }

"""
A dictionary mapping pentomino names to a list all of the orientations of that pentomino.
"""
PENTOMINOS = _pentominos()
