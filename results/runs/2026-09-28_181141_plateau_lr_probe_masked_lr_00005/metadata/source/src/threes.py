import random
from typing import Literal, TypeAlias


Board: TypeAlias = tuple[tuple[int, ...], ...]
Direction: TypeAlias = Literal["up", "down", "left", "right"]
Hint: TypeAlias = int | tuple[int, ...]


def _validate_board(board: Board) -> None:
    if len(board) != 4 or any(len(row) != 4 for row in board):
        raise ValueError("Le plateau doit contenir 4 lignes de 4 cases")
    for row in board:
        for value in row:
            if value in (0, 1, 2):
                continue
            rank = value // 3
            if value < 3 or value > 12288 or value % 3 or rank & (rank - 1):
                raise ValueError(f"Valeur de tuile invalide : {value}")


def _can_merge(first: int, second: int) -> bool:
    return (first == 1 and second == 2) or (first == 2 and second == 1) or (
        3 <= first < 12288 and first == second
    )


def _move_line(values: tuple[int, ...]) -> tuple[int, ...]:
    moved = list(values)
    for index in range(3):
        first, second = moved[index], moved[index + 1]
        if first == 0 or _can_merge(first, second):
            moved[index] = first + second if first else second
            for following_index in range(index + 1, 3):
                moved[following_index] = moved[following_index + 1]
            moved[3] = 0
            break
    return tuple(moved)


def move(board: Board, direction: Direction) -> Board:
    if direction not in ("up", "down", "left", "right"):
        raise ValueError(f"Direction inconnue : {direction}")
    _validate_board(board)

    moved = [list(row) for row in board]
    for line_index in range(4):
        if direction in ("left", "right"):
            columns = range(4) if direction == "left" else range(3, -1, -1)
            positions = [(line_index, column) for column in columns]
        else:
            rows = range(4) if direction == "up" else range(3, -1, -1)
            positions = [(row, line_index) for row in rows]

        values = tuple(board[row][column] for row, column in positions)
        for (row, column), value in zip(positions, _move_line(values)):
            moved[row][column] = value

    return tuple(tuple(row) for row in moved)


def legal_moves(board: Board) -> tuple[Direction, ...]:
    directions: tuple[Direction, ...] = ("up", "down", "left", "right")
    return tuple(direction for direction in directions if move(board, direction) != board)


def spawn_positions(before: Board, moved: Board, direction: Direction) -> tuple[tuple[int, int], ...]:
    """Cases d'apparition équiprobables après un déplacement valide."""
    if direction == "left":
        return tuple((row, 3) for row in range(4) if moved[row] != before[row])
    if direction == "right":
        return tuple((row, 0) for row in range(4) if moved[row] != before[row])
    if direction == "up":
        return tuple(
            (3, column)
            for column in range(4)
            if any(moved[row][column] != before[row][column] for row in range(4))
        )
    if direction == "down":
        return tuple(
            (0, column)
            for column in range(4)
            if any(moved[row][column] != before[row][column] for row in range(4))
        )
    raise ValueError(f"Direction inconnue : {direction}")


def score(board: Board) -> int:
    _validate_board(board)
    return sum(3 ** (value // 3).bit_length() for row in board for value in row if value >= 3)


class Game:
    def __init__(self, board: Board | None = None, *, seed: int) -> None:
        if board is not None:
            _validate_board(board)
        self._random = random.Random(seed)
        self._bag: list[int] = []
        self._visible_ordinary_counts = [0, 0, 0]
        self.board = self._initial_board() if board is None else board
        self.next_hint: Hint = self._draw_hint()

    def _initial_board(self) -> Board:
        cells = [self._draw_tile() for _ in range(9)] + [0] * 7
        self._random.shuffle(cells)
        return tuple(tuple(cells[index : index + 4]) for index in range(0, 16, 4))

    def _draw_tile(self) -> int:
        if not self._bag:
            self._bag = [1, 2, 3] * 4
        tile = self._bag.pop(self._random.randrange(len(self._bag)))
        self._visible_ordinary_counts[tile - 1] += 1
        return tile

    def _draw_hint(self) -> Hint:
        maximum = max(value for row in self.board for value in row)
        if maximum >= 48 and self._random.randrange(21) == 0:
            choices = []
            value = 6
            while value <= maximum // 8:
                choices.append(value)
                value *= 2
            if len(choices) <= 3:
                return tuple(choices)
            start = self._random.randrange(len(choices) - 2)
            return tuple(choices[start : start + 3])
        return self._draw_tile()

    @property
    def visible_ordinary_counts(self) -> tuple[int, int, int]:
        """Nombre de 1, 2 et 3 déjà révélés depuis la création de cette partie."""
        return tuple(self._visible_ordinary_counts)

    @property
    def current_score(self) -> int:
        return score(self.board)

    @property
    def is_over(self) -> bool:
        return any(12288 in row for row in self.board) or not legal_moves(self.board)

    def step(self, direction: Direction) -> bool:
        if any(12288 in row for row in self.board):
            return False
        moved = move(self.board, direction)
        if moved == self.board:
            return False
        if any(12288 in row for row in moved):
            self.board = moved
            return True

        row, column = self._random.choice(spawn_positions(self.board, moved, direction))

        updated = [list(board_row) for board_row in moved]
        updated[row][column] = (
            self._random.choice(self.next_hint)
            if isinstance(self.next_hint, tuple)
            else self.next_hint
        )
        self.board = tuple(tuple(board_row) for board_row in updated)
        self.next_hint = self._draw_hint()
        return True
