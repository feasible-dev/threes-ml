"""Information publique destinée aux futurs agents appris."""

from dataclasses import dataclass

from threes import Game, legal_moves


ACTIONS = ("up", "down", "left", "right")


def tile_rank(value: int) -> int:
    if value < 3:
        return value
    return (value // 3).bit_length() + 2


@dataclass(frozen=True)
class Observation:
    board_ranks: tuple[int, ...]
    hint_mask: tuple[int, ...]
    bag_remaining: tuple[int, int, int]
    legal_actions: tuple[bool, bool, bool, bool]


def observe(game: Game) -> Observation:
    """Calcule seulement des faits visibles ou déductibles du suivi des cartes."""
    board_ranks = tuple(tile_rank(value) for row in game.board for value in row)
    hint_values = game.next_hint if isinstance(game.next_hint, tuple) else (game.next_hint,)
    hint_ranks = {tile_rank(value) for value in hint_values}
    hint_mask = tuple(int(rank in hint_ranks) for rank in range(16))

    counts = game.visible_ordinary_counts
    completed_bags = sum(counts) // 12
    bag_remaining = tuple(4 - (count - 4 * completed_bags) for count in counts)
    available = set() if any(12288 in row for row in game.board) else set(legal_moves(game.board))
    legal_actions = tuple(action in available for action in ACTIONS)
    return Observation(board_ranks, hint_mask, bag_remaining, legal_actions)
