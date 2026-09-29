"""Training reward choices; evaluation always uses the official game score."""

import math
from collections import Counter

from threes import Board


REWARD_DESCRIPTIONS = {
    "log_score_gain_v1": "log1p(official score gain) for a legal move; -1 otherwise",
    "scaled_score_survival_v1":
        "official score gain / 27 + 0.05 per legal move; -1 otherwise",
    "rank_weighted_merges_v1":
        "sum of merged tile levels (3=1, 6=2, 12=3, ...) per legal move; -1 otherwise",
}


def rank_weighted_merges(before: Board, moved: Board) -> int:
    """Count each merge from boards before and after movement, before tile spawn.

    Working from the largest rank down disambiguates simultaneous merges:
    count_after[r] - count_before[r] = merges_into[r] - 2*merges_into[r+1].
    """
    old = Counter(tile for row in before for tile in row if tile >= 3)
    new = Counter(tile for row in moved for tile in row if tile >= 3)
    largest = max((*old, *new), default=3)
    merges_above = 0
    reward = 0
    tile = largest
    while tile >= 3:
        merges_here = new[tile] - old[tile] + 2 * merges_above
        if merges_here < 0:
            raise ValueError("Boards do not describe a valid merge transition")
        level = (tile // 3).bit_length()
        reward += level * merges_here
        merges_above = merges_here
        tile //= 2
    return reward


def move_reward(reward_id: str, score_gain: int, moved: bool) -> float:
    if reward_id not in REWARD_DESCRIPTIONS:
        raise ValueError(f"Unknown reward_id: {reward_id}")
    if not moved:
        return -1.0
    if reward_id == "scaled_score_survival_v1":
        return score_gain / 27.0 + 0.05
    if reward_id == "rank_weighted_merges_v1":
        raise ValueError("rank_weighted_merges_v1 requires the board transition")
    return math.log1p(score_gain)
