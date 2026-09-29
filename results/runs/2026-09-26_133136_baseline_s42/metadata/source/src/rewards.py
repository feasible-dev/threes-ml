"""Training reward choices; evaluation always uses the official game score."""

import math


REWARD_DESCRIPTIONS = {
    "log_score_gain_v1": "log1p(official score gain) for a legal move; -1 otherwise",
    "scaled_score_survival_v1":
        "official score gain / 27 + 0.05 per legal move; -1 otherwise",
}


def move_reward(reward_id: str, score_gain: int, moved: bool) -> float:
    if reward_id not in REWARD_DESCRIPTIONS:
        raise ValueError(f"Unknown reward_id: {reward_id}")
    if not moved:
        return -1.0
    if reward_id == "scaled_score_survival_v1":
        return score_gain / 27.0 + 0.05
    return math.log1p(score_gain)
