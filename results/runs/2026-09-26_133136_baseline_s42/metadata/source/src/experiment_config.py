"""Schéma unique des réglages PPO, y compris les anciens fichiers plats.

Le code travaille avec les clés historiques à plat. Les nouveaux JSON sont
regroupés en sections lisibles ; l'adaptateur évite de réécrire les runs passés.
"""

import json
import math
from pathlib import Path

from rewards import REWARD_DESCRIPTIONS


SECTIONS = {
    "model": ("net_arch",),
    "ppo": ("n_steps", "batch_size", "n_epochs", "learning_rate", "gamma"),
    "training": ("seed", "train_steps"),
    "evaluation": ("eval_interval", "eval_games", "eval_seed"),
}
KEYS = {key for names in SECTIONS.values() for key in names}


def normalize_config(document: dict) -> dict:
    """Valide et aplatit un JSON structuré ou une ancienne config plate."""
    if not isinstance(document, dict):
        raise ValueError("La configuration doit être un objet JSON")
    if set(document) in (set(SECTIONS), set(SECTIONS) | {"reward"}):
        config = {}
        for section, names in SECTIONS.items():
            values = document[section]
            if not isinstance(values, dict) or set(values) != set(names):
                raise ValueError(f"La section {section} doit contenir : {', '.join(names)}")
            config.update(values)
        if "reward" in document:
            reward = document["reward"]
            if not isinstance(reward, dict) or set(reward) != {"reward_id"}:
                raise ValueError("The reward section must contain reward_id")
            config.update(reward)
    elif set(document) in (KEYS, KEYS | {"reward_id"}):
        config = document.copy()
    else:
        raise ValueError("Champs de configuration manquants ou inconnus ; voir configs/ppo.json")

    config.setdefault("reward_id", "log_score_gain_v1")
    if config["reward_id"] not in REWARD_DESCRIPTIONS:
        raise ValueError(f"Unknown reward_id: {config['reward_id']}")

    def integer(name: str, minimum: int = 1) -> None:
        value = config[name]
        if type(value) is not int or value < minimum:
            raise ValueError(f"{name} doit être un entier >= {minimum}")

    integer("seed", 0)
    integer("eval_seed", 0)
    for name in ("n_steps", "batch_size", "n_epochs", "train_steps",
                 "eval_interval", "eval_games"):
        integer(name)
    if (not isinstance(config["net_arch"], list) or not config["net_arch"] or
            any(type(width) is not int or width < 1 for width in config["net_arch"])):
        raise ValueError("net_arch doit être une liste non vide d'entiers positifs")
    if config["n_steps"] < 2 or config["batch_size"] < 2:
        raise ValueError("n_steps et batch_size doivent valoir au moins 2 pour PPO")
    if config["n_steps"] % config["batch_size"]:
        raise ValueError("n_steps doit être divisible par batch_size")
    for name in ("learning_rate", "gamma"):
        value = config[name]
        if (isinstance(value, bool) or not isinstance(value, (int, float)) or
                not math.isfinite(value) or value <= 0):
            raise ValueError(f"{name} doit être un nombre positif")
    if config["gamma"] > 1:
        raise ValueError("gamma doit être <= 1")
    return config


def load_config(path: Path) -> dict:
    return normalize_config(json.loads(path.read_text(encoding="utf-8-sig")))


def config_document(config: dict) -> dict:
    """Produit le format rangé qui sera figé dans les nouvelles runs."""
    checked = normalize_config(config)
    document = {section: {name: checked[name] for name in names}
                for section, names in SECTIONS.items()}
    document["reward"] = {"reward_id": checked["reward_id"]}
    return document


def write_config(path: Path, config: dict) -> None:
    path.write_text(
        json.dumps(config_document(config), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
