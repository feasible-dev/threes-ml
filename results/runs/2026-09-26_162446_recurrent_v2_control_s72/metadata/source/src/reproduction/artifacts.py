"""Human-readable artifacts; no Torch or simulator imports needed to browse runs."""

import csv
import json
from pathlib import Path


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def dashboard_config(config):
    return {"net_arch": [2 * config["model"]["hidden_size"], config["model"]["hidden_size"]],
            **config["ppo"], "seed": config["training"]["seed"],
            "train_steps": config["training"]["steps"],
            "eval_games": config["evaluation"]["games"], "eval_seed": config["evaluation"]["seed"],
            "method": "Published LSTM reconstruction", "activation": "GELU",
            "model_label": f"GELU encoder + shared LSTM {config['model']['hidden_size']}"}


def catalog_record(run_dir):
    config = read_json(run_dir / "metadata" / "config.json")
    method = read_json(run_dir / "metadata" / "method.json")
    provenance = read_json(run_dir / "metadata" / "provenance.json")
    with (run_dir / "data" / "metrics.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    evaluations = [row for row in rows if row.get("eval_mean")]
    last = rows[-1] if rows else {}
    final = evaluations[-1] if evaluations else {}
    best = max(evaluations, key=lambda row: float(row["eval_mean"])) if evaluations else {}
    ppo, training, evaluation = config["ppo"], config["training"], config["evaluation"]
    status_path = run_dir / "metadata" / "status.json"
    status = read_json(status_path).get("status") if status_path.exists() else "in progress"
    review_path = run_dir / "metadata" / "review.json"
    review = read_json(review_path) if review_path.exists() else {}
    comparison_path = run_dir / "evaluation" / "reproduction_validation.json"
    comparison = read_json(comparison_path) if comparison_path.exists() else {}
    independent = comparison.get("metrics", {})
    reference = comparison.get("reference") or {}
    ratio = independent.get("eval_mean", 0) / reference["eval_mean"] if reference.get("eval_mean") else ""
    row = {
        "run": run_dir.name, "steps": int(last.get("steps", 0)), "seconds": float(last.get("seconds", 0)),
        "train_seed": training["seed"], "network": f"LSTM-{config['model']['hidden_size']}",
        "reward_id": "published_merge_shaped" if config["environment"]["heuristic"] else "published_merge",
        "reward": "Published rank*0.0625 merges, invalid -0.05, death -1; clipped to [-1,1]",
        **{key: ppo[key] for key in ("n_steps", "batch_size", "n_epochs", "learning_rate", "gamma")},
        "eval_games": evaluation["games"], "eval_seed": evaluation["seed"],
        "eval_final_mean": final.get("eval_mean", ""), "eval_best_mean": best.get("eval_mean", ""),
        "holdout_mean": independent.get("eval_mean", ""), "holdout_games": independent.get("eval_games", ""),
        "reference_mean": reference.get("eval_mean", ""), "reference_ratio": ratio,
        "relevance": review.get("relevance", "pending review"),
        "retention": review.get("retention", "undecided"),
        "insight": review.get("insight", "Review learning curves and independent evaluation."),
        "started_at_local": provenance["started_at_local"], "git_commit": provenance.get("git_commit") or "",
        "source_status": provenance["source_status"],
    }
    card = f"""# {run_dir.name}

**Status:** {status}. **Method:** recurrent PPO with a reconstruction of the published ONNX architecture.

The network starts from random weights. Published weights are used only in the separate equivalence test and reference evaluation.

| Setting | Value |
| --- | --- |
| Network | 18 x 8 tile embedding + 17 x 8 position embedding; bag 4 -> 32; encoder 168 -> {2*config['model']['hidden_size']} -> {config['model']['hidden_size']} -> {config['model']['hidden_size']}; shared LSTM; two separate hidden heads; GELU |
| Parameters | {method['parameters']:,} |
| Training run seed | {training['seed']} |
| Transitions | {row['steps']:,} |
| Environments / rollout / batch / epochs | {ppo['n_envs']} / {ppo['n_steps']} / {ppo['batch_size']} / {ppo['n_epochs']} |
| Learning rate / gamma / entropy coefficient | {ppo['learning_rate']} / {ppo['gamma']} / {ppo['ent_coef']} |
| Shaping / curriculum probability | {config['environment']['heuristic']} / {config['environment']['scaffold']} |
| Engine revision / reward scale / timeout bootstrap | {method.get('engine_revision', 1)} / {config['environment'].get('reward_scaler', 1)} / {config['environment'].get('bootstrap_timeouts', True)} |
| Learning-rate schedule (absolute transitions) | {training.get('lr_schedule', 'constant')} |
| Device | {method['device']} |
| Latest / best validation mean | {row['eval_final_mean']} / {row['eval_best_mean']} |
| Independent comparison mean / games | {row['holdout_mean']} / {row['holdout_games']} |
| Published reference mean, same rules and seeds | {row['reference_mean']} |

Training game seeds are logged and lie above 2^63. Validation seeds lie below 2^63. Validation uses natural initial boards and legal greedy actions, with no curriculum.

The catalog's historical `holdout` column contains independent validation for this pipeline, not an untouched final test. [Latest comparison]({comparison.get('report', 'evaluation/')}).

## Interpretation and retention

**Relevance:** {row['relevance']}. **Retention:** {row['retention']}.

{row['insight']}

No cleanup happens automatically. Record research judgments in `metadata/review.json`.

## Files

- [Configuration](metadata/config.json), [method and limitations](metadata/method.json), [source provenance](metadata/provenance.json)
- [Metrics](data/metrics.csv), [training episodes and seeds](data/episodes.csv), [HTML curves](report/report.html)
- [Evaluation games](evaluation/), [checkpoints](models/checkpoints/)
"""
    return row, card
