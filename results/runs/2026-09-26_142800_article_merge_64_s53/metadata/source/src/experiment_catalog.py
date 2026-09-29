"""Build a readable run summary and a comparative experiment catalog."""

import csv
import json
from pathlib import Path
from statistics import mean, median

from experiment_config import load_config
from run_layout import (RUNS_ROOT, checkpoints_path, config_path, method_path,
                        metrics_path, model_path, provenance_path, report_path,
                        review_path)


CATALOG_FIELDS = (
    "run", "steps", "seconds", "train_seed", "network", "reward_id", "reward", "n_steps",
    "batch_size", "n_epochs", "learning_rate", "gamma", "eval_games",
    "eval_seed", "eval_final_mean", "eval_best_mean", "holdout_mean",
    "holdout_games", "reference_mean", "reference_ratio", "relevance",
    "retention", "insight", "started_at_local", "git_commit", "source_status",
)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig")) if path.is_file() else {}


def scores_from_csv(path: Path, agent: str) -> list[int]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        scores = [int(row["score"]) for row in csv.DictReader(stream) if row["agent"] == agent]
    if not scores:
        raise ValueError(f"No scores for {agent} in {path}")
    return scores


def holdout_summary(run_dir: Path, specification: dict) -> dict:
    if not specification:
        return {}
    path = run_dir / specification["file"]
    scores = scores_from_csv(path, specification["agent"])
    return {
        "mean": mean(scores), "median": median(scores), "max": max(scores),
        "games": len(scores), "file": specification["file"],
    }


def fmt(value: object, digits: int = 1) -> str:
    if value is None or value == "":
        return "—"
    if isinstance(value, (int, float)):
        return f"{value:,.{digits}f}"
    return str(value)


def safe(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def metric_summary(path: Path) -> tuple[dict, dict, dict]:
    """Read the log once without keeping every training update in memory."""
    last, final_eval, best_eval = {}, {}, {}
    with path.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            if not row.get("steps"):
                continue
            last = row
            if row.get("eval_mean"):
                final_eval = row
                if not best_eval or float(row["eval_mean"]) > float(best_eval["eval_mean"]):
                    best_eval = row
    return last, final_eval, best_eval


def summarize_run(run_dir: Path) -> tuple[dict, str]:
    config = load_config(config_path(run_dir))
    method = read_json(method_path(run_dir))
    review = read_json(review_path(run_dir))
    provenance = read_json(provenance_path(run_dir))
    last, final_eval, best_eval = metric_summary(metrics_path(run_dir))
    holdout = holdout_summary(run_dir, review.get("holdout", {}))
    reference = holdout_summary(run_dir, review.get("reference", {}))
    ratio = holdout["mean"] / reference["mean"] if holdout and reference else None
    network = method.get("network", {})
    observation = method.get("observation", {})
    ppo = method.get("ppo", {})
    hidden = network.get("policy_hidden", config.get("net_arch", []))
    row = {
        "run": run_dir.name,
        "steps": int(last["steps"]) if last else 0,
        "seconds": float(last["seconds"]) if last else 0,
        "train_seed": config.get("seed", ""),
        "network": "-".join(str(width) for width in hidden),
        "reward_id": method.get("reward_id", "unknown"),
        "reward": method.get("reward", "method not archived"),
        "n_steps": config.get("n_steps", ""),
        "batch_size": config.get("batch_size", ""),
        "n_epochs": config.get("n_epochs", ""),
        "learning_rate": config.get("learning_rate", ""),
        "gamma": config.get("gamma", ""),
        "eval_games": config.get("eval_games", ""),
        "eval_seed": config.get("eval_seed", ""),
        "eval_final_mean": final_eval.get("eval_mean", ""),
        "eval_best_mean": best_eval.get("eval_mean", ""),
        "holdout_mean": round(holdout["mean"], 2) if holdout else "",
        "holdout_games": holdout.get("games", ""),
        "reference_mean": round(reference["mean"], 2) if reference else "",
        "reference_ratio": round(ratio, 3) if ratio is not None else "",
        "relevance": review.get("relevance", "pending review"),
        "retention": review.get("retention", "undecided"),
        "insight": review.get("insight", "Discuss after the run."),
        "started_at_local": provenance.get("started_at_local", ""),
        "git_commit": provenance.get("git_commit") or "",
        "source_status": provenance.get("source_status", "inconnu"),
    }

    status = "complete" if model_path(run_dir).is_file() else "interrupted or in progress"
    source_status = provenance.get("source_status", "inconnu")
    code_description = ("Source copy archived in `metadata/source/` with SHA-256 hashes."
                        if source_status == "snapshot_saved" else
                        "The exact code for this historical run was not archived; current files may differ.")
    if provenance.get("resumes"):
        code_description += (f" Source from {len(provenance['resumes'])} resumed segment(s) "
                             "is archived in `metadata/resumes/`.")
    default_question = (f"Evaluate {method.get('method', 'MaskablePPO')} with a "
                        f"{row['network']} network, {row['reward_id']} reward, and seed "
                        f"{row['train_seed']}.")
    card = [
        f"# {run_dir.name}", "", f"**Status:** {status}. **Question:** {review.get('question', default_question)}", "",
        "## Provenance and code", "",
        f"- Started: {provenance.get('started_at_local', 'unknown')}",
        f"- Git commit: {provenance.get('git_commit') or 'unavailable (no accessible Git repository)'}",
        f"- Code: {code_description}",
        "- [Provenance and hashes](metadata/provenance.json)", "",
        "## Parameters", "",
        "| Item | Value |", "| --- | --- |",
        f"| Method | {safe(method.get('method', 'MaskablePPO'))} |",
        f"| Policy network | {observation.get('dimension', '—')} → {safe(row['network'])} → {network.get('policy_outputs', '—')}, activation {safe(network.get('activation', '—'))} |",
        f"| Value network | {observation.get('dimension', '—')} → {safe('-'.join(str(x) for x in network.get('value_hidden', hidden)))} → {network.get('value_outputs', '—')} |",
        f"| Trainable parameters | {safe(network.get('trainable_parameters', '—'))} |",
        f"| Reward | {safe(row['reward_id'])}: {safe(row['reward'])} |",
        f"| Training seed | {safe(row['train_seed'])} |",
        f"| Initial budget / cumulative transitions | {safe(config.get('train_steps', '—'))} / {row['steps']} |",
        f"| PPO: rollout / batch / epochs | {row['n_steps']} / {row['batch_size']} / {row['n_epochs']} |",
        f"| PPO: learning rate / γ / GAE λ | {row['learning_rate']} / {row['gamma']} / {safe(ppo.get('gae_lambda', '—'))} |",
        f"| PPO: clip / value weight / entropy weight | {safe(ppo.get('clip_range', '—'))} / {safe(ppo.get('vf_coef', '—'))} / {safe(ppo.get('ent_coef', '—'))} |",
        f"| Periodic evaluation | {row['eval_games']} games from seed {row['eval_seed']}, every {config.get('eval_interval', '—')} steps |",
        "", safe(method.get("action_mask", "Legal-action masks are supplied separately from the observation")) + ".", "",
        "## Results", "",
        "| Measure | Value |", "| --- | ---: |",
        f"| Cumulative duration | {fmt(row['seconds'])} s |",
        f"| Latest periodic mean | {fmt(float(row['eval_final_mean'])) if row['eval_final_mean'] else '—'} |",
        f"| Best periodic mean | {fmt(float(row['eval_best_mean'])) if row['eval_best_mean'] else '—'} |",
        f"| Held-out mean | {fmt(holdout.get('mean'))} ({holdout.get('games', '—')} games) |",
        f"| Reference on these games | {fmt(reference.get('mean'))} |",
        f"| Share of reference | {fmt(100 * ratio) + '%' if ratio is not None else '—'} |",
        "", "Periodic evaluations track training; held-out games support the final comparison.", "",
        "## Interpretation and retention", "",
        f"**Relevance:** {review.get('relevance', 'pending review')}. ",
        f"**Finding:** {review.get('insight', 'Discuss after the run.')}", "",
        f"**Suggested retention:** {review.get('retention', 'undecided')}. ",
        review.get("retention_reason", "No files are deleted automatically."), "",
        "## Files", "",
        "- [Frozen configuration](metadata/config.json)",
        "- [Method description](metadata/method.json)" if method_path(run_dir).is_file() else "- Method description unavailable",
        "- [Per-update metrics](data/metrics.csv)",
        "- [Charts](report/report.html)" if report_path(run_dir).is_file() else "- Charts unavailable",
        "- [Final model](models/model.zip)" if model_path(run_dir).is_file() else "- Final model unavailable",
        "- [Checkpoints](models/checkpoints/)" if checkpoints_path(run_dir).is_dir() else "- No checkpoints",
        "- [Editable review](metadata/review.json)" if review_path(run_dir).is_file() else "- Create a review in metadata/review.json",
        "- [Held-out evaluation](evaluation/holdout.csv)" if (run_dir / "evaluation" / "holdout.csv").is_file() else "- Held-out evaluation unavailable",
        "",
    ]
    return row, "\n".join(card)


def write_run_card(run_dir: Path) -> dict:
    """Write one run summary, even when training is incomplete."""
    row, card = summarize_run(run_dir)
    (run_dir / "FICHE.md").write_text(card, encoding="utf-8")
    return row


def update_catalog(runs_root: Path = RUNS_ROOT) -> None:
    rows = []
    for run_dir in sorted(runs_root.iterdir()):
        if not run_dir.is_dir() or not config_path(run_dir).is_file() or not metrics_path(run_dir).is_file():
            continue
        rows.append(write_run_card(run_dir))

    results_dir = runs_root.parent
    with (results_dir / "experiences.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CATALOG_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Experiment catalog", "",
        "Each run has a summary in its directory. Numbers come from CSV files; the review in",
        "`metadata/review.json` is written after we examine the results.",
        "“Pending review” marks a completed run that we have not discussed yet.", "",
        "## Configurations", "",
        "| Run | Local start | Code | Steps | Seed | Network | Reward | PPO rollout/batch/epochs | Rate / γ |",
        "| --- | --- | --- | ---: | ---: | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| [{row['run']}](runs/{row['run']}/FICHE.md) | {row['started_at_local']} | "
            f"{'snapshot archived' if row['source_status'] == 'snapshot_saved' else 'incomplete history'} | {row['steps']} | "
            f"{row['train_seed']} | {row['network']} | {row['reward_id']} | "
            f"{row['n_steps']}/{row['batch_size']}/{row['n_epochs']} | {row['learning_rate']} / {row['gamma']} |"
        )
    lines += [
        "", "## Results and relevance", "",
        "| Run | Latest tracking | Best tracking | Held-out games | Reference | Ratio | Relevance | Retention |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for row in rows:
        holdout = f"{fmt(row['holdout_mean'])} ({row['holdout_games']})" if row["holdout_mean"] != "" else "—"
        ratio = f"{fmt(100 * row['reference_ratio'])} %" if row["reference_ratio"] != "" else "—"
        lines.append(
            f"| [{row['run']}](runs/{row['run']}/FICHE.md) | "
            f"{fmt(float(row['eval_final_mean'])) if row['eval_final_mean'] else '—'} | "
            f"{fmt(float(row['eval_best_mean'])) if row['eval_best_mean'] else '—'} | "
            f"{holdout} | {fmt(row['reference_mean'])} | {ratio} | {safe(row['relevance'])} | {safe(row['retention'])} |"
        )
    lines += [
        "", "The latest tracking and held-out columns use different seeds.",
        "The same initial seed does not force two agents to see the same draws after they choose different moves.", "",
        "## Read and sort", "",
        "- Open a run summary for its parameters, question, finding, and files.",
        "- Open [experiences.csv](experiences.csv) in a spreadsheet to filter or sort runs.",
        "- `metadata/config.json` freezes chosen parameters; `metadata/method.json` describes the model and PPO.",
        "- `metadata/provenance.json` records the commit or source hashes and code snapshot.",
        "- `data/metrics.csv` has every update; `report/report.html` shows the charts.",
        "- `metadata/review.json` holds our assessment. The catalog never deletes files.", "",
        "## Relevance scale", "",
        "- **high**: a result or diagnosis worth revisiting.",
        "- **medium**: a useful comparison or validation, partly redundant.",
        "- **low**: a technical check or result without a reliable conclusion.",
        "Performance and relevance are separate: a weak run may reveal an important problem.", "",
        "## Suggested retention", "",
        "- **complete**: keep the model and detailed records for later work.",
        "- **trim later**: keep results and the summary; review checkpoints during cleanup.",
        "- **undecided**: wait until the run is interpreted.",
        "These are recommendations; no files are deleted automatically.", "",
        "## Early runs outside the table", "",
        "The [first 20,096-transition PPO run](runs/2026-09-25_182257_ppo_20k_initial/FICHE.md) and the [PPO smoke test](runs/2026-09-25_182225_ppo_smoke/FICHE.md) predate automatic run tracking.",
        "The first scored 344.9 points over 100 games. That historical measurement does not use",
        "the held-out seeds from the runs in this table.", "",
    ]
    (results_dir / "EXPERIENCES.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Catalog updated: {results_dir / 'EXPERIENCES.md'} ({len(rows)} runs)", flush=True)


if __name__ == "__main__":
    update_catalog()
