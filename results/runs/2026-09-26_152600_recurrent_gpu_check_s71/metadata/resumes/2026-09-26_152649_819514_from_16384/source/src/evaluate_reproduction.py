"""Compare trained recurrent policies and optionally the published reference."""

import argparse
import csv
import hashlib
import json
import math
from datetime import datetime
from pathlib import Path
from time import perf_counter

import gymnasium as gym
import numpy as np
import torch

from reproduction.evaluation import evaluate
from reproduction.policy import PublishedPolicy
from reproduction.training import TrackedRecurrentPPO
from run_layout import PROJECT_ROOT


def wilson(successes, count):
    z = 1.959963984540054
    p = successes / count
    denominator = 1 + z * z / count
    center = (p + z * z / (2 * count)) / denominator
    radius = z * math.sqrt(p * (1 - p) / count + z * z / (4 * count * count)) / denominator
    return center - radius, center + radius


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models', nargs='*', type=Path, default=[])
    parser.add_argument('--published', action='store_true', help='Also evaluate the downloaded pretrained reference')
    parser.add_argument('--games', type=int, default=300)
    parser.add_argument('--seed', type=int, default=800000)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if not args.models and not args.published:
        parser.error('Provide --models and/or --published')
    if args.games < 1 or args.batch_size < 1 or not 0 <= args.seed < 2**63 - args.games:
        parser.error('Positive games/batch size and evaluation seeds below 2**63 are required')
    if args.device == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA unavailable; use .venv-repro or request --device cpu')
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    output = args.output or PROJECT_ROOT / 'results' / 'comparisons' / datetime.now().strftime('%Y-%m-%d_%H%M%S_recurrent')
    output.mkdir(parents=True, exist_ok=False)
    sources = [(path.parent.parent.name, path, False) for path in args.models]
    if args.published:
        sources.append(('published_pretrained_reference', PROJECT_ROOT / 'external_models/kiok/onnx_model.onnx', True))
    rows = []
    print(f'{args.games} games per policy | seeds {args.seed}..{args.seed + args.games - 1} | native reproduction rules', flush=True)
    for index, (name, path, pretrained) in enumerate(sources):
        print(f'Evaluating {name} ...', flush=True)
        if pretrained:
            policy = PublishedPolicy(gym.spaces.Box(0, 17, (21,), np.float32), gym.spaces.Discrete(4), lambda _: 0.)
            policy.import_onnx(path)
            policy = policy.to(args.device)
        else:
            model = TrackedRecurrentPPO.load(path, device=args.device)
            policy = model.policy
        start = perf_counter()
        metrics = evaluate(policy, args.games, args.seed, batch_size=args.batch_size,
                           output=output / f'{index:02d}_{name}.csv')
        lo, hi = wilson(round(metrics['eval_6144'] * args.games), args.games)
        row = {'model': name, 'pretrained': pretrained, **metrics, '6144_ci_low': lo, '6144_ci_high': hi,
               'seconds': perf_counter() - start, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
        rows.append(row)
        print(f"Mean {metrics['eval_mean']:.1f} | >=3072 {metrics['eval_3072']:.1%} | "
              f">=6144 {metrics['eval_6144']:.1%} (95% CI {lo:.1%}..{hi:.1%}) | "
              f"truncated {metrics['eval_truncated']}", flush=True)
        # Preserve partial results if a later evaluation is interrupted.
        with (output / 'models.csv').open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(row))
            writer.writeheader()
            writer.writerows(rows)
        del policy
        if not pretrained:
            del model
    metadata = {'games': args.games, 'seed': args.seed, 'device': args.device, 'terminal_rank': 16,
                'rules': 'reproduction.environment.NativeBatch', 'action_selection': 'legal greedy',
                'curriculum': False, 'purpose': 'validation; final-test claims require a frozen protocol',
                'environment_sha256': hashlib.sha256((PROJECT_ROOT / 'src/reproduction/environment.py').read_bytes()).hexdigest()}
    (output / 'protocol.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    lines = ['# Recurrent model comparison', '',
             f'{args.games} games per policy; seeds {args.seed} through {args.seed + args.games - 1}.', '',
             'All policies use the same native reproduction simulator, natural initial boards, legal greedy actions, and reset recurrent memory between games.', '',
             '| Model | Mean | Median | >=3072 | >=6144 | 6144 95% Wilson CI | Truncated |',
             '| --- | ---: | ---: | ---: | ---: | --- | ---: |']
    for row in rows:
        lines.append(f"| {row['model']} | {row['eval_mean']:.1f} | {row['eval_median']:.1f} | "
                     f"{row['eval_3072']:.1%} | {row['eval_6144']:.1%} | "
                     f"{row['6144_ci_low']:.1%}–{row['6144_ci_high']:.1%} | {row['eval_truncated']} |")
    lines.extend(['', 'These are validation results once inspected for model selection. They are not directly interchangeable with scores from our historical game variant.', '',
                  'The published reference, when included, uses downloaded weights. Our training runs start randomly; importing reference weights is restricted to verification/evaluation.', '',
                  'Retention decisions remain pending human/agent review of learning curves and this comparison. No files are deleted automatically.', ''])
    (output / 'comparison.md').write_text('\n'.join(lines), encoding='utf-8')
    print(f'Report: {output / "comparison.md"}', flush=True)


if __name__ == '__main__':
    main()
