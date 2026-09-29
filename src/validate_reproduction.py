"""Archive numeric ONNX equivalence and a short native simulator benchmark."""

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from reproduction.environment import ReproductionVecEnv
from reproduction.policy import verify_onnx
from run_layout import PROJECT_ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    weights = PROJECT_ROOT / 'external_models/kiok/onnx_model.onnx'
    results = {'weights_sha256': hashlib.sha256(weights.read_bytes()).hexdigest(),
               'torch': torch.__version__, 'cuda': torch.cuda.is_available(),
               'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
               'equivalence': [verify_onnx(weights, 'cpu')]}
    if torch.cuda.is_available():
        results['equivalence'].append(verify_onnx(weights, 'cuda'))
    env = ReproductionVecEnv(128, 75)
    env.reset()
    rng = np.random.default_rng(75)
    env.step(rng.integers(0, 4, 128))  # Exclude JIT compilation.
    start = perf_counter()
    for _ in range(200):
        env.step(rng.integers(0, 4, 128))
    results['native_random_transitions_per_second'] = 25600 / (perf_counter() - start)
    env.close()
    results['scope'] = 'Engineering validation only: no policy training or performance claim.'
    output = args.output or PROJECT_ROOT / 'results/references' / datetime.now().strftime('%Y-%m-%d_%H%M%S_reproduction_validation.json')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2), encoding='utf-8')
    print(json.dumps(results, indent=2))
    print(f'Evidence: {output}')


if __name__ == '__main__':
    main()
