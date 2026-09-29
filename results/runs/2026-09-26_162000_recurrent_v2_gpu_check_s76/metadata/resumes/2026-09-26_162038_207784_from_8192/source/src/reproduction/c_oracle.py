"""Differential checks against the locally downloaded, pinned upstream C engine.

The generated translation unit omits rendering, supplies the same random stream
to both engines, and intercepts automatic reset so terminal states can be compared.
No upstream source or compiler binary is redistributed with this project.
"""

import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

from run_layout import PROJECT_ROOT
from .environment import ENGINE_REVISION, NativeBatch


COMMIT = "9527295de72333c91592fff45e8ccbcabd6ba80d"
HEADER_SHA256 = "eb6601f1a6bc2b362c1080ecdddeecc46ad28ef33d5033f43939b6510e094cc5"
WRAPPER = r'''
__declspec(dllexport) void oracle_step(
    unsigned char *board, long long *bag, long long *meta,
    unsigned long long *rng, int action, int heuristic, float snake,
    float scale, float *reward, int *done, unsigned char *obs) {
    Game g = {0};
    unsigned char terminal = 0;
    memcpy(g.grid, board, 16);
    for (int j=0; j<3; j++) g.bag[j] = bag[j];
    g.next_box=meta[0]; g.next_triplet=meta[1]; g.moves_made=meta[2];
    g.tick=meta[3]; g.bonus=meta[4]; g.lifetime_max_tile=meta[5];
    g.max_tile=meta[6]; g.is_scaffolding_episode=meta[7];
    g.max_episode_ticks=meta[8];
    g.stop_at_65536=true; g.use_heuristic_rewards=heuristic;
    g.snake_reward_weight=snake; g.reward_scaler=scale;
    g.actions=&action; g.rewards=reward; g.terminals=&terminal;
    g.observations=obs; g.grid_changed=true;
    for (int j=0; j<16; j++) {
        g.empty_count += board[j]==0;
        g.score += (int)piece_scores[board[j]];
    }
    update_observations(&g);
    oracle_rng=*rng;
    c_step(&g);
    *rng=oracle_rng;
    *done=terminal;
    memcpy(board, g.grid, 16);
    for (int j=0; j<3; j++) bag[j]=g.bag[j];
    meta[0]=g.next_box; meta[1]=obs[16]; meta[2]=g.moves_made;
    meta[3]=g.tick; meta[4]=g.bonus; meta[5]=g.lifetime_max_tile;
    meta[6]=g.max_tile; meta[7]=g.is_scaffolding_episode;
    meta[8]=g.max_episode_ticks;
}
'''


def build_oracle(header):
    import ziglang

    original = header.read_text(encoding="utf-8")
    if hashlib.sha256(header.read_bytes()).hexdigest() != HEADER_SHA256:
        raise ValueError("Upstream header differs from the audited source; review it before changing the pin")
    # Keep the upstream simulation functions verbatim. Only the render-only
    # dependency and the terminal autoreset boundary are replaced.
    core = original.split("// Rendering optimizations\nvoid c_render", 1)[0]
    if core == original:
        raise ValueError("Unexpected upstream source layout")
    core = core.replace('#include "raylib.h"',
                        "typedef struct { unsigned char r,g,b,a; } Color;")
    core = core.replace("void c_step(Game* game) {",
                        "void oracle_terminal_reset(Game* game) {}\n"
                        "#define c_reset oracle_terminal_reset\nvoid c_step(Game* game) {")
    prefix = """#include <stdlib.h>
#include <stdint.h>
static uint64_t oracle_rng;
static uint64_t oracle_rand(void) {
    uint64_t x=oracle_rng;
    x ^= x >> 12; x ^= x << 25; x ^= x >> 27;
    oracle_rng=x;
    return x * UINT64_C(2685821657736338717);
}
#define rand oracle_rand
"""
    source = prefix + core + "\n#undef c_reset\n" + WRAPPER
    digest = hashlib.sha256(source.encode()).hexdigest()
    directory = PROJECT_ROOT / ".external" / "c_oracle" / digest[:16]
    directory.mkdir(parents=True, exist_ok=True)
    dll = directory / "oracle.dll"
    if not dll.exists():
        translation = directory / "oracle.c"
        translation.write_text(source, encoding="utf-8")
        compiler = Path(ziglang.__file__).parent / "zig.exe"
        env = dict(os.environ, ZIG_GLOBAL_CACHE_DIR=str(directory / "cache"),
                   ZIG_LOCAL_CACHE_DIR=str(directory / "local_cache"))
        subprocess.run([str(compiler), "cc", "-O2", "-shared", str(translation),
                        "-o", str(dll)], check=True, env=env,
                       creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0)
    lib = ctypes.CDLL(str(dll))
    lib.oracle_step.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_int] * 2 + [ctypes.c_float] * 2 + [ctypes.c_void_p] * 3
    lib.oracle_step.restype = None
    return lib, hashlib.sha256(header.read_bytes()).hexdigest(), digest


def validate(header, cases=10000):
    lib, source_hash, harness_hash = build_oracle(header)
    generator = np.random.default_rng(20260926)
    env = NativeBatch(1)
    checked = 0
    terminals = invalid = bonuses = rank_limit = 0
    for case in range(cases):
        # Natural trajectories plus deliberately difficult high-rank states.
        if case % 20 == 0:
            env.meta[0, 5] = 0
            env.reset([0], [case])
        if case % 20 >= 10:
            env.board[0] = generator.integers(0, 7 + case % 9, 16, dtype=np.uint8)
            if case % 20 == 19:
                env.board[0, :4] = [15, 15, 0, 0]
            env.meta[0, 6] = env.board[0].max()
            env.meta[0, 5] = generator.integers(0, 17)
            env.meta[0, 7] = case % 2
            env.meta[0, 3] = 999 if case % 3 == 0 else 0
            env.meta[0, 8] = 1000
            env.meta[0, 2] = 20 if case % 2 else 0
            env.meta[0, 4] = 0
        action = int(generator.integers(4))
        env.heuristic = bool(case % 2)
        env.snake_weight = 0.0001 if case % 3 else 0.0
        env.reward_scaler = (0.5, 1.0, 2.0)[case % 3]
        board, bag, meta, rng = (getattr(env, name).copy() for name in ("board", "bag", "meta", "rng"))
        reward = np.zeros(1, np.float32)
        done = np.zeros(1, np.int32)
        obs = np.zeros((1, 21), np.uint8)
        lib.oracle_step(board.ctypes.data, bag.ctypes.data, meta.ctypes.data, rng.ctypes.data,
                        action, env.heuristic, env.snake_weight, env.reward_scaler,
                        reward.ctypes.data, done.ctypes.data, obs.ctypes.data)
        actual_obs, actual_reward, actual_done, stats = env.step([action])
        for name, expected in (("board", board), ("bag", bag), ("meta", meta), ("rng", rng)):
            np.testing.assert_array_equal(getattr(env, name), expected, err_msg=f"case {case}: {name}")
        np.testing.assert_array_equal(actual_obs, obs, err_msg=f"case {case}: observation")
        np.testing.assert_allclose(actual_reward, np.clip(reward, -1, 1), atol=1e-6, rtol=1e-6,
                                   err_msg=f"case {case}: reward")
        np.testing.assert_array_equal(actual_done, done.astype(bool), err_msg=f"case {case}: terminal")
        checked += 1
        terminals += int(actual_done[0])
        invalid += int(stats[0, 4])
        bonuses += int(meta[0, 0] >= 4)
        rank_limit += int(meta[0, 6] >= 16)
        if actual_done[0]:
            env.reset([0], [case + 100000])
    return {"passed": True, "engine_revision": ENGINE_REVISION, "cases": checked,
            "terminal_cases": terminals, "invalid_cases": invalid, "bonus_cases": bonuses,
            "rank_limit_cases": rank_limit,
            "upstream_commit": COMMIT, "header_sha256": source_hash, "harness_sha256": harness_hash,
            "validator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "environment_sha256": hashlib.sha256(Path(__file__).with_name("environment.py").read_bytes()).hexdigest(),
            "scope": "One-step and short trajectory parity: board, deck, preview, memory, RNG, reward, termination",
            "exclusions": ["C reset/curriculum initialization bugs are deliberately not reproduced",
                           "Shared controlled RNG replaces C rand(); identical numeric seeds across engines are not claimed",
                           "Optional endgame-only and sparse-reward modes are not implemented"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--header", type=Path, default=PROJECT_ROOT / ".external/threes-web/threes.h")
    parser.add_argument("--cases", type=int, default=10000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = validate(args.header, args.cases)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
