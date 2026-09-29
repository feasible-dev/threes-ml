"""Watch or override a recurrent policy in the audited Threes simulator (CPU only)."""

import argparse
from datetime import datetime
from pathlib import Path

from inspector.session import CpuPolicy, DEFAULT_MODEL, InspectionSession, ROOT
from inspector.window import run_window


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="Published .onnx or our recurrent .zip checkpoint")
    parser.add_argument("--seed", type=int, default=730000, help="Inspection seed; reserved final-test range is refused")
    parser.add_argument("--speed", type=float, default=3.0, help="Autoplay moves/s, 0.25..60")
    parser.add_argument("--autoplay", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-frames", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--screenshot", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not .25 <= args.speed <= 60:
        parser.error("Speed must lie in [0.25,60]")
    if not 0 <= args.seed < 2**63 or 900000 <= args.seed <= 909999:
        parser.error("Choose an inspection seed below 2**63 outside reserved range 900000..909999")
    print("Loading policy on CPU; the training GPU is not used.", flush=True)
    policy = CpuPolicy(args.model)
    label = "published" if args.model.resolve() == DEFAULT_MODEL.resolve() else args.model.stem
    output = args.output or ROOT / "results/inspections" / (datetime.now().strftime("%Y-%m-%d_%H%M%S_%f") + f"_{label}_s{args.seed}")
    session = InspectionSession(policy, args.seed, output)
    print(f"Model: {policy.label}\nInspection log: {output}\nArrows override; Space plays/pauses; N steps; U undoes; E exports.", flush=True)
    try:
        run_window(session, args.speed, args.autoplay, args.max_frames, args.screenshot)
    finally:
        session.close()


if __name__ == "__main__":
    main()
