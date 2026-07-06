from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

from py_geo_fd import Stent


def run_once(config_path: Path) -> float:
    t0 = time.perf_counter()
    _ = Stent(str(config_path))
    return time.perf_counter() - t0


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark stent construction time.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("stent_config.json"),
        help="Path to stent JSON config file.",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=3,
        help="Number of benchmark repetitions.",
    )
    args = parser.parse_args()

    if args.runs < 1:
        raise ValueError("--runs must be >= 1")

    durations = [run_once(args.config) for _ in range(args.runs)]

    print("Benchmark results")
    print(f"Config: {args.config}")
    print(f"Runs: {args.runs}")
    print(f"Mean: {statistics.mean(durations):.3f} s")
    if len(durations) > 1:
        print(f"Std:  {statistics.pstdev(durations):.3f} s")
    print(f"Min:  {min(durations):.3f} s")
    print(f"Max:  {max(durations):.3f} s")


if __name__ == "__main__":
    main()
