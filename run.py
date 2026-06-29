import argparse
import os
import pickle
import subprocess
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).parent.resolve()

UTILITY_CONFIGS = (
    ("ego-facebook", "attr", ("edge", "ggsd", "grum", "spectre")),
    ("ego-twitter", "attr", ("edge", "ggsd", "grum", "spectre")),
    ("elliptic", "attr", ("edge", "ggsd")),
    ("ego-facebook", "unattr", ("edge", "grum", "spectre")),
    ("ego-twitter", "unattr", ("edge", "grum", "spectre", "gran")),
    ("elliptic", "unattr", ("edge", "gran")),
)


def run_command(*args: str) -> None:
    """Run a Python command from the repository root."""
    subprocess.run([sys.executable, *args], cwd=ROOT_DIR, check=True)


def run_smoke_test() -> None:
    """Run one small end-to-end attack on bundled graph data."""
    if str(ROOT_DIR) not in sys.path:
        sys.path.insert(0, str(ROOT_DIR))

    from exps.graph_funcs import GraphSinglingOutExpConf, _attack_graph_single

    data_dir = ROOT_DIR / "exps" / "datasets" / "ego-twitter"
    with (data_dir / "ego-twitter_unattr.pickle").open("rb") as handle:
        reference = pickle.load(handle)
    with (data_dir / "edge_ego-twitter_unattr.pickle").open("rb") as handle:
        synthetic = pickle.load(handle)

    config = GraphSinglingOutExpConf(
        dataset="ego-twitter",
        model="edge",
        n_attacks=20,
        n_cols=2,
    )
    result = _attack_graph_single(
        config,
        reference["train"],
        synthetic,
        reference["control"],
    )

    if not 0 <= result.risk_value <= 1:
        raise RuntimeError(f"Unexpected privacy risk: {result.risk_value}")

    print(
        "Smoke test passed: "
        f"community singling-out risk={result.risk_value:.4f}, "
        f"95% CI=[{result.risk_ci_lwr:.4f}, {result.risk_ci_upr:.4f}]",
        flush=True,
    )


def process_results() -> None:
    run_command("exps/proc_graph_exps.py")


def clear_attack_results() -> None:
    attack_results_dir = ROOT_DIR / "exps" / "results"

    for fname in attack_results_dir.glob("*.csv"):
        fname.unlink(missing_ok=True)


def clear_utility_results() -> None:
    (ROOT_DIR / "exps" / "utility.csv").unlink(missing_ok=True)


def run_attacks() -> None:
    clear_attack_results()
    run_command("exps/run_graph_exps.py")
    process_results()


def run_utility() -> None:
    clear_utility_results()
    subprocess.run(
        ["g++", "-O2", "-std=c++11", "-o", "orca", "orca.cpp"],
        cwd=ROOT_DIR / "exps" / "orca",
        check=True)

    for dataset, data_type, models in UTILITY_CONFIGS:
        run_command(
            "exps/compute_utility.py",
            "--dataset",
            dataset,
            "--type",
            data_type,
            "--models",
            *models,
        )

    process_results()


def run(mode: str) -> None:
    os.chdir(ROOT_DIR)

    if mode == "quick":
        run_smoke_test()
        process_results()
    elif mode == "attacks":
        run_attacks()
    elif mode == "utility":
        run_utility()
    elif mode == "full":
        run_attacks()
        run_utility()

    print(f"Completed '{mode}'. Outputs are under exps/.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run and verify the SyntheGrAnon artifact evaluation.")
    parser.add_argument(
        "mode",
        nargs="?",
        default="quick",
        choices=("quick", "attacks", "utility", "full"),
        help="evaluation stage to run (default: quick)",
    )
    args = parser.parse_args()
    run(args.mode)


if __name__ == "__main__":
    main()
