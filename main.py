"""Command-line entry point. Run `python main.py --help` for the available experiments."""

import argparse
from pathlib import Path

from tailrisk.experiments import ROOT, benchmark, demo, prepare_data, studies, write_json


def main():
    parser = argparse.ArgumentParser(
        description="Portfolio tail risk with Qiskit amplitude estimation"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("prepare-data", help="Download public data and fit the two models")
    demo_parser = commands.add_parser("demo", help="Run a verified finite-shot IQAE demonstration")
    demo_parser.add_argument("--family", choices=["gaussian", "student"], default="student")
    demo_parser.add_argument("--threshold", type=float, default=0.10)
    demo_parser.add_argument("--bits", type=int, choices=[2, 3, 4], default=3)
    demo_parser.add_argument("--epsilon", type=float, default=0.01)
    bench_parser = commands.add_parser("benchmark", help="Run fixed-budget replicated experiments")
    bench_parser.add_argument("--replicates", type=int, default=30)
    bench_parser.add_argument("--levels", type=int, default=6)
    bench_parser.add_argument("--output", type=Path, default=ROOT / "results/benchmark")
    commands.add_parser("studies", help="Run grid, oracle, IQAE, VaR, ES, and noise studies")
    commands.add_parser("figures", help="Render figures from saved experiments")
    scale_parser = commands.add_parser("scaling", help="Matched N and scenario-size experiments")
    scale_parser.add_argument("--replicates", type=int, default=30)
    args = parser.parse_args()
    if args.command == "prepare-data":
        result = prepare_data()
        print(f"Fitted {result['ten_day_windows']} windows: {result['start']} to {result['end']}")
    elif args.command == "demo":
        result = demo(args.family, args.threshold, args.bits, args.epsilon)
        write_json(ROOT / "results/demo.json", result)
        estimate = result["quantum"]
        print(f"Exact grid tail: {result['exact_grid_tail']:.6f}")
        print(f"IQAE estimate:   {estimate['estimate']:.6f}")
        print(f"95% interval:    {estimate['confidence_interval']}")
        print(f"A/A-dagger calls: {estimate['queries']}; shots: {estimate['shots']}")
        print("Saved results/demo.json")
    elif args.command == "benchmark":
        benchmark(args.output, args.replicates, levels=args.levels)
    elif args.command == "studies":
        studies()
    elif args.command == "scaling":
        from tailrisk.scaling import matched_budget, scenario_scaling

        matched_budget()
        scenario_scaling(replicates=args.replicates)
    elif args.command == "figures":
        from tailrisk.figures import render_all

        render_all()


if __name__ == "__main__":
    main()
