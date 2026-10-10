"""Run live Qiskit work outside Streamlit's rerun threads, using JSON on stdin/stdout."""

import json
import sys

from tailrisk.experiments import demo


def main():
    settings = json.load(sys.stdin)
    print(json.dumps(demo(**settings), allow_nan=False))


if __name__ == "__main__":
    main()
