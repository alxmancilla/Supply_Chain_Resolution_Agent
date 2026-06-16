"""Evaluation harness for the Supply Chain Resolution Agent.

Two run modes:

- `fast` (default) wires fakes from `tests/fakes.py` into the protocol
  slots so metrics can run in CI without Atlas or any API key.
- `live` exercises the real Atlas-backed stack \u2014 used to regenerate the
  committed baseline file.

The harness is intentionally small: three datasets, three metrics, one
runner. Adding a new metric means adding a `*.py` to `evals/metrics/`
and a list entry in `runner.py`.
"""
