"""Runs eval suites and emits JSON plus length-degradation curves.

Built and validated before any model code exists, because every later claim
in the repo depends on trusting this measurement path."""

from __future__ import annotations


def run_suite(*args, **kwargs):
    """Run an eval suite against a model and write results.json."""
    raise NotImplementedError("Milestone 0: the first thing to build.")
