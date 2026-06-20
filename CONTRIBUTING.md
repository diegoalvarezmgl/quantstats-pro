# Contributing to QuantStats Pro

Thank you for your interest in contributing.

## Before you start

- Search [existing issues](https://github.com/diegoalvarezmgl/quantstats-pro/issues) to avoid duplicates
- For large changes, open an issue first to discuss approach

## Development setup

```bash
git clone git@github.com:diegoalvarezmgl/quantstats-pro.git
cd quantstats-pro
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

## Pull request process

1. Fork the repository
2. Create a branch from `main`: `fix/short-description` or `feat/short-description`
3. Make your changes with tests for bugfixes
4. Ensure `pytest` passes
5. Update `CHANGELOG.md` for user-visible changes
6. Open a PR against `main` with a clear summary and test plan

## Code guidelines

- Minimal, focused diffs — one logical change per PR
- Match existing conventions in `quantstats/`
- Every bugfix needs a regression test
- Do not break the drop-in API (`import quantstats as qs`) without discussion

## Attribution

QuantStats Pro is a fork of [QuantStats](https://github.com/ranaroussi/quantstats) by Ran Aroussi (Apache 2.0). Please respect upstream attribution in commits that port fixes.
