# Contributing to KVWeave

KVWeave is an experimental research preview. Contributions should help test the
project's storage/indexing hypothesis without presenting the reference code as
a production inference runtime or a stable API.

## Development setup

The repository pins Python, Pants, and the Python dependency lock through Mise:

```bash
mise install
mise run lint
mise run test
mise run package
```

Python 3.11 or newer is required. Pants-based development currently uses the
repository's pinned CPython 3.11 toolchain. Use the wheel built under `dist/`
when checking package installation independently of the repository build
graph:

```bash
python -m venv /tmp/kvweave-wheel-check
source /tmp/kvweave-wheel-check/bin/activate
python -m pip install dist/kvweave-0.0.1-py3-none-any.whl
python -c "from importlib.metadata import version; print(version('kvweave'))"
```

The Pants environment pins validated dependency versions in
`3rdparty/python/constraints.txt` and records the complete transitive graph in
`3rdparty/python/default.lock`. Update both intentionally with:

```bash
mise exec -- pants generate-lockfiles --resolve=python-default
```

Do not remove required transitive packages from the resolve to reduce its
size; select a compatible upstream distribution or change the environment
materialization strategy instead.

## Running tests and experiments

The default suite is offline and excludes tests marked `model_download`.
It includes a tiny, randomly initialized GPT-NeoX model that checks the full
decode loop without downloading weights or a tokenizer. The `test` extra
therefore includes the already pinned Transformers dependency:

```bash
mise run test
```

Opt-in Pythia tests require the optional model dependency and download the
pinned model/tokenizer if they are not already cached:

```bash
mise exec -- pants test tests/integration/test_pythia_real_model.py -- -m model_download
mise exec -- pants test tests/integration/test_pythia_decode.py -- -m model_download
```

Benchmarks are manual research workflows, not ordinary pull-request tests. See
[benchmarks/README.md](benchmarks/README.md) for exact commands and evidence
boundaries. Do not commit generated files under `benchmarks/results/`.

CI also checks the installed wheel's public retrieval path and compares its
metadata with `pyproject.toml`, including dependencies and extras. Keep the
Pants `BUILD` distribution metadata synchronized with that table. To repeat
the wheel check, install the built wheel and its dependencies into a separate
virtual environment, then run from the repository root:

```bash
/path/to/wheel-environment/bin/python -I scripts/check_wheel.py --project pyproject.toml
```

The isolated interpreter prevents the source checkout or `PYTHONPATH` from
masking a missing wheel module. Repository tests continue to use Pants and its
locked dependency environment.

## Contribution principles

- Establish correctness before optimizing.
- Keep storage, retrieval/indexing, and model integration separate.
- Add tests for behavioral changes and regression tests for fixed bugs when
  practical.
- Make benchmarks deterministic and reproducible.
- Document every performance claim with its model/revision, hardware, dtype,
  context length, retrieval budget, batch size, configuration, baseline, and
  commit.
- Do not copy research code without first verifying a compatible license and
  preserving required notices.
- Attribute research-derived algorithms and distinguish independent work from
  derived or adapted source.

## Performance pull requests

A performance-related pull request must include:

- a before/after benchmark using the same workload and methodology;
- exact CPU/GPU/accelerator and memory details;
- model ID and immutable revision when model-dependent;
- context length, generated-token count, batch size, and retrieval budget;
- all changed index/kernel/runtime parameters;
- retrieval, attention, decode, and memory measurements relevant to the claim;
- quality and correctness regression checks, including full-budget controls;
  and
- limitations and any measurement noise or profiler perturbation.

Do not infer an end-to-end speedup from a component timing. Keep heavyweight
benchmark artifacts out of Git and link or summarize reproducible results in
the pull request.

## Research algorithm contributions

Before implementing a new research-derived strategy, record in
[docs/RESEARCH.md](docs/RESEARCH.md):

- the paper citation and stable paper URL;
- the official implementation and inspected revision;
- the upstream repository license and any uncertainty or file-level provenance
  caveats;
- whether the contribution is independent, derived, or adapted; and
- the precise subset of the paper/system being tested.

The contribution must include unit and correctness tests, a brute-force or
full-attention control, and a reproducible benchmark through the smallest
existing KVWeave boundary that can test the hypothesis.

## Pull requests

Keep each pull request focused. Explain behavioral, quality, provenance, and
benchmark impact in the pull-request template. By contributing, you agree that
your contribution is submitted under the repository's Apache-2.0 license unless
you explicitly state otherwise.
