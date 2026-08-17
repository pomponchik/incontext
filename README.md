<details>
  <summary>Project status</summary>

[![Tests](https://github.com/pomponchik/incontext/actions/workflows/tests_and_coverage.yml/badge.svg?branch=develop)](https://github.com/pomponchik/incontext/actions/workflows/tests_and_coverage.yml)
[![Hermes e2e](https://github.com/pomponchik/incontext/actions/workflows/hermes_e2e.yml/badge.svg?branch=develop)](https://github.com/pomponchik/incontext/actions/workflows/hermes_e2e.yml)
[![Python versions](https://img.shields.io/pypi/pyversions/incontext.svg)](https://pypi.org/project/incontext/)
[![PyPI version](https://badge.fury.io/py/incontext.svg)](https://pypi.org/project/incontext/)
[![Checked with mypy](https://www.mypy-lang.org/static/mypy_badge.svg)](https://mypy-lang.org/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

</details>

![incontext logo](https://raw.githubusercontent.com/pomponchik/incontext/develop/docs/assets/logo.svg)

`incontext` is a Hermes Agent plugin that gives each LLM request exactly the
output budget still available below Hermes' context-compression boundary. It
prevents a fixed, oversized `max_tokens` value from consuming the input space
where Hermes must still be able to compress the conversation.

The plugin reads the effective compression window from the installed Hermes
`ContextCompressor`, asks the serving vLLM instance to tokenize the exact
provider-visible prompt, and applies:

```text
max_tokens = max(1, compression_window - prompt_tokens)
```

This addresses the same output-budget arithmetic discussed in
[NousResearch/hermes-agent#38652](https://github.com/NousResearch/hermes-agent/issues/38652).

## Installation

Once a release is available on PyPI, install and enable the package using the
same plugin name, `incontext`:

```bash
python -m pip install incontext
hermes plugins enable incontext
```

Until the first PyPI release, install the reviewed `develop` revision:

```bash
python -m pip install 'git+https://github.com/pomponchik/incontext.git@develop'
hermes plugins enable incontext
```

Restart the long-running Hermes gateway after installing or upgrading the
Python package. Hermes discovers it through the official
`hermes_agent.plugins` entry-point group; no source file has to be copied into
`$HERMES_HOME/plugins`.

## Configuration

`INCONTEXT_TOKENIZER_URL` is required and must point to the `/tokenize`
endpoint of the same vLLM model Hermes uses:

```bash
export INCONTEXT_TOKENIZER_URL='https://inference.example/tokenize'
export INCONTEXT_TOKENIZER_TIMEOUT_SECONDS='30'
export INCONTEXT_TOKENIZER_USER_AGENT='incontext/0.1'
export INCONTEXT_FALLBACK_MARGIN_TOKENS='1024'
```

Hermes' `model.default`, `model.context_length`, and `compression.threshold`
remain the source of truth. The plugin constructs Hermes' installed
`ContextCompressor` and uses its resolved `threshold_tokens`; it does not copy
version-sensitive threshold arithmetic.

The optional variables are:

| Variable | Default | Meaning |
|---|---:|---|
| `INCONTEXT_TOKENIZER_TIMEOUT_SECONDS` | `30` | `/tokenize` request timeout |
| `INCONTEXT_TOKENIZER_USER_AGENT` | `incontext/0.1` | HTTP user agent |
| `INCONTEXT_FALLBACK_MARGIN_TOKENS` | `1024` | Extra reserve only when exact tokenization fails |
| `INCONTEXT_COMPRESSION_WINDOW_TOKENS` | unset | Explicit emergency override for the resolved Hermes boundary |

The former `HERMES_VLLM_TOKENIZER_*` and
`HERMES_DYNAMIC_BUDGET_FALLBACK_MARGIN_TOKENS` names are accepted as migration
aliases. New deployments should use the `INCONTEXT_*` names.

## Safety properties

- vLLM applies its real chat template to messages, tools, and
  `chat_template_kwargs`; local tokenizer approximations are not used.
- The returned `max_model_len` must equal Hermes' configured context length.
- `max_tokens`, `max_completion_tokens`, and `max_output_tokens` are normalized
  to one unambiguous `max_tokens` field.
- The incoming request is copied and never mutated.
- Exact counts use a bounded, thread-safe cache.
- If `/tokenize` fails, Hermes' own rough estimator is used with an additional
  safety margin. If both counters fail, the middleware leaves the request
  unchanged instead of taking Hermes down.
- Logs contain counts and exception types, never prompts, credentials, or raw
  provider errors.

The tokenizer endpoint sees the prompt content by design. Run it on a trusted
network path and use the same access controls as the inference endpoint.

## Development

The repository layout follows the conventions used by
[`pristan`](https://github.com/mutating/pristan): strict typing and linting,
cross-platform CI, full branch coverage, issue templates, build validation,
and a main-only trusted-publishing workflow.

```bash
python -m venv .venv
.venv/bin/pip install -r requirements_dev.txt
.venv/bin/pip install -e .

.venv/bin/ruff format --check .
.venv/bin/ruff check .
.venv/bin/mypy incontext
.venv/bin/coverage run -m pytest tests/units
.venv/bin/coverage report -m
.venv/bin/python -m build
.venv/bin/twine check dist/*
```

Unit tests live in `tests/units` and enforce 100% line and branch coverage—the
maximum possible value. The test suite also uses property-based checks for the
budget invariants. `tests/e2e` installs the built artifact into real Hermes
Agent environments and exercises entry-point discovery plus the actual
`llm_request` middleware pipeline.

The release workflow runs only after a push to `main`. Work on `develop` never
publishes to PyPI.

## License

MIT
