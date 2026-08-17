<details>
  <summary>ⓘ</summary>

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

The package supports CPython 3.8 through 3.15, including free-threaded Python
3.14. Unit tests, formatting, linting, and static type checks run across the
complete version matrix in CI.

The plugin reads the effective compression window from the installed Hermes
`ContextCompressor`, asks the selected inference backend to tokenize the exact
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

The bundled `vllm` backend is selected by default. Its
`INCONTEXT_TOKENIZER_URL` setting is required and must point to the `/tokenize`
endpoint of the same vLLM model Hermes uses:

```bash
export INCONTEXT_BACKEND='vllm'
export INCONTEXT_TOKENIZER_URL='https://inference.example/tokenize'
export INCONTEXT_TOKENIZER_TIMEOUT_SECONDS='30'
export INCONTEXT_TOKENIZER_USER_AGENT='incontext/0.0.1'
export INCONTEXT_FALLBACK_MARGIN_TOKENS='1024'
```

Environment variables are loaded through typed `skelet.Storage` fields backed
by ordered `skelet.EnvSource` instances. Primary `INCONTEXT_*` names take
precedence over the supported legacy aliases. Text normalization and blank
value rejection are implemented by the fields' native `conversion` and
`validation` rules.

Hermes' `model.default`, `model.context_length`, and `compression.threshold`
remain the source of truth. The plugin constructs Hermes' installed
`ContextCompressor` and uses its resolved `threshold_tokens`; it does not copy
version-sensitive threshold arithmetic.

The optional variables are:

| Variable | Default | Meaning |
|---|---:|---|
| `INCONTEXT_BACKEND` | `vllm` | Named `pristan` backend plugin |
| `INCONTEXT_TOKENIZER_TIMEOUT_SECONDS` | `30` | `/tokenize` request timeout |
| `INCONTEXT_TOKENIZER_USER_AGENT` | `incontext/0.0.1` | HTTP user agent |
| `INCONTEXT_FALLBACK_MARGIN_TOKENS` | `1024` | Extra reserve only when exact tokenization fails |
| `INCONTEXT_COMPRESSION_WINDOW_TOKENS` | unset | Explicit emergency override for the resolved Hermes boundary |

The former `HERMES_VLLM_TOKENIZER_*` and
`HERMES_DYNAMIC_BUDGET_FALLBACK_MARGIN_TOKENS` names are accepted as migration
aliases. New deployments should use the `INCONTEXT_*` names.

## Replacing the inference backend

The budgeting core depends only on the abstract `incontext.Backend` contract.
It has no import or construction dependency on vLLM. A backend supplies three
operations: its safe diagnostic `source`, exact `count(...)`, and
`clear_cache()`.

Backend implementations are named `pristan` plugins in the
`incontext.backends` entry-point group. The generic `skelet` environment has a
typed `backend` field whose default is `vllm`. At runtime incontext performs
the single named resolution directly:

```python
backend = backends[environment.backend].one()
```

The `incontext` distribution itself publishes the `vllm` entry point. Loading
that entry point imports `incontext.vllm_provider`, whose only responsibility is
to construct `VllmBackend`. All `/tokenize` payload rules, vLLM response fields,
context-length validation, transport settings, and caching live inside that
class rather than in the budgeting core.

A third-party distribution can provide another backend without changing
incontext. Its implementation subclasses the stable abstract contract and its
plugin module registers a provider under a new name:

```python
# acme_backend/plugin.py
from __future__ import annotations

from typing import Any

from incontext import Backend, backends


class AcmeBackend(Backend):
    @property
    def source(self) -> str:
        return "acme-tokenizer"

    def count(
        self,
        request: dict[str, Any],
        *,
        context_length: int,
    ) -> int:
        ...

    def clear_cache(self) -> None:
        ...


@backends.plugin("acme")
def provide_acme_backend() -> Backend:
    return AcmeBackend()
```

The third-party package makes that module discoverable in `pyproject.toml`:

```toml
[project.entry-points."incontext.backends"]
acme = "acme_backend.plugin"
```

After installing the package, select it through the same typed configuration
field and restart the Hermes process:

```bash
export INCONTEXT_BACKEND='acme'
```

Only the selected provider is instantiated. An unknown name fails `.one()`;
the unique slot rejects duplicate providers under the same name while loading
entry points. Startup therefore fails instead of choosing a backend implicitly.
Each backend owns and validates its backend-specific configuration; the generic
settings object contains only the compression-window and fallback-budget policy.

## Safety properties

- With the bundled backend, vLLM applies its real chat template to messages, tools, and
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
