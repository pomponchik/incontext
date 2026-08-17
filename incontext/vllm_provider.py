"""Bundled pristan provider for the vLLM backend."""

from .backend import Backend, backends
from .vllm import VllmBackend


@backends.plugin("vllm", unique=True)
def provide_vllm_backend() -> Backend:
    """Construct the bundled vLLM backend on demand."""

    return VllmBackend()
