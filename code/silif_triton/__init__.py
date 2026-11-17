"""Triton-accelerated SiLIF neuron (with a validated pure-torch reference)."""
from silif_triton.silif_neuron import SiLIFNodeTriton
from silif_triton.silif_triton_kernel import SiLIFMultiStepTriton, _HAS_TRITON
from silif_triton.silif_reference import (
    silif_scan_torch, silif_manual_backward, derive_alpha_beta)

__all__ = [
    "SiLIFNodeTriton", "SiLIFMultiStepTriton", "_HAS_TRITON",
    "silif_scan_torch", "silif_manual_backward", "derive_alpha_beta",
]
