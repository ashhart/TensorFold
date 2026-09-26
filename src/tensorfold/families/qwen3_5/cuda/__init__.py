"""Qwen3.8 dense on CUDA: the exact lane engine's kernels and forward in PyTorch and Triton.

Same contract as the Metal lane engine: every kernel on the verify path gives a row the same
bits whether it runs alone (serial decoding) or as one of up to 128 rows of a verify window, so
a drafted token is the token serial decoding would have produced on this machine. Bits differ
from the Mac's; serial is defined by these kernels.
"""
