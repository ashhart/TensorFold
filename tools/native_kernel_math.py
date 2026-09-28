"""Make the Python MLX wheel's transcendental semantics explicit for native MLX.

The staged native MLX library and the Python 0.32.2 wheel choose different
defaults for unqualified Metal math functions. Deliberately fast calls are unchanged.
Used by export and fixture identification, never to compute expected outputs.
"""
import re


def explicit_math(source):
    return re.sub(r"metal::(exp|exp2|log|log2|sqrt|rsqrt|pow)\(", r"metal::precise::\1(", source)
