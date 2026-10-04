"""Assemble a kernel's full Metal source and function name the way mx.fast.metal_kernel builds them, without MLX."""

from __future__ import annotations

import re
from dataclasses import dataclass

# Metal type names of array dtypes in generated signatures
TYPE_NAMES = {
    "bool": "bool", "uint8": "uint8_t", "uint16": "uint16_t", "uint32": "uint32_t", "uint64": "uint64_t",
    "int8": "int8_t", "int16": "int16_t", "int32": "int32_t", "int64": "int64_t", "float16": "float16_t",
    "bfloat16": "bfloat16_t", "float32": "float", "float64": "double", "complex64": "complex64_t",
}

# Thread attributes a body may name, in the order the signature lists them, with their types
ATTRIBUTES = (
    ("dispatch_quadgroups_per_threadgroup", "uint"), ("dispatch_simdgroups_per_threadgroup", "uint"),
    ("dispatch_threads_per_threadgroup", "uint3"), ("grid_origin", "uint3"), ("grid_size", "uint3"),
    ("quadgroup_index_in_threadgroup", "uint"), ("quadgroups_per_threadgroup", "uint"),
    ("simdgroup_index_in_threadgroup", "uint"), ("simdgroups_per_threadgroup", "uint"),
    ("thread_execution_width", "uint"), ("thread_index_in_quadgroup", "uint"), ("thread_index_in_simdgroup", "uint"),
    ("thread_index_in_threadgroup", "uint"), ("thread_position_in_grid", "uint3"),
    ("thread_position_in_threadgroup", "uint3"), ("threadgroup_position_in_grid", "uint3"),
    ("threadgroups_per_grid", "uint3"), ("threads_per_grid", "uint3"), ("threads_per_simdgroup", "uint"),
    ("threads_per_threadgroup", "uint3"),
)

CONSTANT_BELOW = 8          # inputs with fewer elements are bound in the constant address space

# Our preamble: the type names generated signatures use (MLX's own preamble is not copied or needed)
PREAMBLE = """#include <metal_stdlib>
using namespace metal;
typedef bfloat bfloat16_t;
typedef half float16_t;
"""


@dataclass(frozen=True)
class Arg:
    """A kernel input or output: its name, dtype name (as mlx.core spells it), element count and rank."""

    name: str
    dtype: str
    size: int = CONSTANT_BELOW
    ndim: int = 1


def _template_value(value: object) -> str:
    if isinstance(value, bool):
        return str(int(value))
    if isinstance(value, int):
        return str(value)
    return TYPE_NAMES[str(value)]


def _template_param(value: object) -> str:
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    return "typename"


def function_name(name: str, inputs: list[Arg], outputs: list[Arg], template: list[tuple[str, object]]) -> str:
    """custom_kernel_<name>[_<template values>]_<input types and passing>_<output types>."""

    out = "custom_kernel_" + name
    if template:
        out += "_" + "_" + "_".join(_template_value(v) for _, v in template)
    for arg in inputs:
        out += "_" + TYPE_NAMES[arg.dtype]
        out += "s" if arg.ndim == 0 else ("c" if arg.size < CONSTANT_BELOW else "")
    for arg in outputs:
        out += "_" + TYPE_NAMES[arg.dtype]
    return out


def kernel_text(name: str, inputs: list[Arg], outputs: list[Arg], source: str, header: str = "",
                template: list[tuple[str, object]] | None = None, atomic_outputs: bool = False) -> tuple[str, str]:
    """(function name, the kernel's text after the preamble): header, generated signature, body, instantiation."""

    template = list(template or [])
    fname = function_name(name, inputs, outputs, template)
    text = header
    if template:
        text += "template <" + ", ".join(f"{_template_param(v)} {k}" for k, v in template) + ">\n"
    text += f"[[kernel]] void {fname}(\n"
    attrs = [f"  {kind} {attr} [[{attr}]]" for attr, kind in ATTRIBUTES if attr in source]
    index = 0
    for arg in inputs:
        space = "constant" if arg.size < CONSTANT_BELOW else "device"
        ref = "&" if arg.ndim == 0 else "*"
        text += f"  const {space} {TYPE_NAMES[arg.dtype]}{ref} {arg.name} [[buffer({index})]],\n"
        index += 1
        if arg.ndim > 0:
            for suffix, kind in (("_shape", "const constant int*"), ("_strides", "const constant int64_t*"),
                                 ("_ndim", "const constant int&")):
                if arg.name + suffix in source:
                    text += f"  {kind} {arg.name}{suffix} [[buffer({index})]],\n"
                    index += 1
    for arg in outputs:
        kind = f"atomic<{TYPE_NAMES[arg.dtype]}>" if atomic_outputs else TYPE_NAMES[arg.dtype]
        text += f"  device {kind}* {arg.name} [[buffer({index})]]"
        # the separator test counts buffer slots, shape slots included, as mx.fast.metal_kernel does
        text += ",\n" if index < len(inputs) + len(outputs) - 1 or attrs else ") {\n"
        index += 1
    if attrs:
        text += ",\n".join(attrs) + ") {\n"
    text += source + "\n}\n"
    if template:
        args = fname + "<" + ", ".join(_template_value(v) for _, v in template) + ">"
        text += f'\ntemplate [[host_name("{fname}")]] [[kernel]] decltype({args}) {args};\n'
    return fname, text


_BLOCK = re.compile(r"/\*.*?\*/", re.S)
_LINE = re.compile(r"//.*")


def strip_comments(text: str) -> str:
    """Metal text without its comments and the lines they leave empty: comments never change codegen."""

    out = []
    for line in _BLOCK.sub("", text).split("\n"):
        kept = _LINE.sub("", line).rstrip()
        if kept or not line.strip():
            out.append(kept)
    return "\n".join(out)
