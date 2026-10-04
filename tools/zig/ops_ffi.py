"""Bind raw CUDA operator launchers without a framework extension."""

import ctypes

P = ctypes.c_void_p
U = ctypes.c_uint64
I = ctypes.c_uint32


def pointer(tensor):
    return P(tensor.data_ptr()) if tensor is not None else P()


def bind(library, name, signature):
    function = getattr(library, name)
    function.argtypes, function.restype = signature, ctypes.c_int
    return function
