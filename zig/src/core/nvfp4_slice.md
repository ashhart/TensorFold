# Rank-local NVFP4 checkpoint reads

`nvfp4_slice.read(checkpoint, allocator, module, rows, inputs)` reads a logical
rectangle without expanding the whole projection. The result owns its packed
FP4 code bytes and row-major E4M3 scale bytes; call `deinit` with the same
allocator. `global` preserves the stored FP32 tensor scale. There is no numeric
conversion or requantization.

The source weight is U8 `[N, K/2]`. Plain scales are F8_E4M3 `[N, K/16]`.
Alternatively, the source may contain the fixed U8 and U32 exception tensors
described by [the CSF codec](nvfp4_csf.md). Both representations cannot be present
at once, and a partial compressed pair is refused. Tensor scale and optional
input calibration must be FP32 scalars. Calibration is consumed as checkpoint
metadata; this reader does not implement an activation quantizer.

Row and input spans are half-open, nonempty and bounded by the original tensor.
Input cuts must align to groups of 16. Row cuts can cross a compressed 16-row
slab. Source geometry, dtype, byte extents and complete compressed streams are
validated before allocation. Allocation failures release every intermediate
buffer. The result contains only the requested code and scale rectangle.

`dimensions(source)` and `slice(allocator, source, rows, inputs)` also accept
borrowed tensor views for callers with their own checkpoint mapping. The caller
owns those mappings; the returned buffers are independent of them.

Tests compare plain and compressed row/column partitions with the independent
FP64 decoder, exercise allocator failures and malformed inputs, and read a real
synthetic safetensors file through the checkpoint reader. These are host format
checks, not tensor-parallel execution or GPU qualification.
