# Lossless NVFP4 scale planes

`nvfp4_csf.Plane` reads `byte-window4-fixed-stream-u24-exceptions/1` storage.
It preserves the original scale bytes, including signed zeros and NaN encodings;
it performs no floating-point conversion or requantization.

Each 16-row slab contains 16 byte bases, followed by row-major pairs of four-bit
offsets. The even column occupies the low nibble. A decoded byte is its row base
plus its offset. Bases cannot exceed 240. Exceptions are little-endian uint32
words: the low 24 bits name a row-major position, and the high byte replaces its
value. Positions must be unique, sorted, and inside the logical plane.

Rows must be a positive multiple of 16, columns a positive multiple of two, and
the plane can contain at most 2^24 bytes. Stream extents must match exactly.
`Plane.init` validates the entire plane before returning a borrowed view. The
caller keeps the streams alive and immutable while that view is used.

```zig
const plane = try core.nvfp4_csf.Plane.init(rows, columns, fixed_bytes, exception_bytes);
try plane.decode(rank_scales, row_begin, row_count, column_begin, column_count);
```

`decode` writes exactly the selected rectangle in row-major order, with no heap
allocation. Both row and column partitions can end inside a storage slab or a
packed byte. Invalid output extents and overlap with either borrowed stream are
rejected before writing. Sorted exceptions are searched per selected row, so a
rank does not need to materialize another rank's scales.

This is a host format primitive. It does not change checkpoint admission,
projection arithmetic, CUDA kernels, or the runtime capability advertisement.
Callers must validate safetensors dtypes/shapes and select this exact codec before
constructing the view.

The public format reference is the [byte-window implementation in b12x](https://github.com/local-inference-lab/b12x/blob/2cc7f66aaef76a9c7cbb660a1efb865cf3f26c88/b12x/_lib/quant/nvfp4_csf.py).
Tests use independently generated synthetic source bytes, handwritten slab
examples, every replacement byte, both partition axes, and malformed streams.
