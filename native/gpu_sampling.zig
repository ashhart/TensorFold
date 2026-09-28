//! Original TensorFold GPU sampler: fp32 arithmetic and 24-bit hash uniforms.
//! Selectable separately from the CPU f64 sampler.
const mx = @import("mlx.zig");
const src = @import("kernel_sources.zig");
const Sampling = @import("sampling.zig").Sampling;

pub fn sample(k: *mx.Kernels, s: *mx.Scope, logits: mx.Array, positions: []const i32, settings: Sampling, ids: ?mx.Array) !mx.Array {
    try settings.validate();
    const vocab = mx.dim(logits, -1);
    if (vocab <= 0 or positions.len == 0 or mx.c.mlx_array_size(logits) != positions.len * @as(usize, @intCast(vocab))) return error.InvalidSamplingShape;
    if (ids) |mapping| if (mx.dtype(mapping) != mx.c.MLX_UINT32 or mx.c.mlx_array_size(mapping) != @as(usize, @intCast(vocab))) return error.InvalidSamplingMapping;
    const rows: i32 = @intCast(positions.len);
    const x = try s.reshape(logits, &.{ rows, vocab });
    if (settings.temperature == 0) {
        const picked = try s.argmax(x);
        return if (ids) |mapping| s.take(mapping, picked, 0) else picked;
    }
    const seeds = try mx.allocator.alloc(u32, positions.len * 2);
    defer mx.allocator.free(seeds);
    for (positions, 0..) |position, i| {
        if (position < 0) return error.InvalidSamplingPosition;
        seeds[2 * i] = @truncate(settings.seed);
        seeds[2 * i + 1] = @truncate(settings.seed >> 32);
    }
    const cfg = [_]f32{ @floatCast(1 / @max(settings.temperature, 1e-6)), @floatCast(settings.top_p), 20 };
    const cap: u32 = @intCast(@min(settings.top_k, @as(usize, @intCast(vocab))));
    // Upstream's sampler accepts independent settings for each row. The CLI
    // uses one setting, repeated with the same per-row layout.
    const configs = try mx.allocator.alloc(f32, positions.len * 3);
    defer mx.allocator.free(configs);
    const caps = try mx.allocator.alloc(u32, positions.len);
    defer mx.allocator.free(caps);
    for (caps, 0..) |*value, i| {
        @memcpy(configs[i * 3 ..][0..3], &cfg);
        value.* = cap;
    }
    var inputs = [_]mx.Array{
        x,                                                  try s.data(seeds.ptr, &.{rows * 2}, mx.c.MLX_UINT32),
        try s.cast(try s.ints(positions), mx.c.MLX_UINT32), try s.data(configs.ptr, &.{ rows, 3 }, mx.f32t),
        try s.data(caps.ptr, &.{rows}, mx.c.MLX_UINT32),    ids orelse mx.empty,
    };
    return (try k.run(s, if (ids != null) src.gpu_sample_ids else src.gpu_sample, inputs[0..if (ids != null) @as(usize, 6) else 5], &.{ mx.ti("V", vocab), mx.ti("C", 1024) }, .{ 1024 * rows, 1, 1 }, .{ 1024, 1, 1 }, &.{.{ .shape = &.{rows}, .dtype = mx.c.MLX_UINT32 }}))[0];
}

pub fn topk(k: *mx.Kernels, s: *mx.Scope, x: mx.Array, count: i32) ![2]mx.Array {
    const vocab = mx.dim(x, -1);
    if (count < 1 or count > 64 or count > vocab) return error.InvalidTopK;
    const rows: i32 = @intCast(mx.c.mlx_array_size(x) / @as(usize, @intCast(vocab)));
    const data = try s.contiguous(try s.reshape(try s.cast(x, mx.bf16), &.{ rows, vocab }));
    const out = try k.run(s, src.radix_topk, &.{ data, try s.ints(&.{ vocab, count }) }, &.{ mx.ti("TPG", 1024), mx.ti("MAXK", 64), mx.ti("MAXT", 2048) }, .{ rows * 1024, 1, 1 }, .{ 1024, 1, 1 }, &.{ .{ .shape = &.{ rows, count }, .dtype = mx.i32t }, .{ .shape = &.{ rows, count }, .dtype = mx.f32t } });
    return .{ out[0], out[1] };
}
