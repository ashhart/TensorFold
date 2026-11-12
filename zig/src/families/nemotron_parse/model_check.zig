//! Development-only byte checks of actual model state at prompt cuts, shared forwards and suffix rollback.
const std = @import("std");
const hip = @import("hip");
const Compute = @import("compute.zig").Compute;
const decoder = @import("decoder.zig");
const config = @import("config.zig");
const Tensor = hip.ops.Tensor;

pub fn run(a: std.mem.Allocator, c: Compute, r: *const hip.Runtime, prefix: Tensor, image_rows: usize, ids: []const i32, baseline: *decoder.State) !void {
    const rows = image_rows + ids.len;
    if (ids.len == 0) return error.BadReferenceIds;
    var text = try hip.DeviceBuffer.fromHost(r, std.mem.sliceAsBytes(ids));
    defer text.free();
    var input = try hip.DeviceBuffer.alloc(r, rows * config.hidden * 2);
    defer input.free();
    const tensor: Tensor = .{ .ptr = input.base(), .kind = .bf16 };
    try c.ops.copyCols(prefix, config.hidden, 0, tensor.ptr, image_rows, config.hidden);
    try c.kernels.embed(c.stream, try c.weights.get("decoder.embed_tokens.weight"), text.base(), slice(tensor, image_rows), ids.len, config.hidden, 32);
    try c.stream.synchronize();
    var cut = try decoder.State.init(r, rows + 4);
    defer cut.deinit();
    var other = try decoder.State.init(r, rows + 4);
    defer other.deinit();
    var shared = try decoder.State.init(r, rows + 7);
    defer shared.deinit();
    var chunk: usize = 17;
    while (cut.len < rows) {
        c.ops.arena.reset();
        const n = @min(chunk, rows - cut.len);
        _ = try decoder.forward(c, &cut, slice(tensor, cut.len), n);
        chunk = if (chunk == 17) 511 else 256;
    }
    try c.stream.synchronize();
    try same(a, baseline, &cut, rows);
    std.debug.print("STATE: full/chunked prompt KV and hidden bytes agree ({d} rows)\n", .{rows});

    // A different final text token makes cross-stream KV contamination visible.
    const different: i32 = if (ids[ids.len - 1] == 243) 809 else 243;
    try text.upload(0, std.mem.asBytes(&different));
    var combined = try hip.DeviceBuffer.alloc(r, 2 * rows * config.hidden * 2);
    defer combined.free();
    try hip.raw.copy(combined, 0, tensor.ptr, rows * config.hidden * 2, c.stream.handle);
    try hip.raw.copy(combined, rows * config.hidden * 2, tensor.ptr, rows * config.hidden * 2, c.stream.handle);
    const combined_tensor: Tensor = .{ .ptr = combined.base(), .kind = .bf16 };
    try c.kernels.embed(c.stream, try c.weights.get("decoder.embed_tokens.weight"), text.base(), slice(combined_tensor, 2 * rows - 1), 1, config.hidden, 32);
    c.ops.arena.reset();
    _ = try decoder.forward(c, &other, slice(combined_tensor, rows), rows);
    try cut.keep(0);
    c.ops.arena.reset();
    _ = try decoder.forwardMany(c, &.{ .{ .state = &cut, .rows = rows }, .{ .state = &shared, .rows = rows } }, combined_tensor);
    try c.stream.synchronize();
    try same(a, baseline, &cut, rows);
    try same(a, &other, &shared, rows);
    std.debug.print("STATE: two distinct prompts share one dense forward and match solo KV/hidden bytes\n", .{});

    // Roll back inside a target window, then replay only its accepted suffix.
    try shared.keep(rows - 3);
    c.ops.arena.reset();
    _ = try decoder.forward(c, &shared, slice(combined_tensor, 2 * rows - 3), 3);
    try c.stream.synchronize();
    try same(a, &other, &shared, rows);
    std.debug.print("STATE: partial suffix rollback/replay preserves every KV and hidden byte\n", .{});
    c.ops.arena.reset();
}

fn slice(x: Tensor, start: usize) Tensor {
    return .{ .ptr = x.ptr + start * config.hidden * 2, .kind = .bf16 };
}

fn bytes(a: std.mem.Allocator, x: hip.DeviceBuffer, xo: usize, y: hip.DeviceBuffer, yo: usize, n: usize) !void {
    const left = try a.alloc(u8, n);
    defer a.free(left);
    const right = try a.alloc(u8, n);
    defer a.free(right);
    try x.download(xo, left);
    try y.download(yo, right);
    if (!std.mem.eql(u8, left, right)) {
        var different: usize = 0;
        for (left, right) |l, v| different += @intFromBool(l != v);
        std.debug.print("STATE mismatch: {d}/{d} bytes, offsets {d}/{d}\n", .{ different, n, xo, yo });
        return error.ModelStateMismatch;
    }
}

fn same(a: std.mem.Allocator, x: *const decoder.State, y: *const decoder.State, rows: usize) !void {
    if (x.len < rows or y.len < rows) return error.BadComparedState;
    try bytes(a, x.hidden, 0, y.hidden, 0, rows * config.hidden * 2);
    for (x.layers, y.layers, 0..) |xl, yl, layer| {
        for (0..config.kv_heads) |head| {
            const xo = head * x.total * config.head_dim * 2;
            const yo = head * y.total * config.head_dim * 2;
            bytes(a, xl.k, xo, yl.k, yo, rows * config.head_dim * 2) catch |err| {
                std.debug.print("STATE: K layer {d}, head {d}\n", .{ layer, head });
                return err;
            };
            try bytes(a, xl.v, xo, yl.v, yo, rows * config.head_dim * 2);
        }
    }
}
