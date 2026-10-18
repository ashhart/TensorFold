//! The CUDA H3 family as a C library for a host that shares its CUDA context: tf_h3_open, tf_h3_forward, tf_h3_close.
const std = @import("std");
const h3 = @import("cuda.zig");
const weights = @import("cuda_weights.zig");
const host = @import("cuda_host.zig");

const gpa = std.heap.c_allocator;

/// Input layout version: this library accepts a complete single-device sequence.
export fn tf_h3_abi_version() u32 {
    return 1;
}

/// The transformer at `path` (one safetensors file) on the GPU in its bf16; null on failure.
export fn tf_h3_open(path: [*:0]const u8) ?*h3.Model {
    return h3.Model.load(gpa, path) catch |e| {
        std.debug.print("tf_h3_open: {t}\n", .{e});
        return null;
    };
}

/// Failed completion retains the entire handle and reports it on stderr; retry only after context recovery.
export fn tf_h3_close(model: *h3.Model) void {
    _ = @import("cuda_load_retained.zig").recover();
    model.deinit();
}

/// hidden, inner, heads, mlp, blocks, the load's seconds in thousandths, and the weights' bits (16).
export fn tf_h3_shape(model: *const h3.Model, out: *[7]u32) void {
    out.* = .{ @intCast(model.hidden), @intCast(model.inner), @intCast(model.heads), @intCast(model.mlp), @intCast(model.blocks.len), @intFromFloat(model.loaded_seconds * 1000), 16 };
}

/// Every block over the packed stream, in place. Returns the seconds taken, or a negative number on failure.
export fn tf_h3_forward(model: *h3.Model, in: *const h3.Model.Inputs) f64 {
    return model.forward(in) catch |e| {
        std.debug.print("tf_h3_forward: {t}\n", .{e});
        return -1;
    };
}

/// One block projection (weights.Projection) from bf16 (rows, cols) on the device; 0, or -1 on a wrong block or shape.
export fn tf_h3_set_weight(model: *h3.Model, block: u32, which: c_int, w: u64, rows: u32, cols: u32) c_int {
    const projection = std.enums.fromInt(weights.Projection, which) orelse return -1;
    weights.set(model, block, projection, w, rows, cols) catch |e| {
        std.debug.print("tf_h3_set_weight: {t}\n", .{e});
        return -1;
    };
    return 0;
}

/// With `on`, a forward waits for the GPU after every stage and adds up where the time went.
export fn tf_h3_profile(model: *h3.Model, on: c_int) void {
    model.profile = on != 0;
    model.spent = @splat(0);
}

/// The stages' seconds since profiling was switched on; returns how many there are.
export fn tf_h3_spent(model: *const h3.Model, out: [*]f64, names: [*][*:0]const u8) c_int {
    inline for (h3.stages, 0..) |s, i| {
        out[i] = model.spent[i];
        names[i] = s ++ "";
    }
    return h3.stages.len;
}

/// tf_h3_forward for inputs in host memory (every address of `in` a host address; pinned memory copies fastest).
export fn tf_h3_forward_host(model: *h3.Model, in: *const h3.Model.Inputs) f64 {
    return host.forwardHost(model, in) catch |e| {
        std.debug.print("tf_h3_forward_host: {t}\n", .{e});
        return -1;
    };
}

/// Pinned host memory the GPU maps, for a host without a framework: returns its host address, stores its device one.
export fn tf_h3_host_alloc(model: *h3.Model, bytes: u64, device: *u64) ?*anyopaque {
    return host.hostAlloc(model, bytes, device) catch |e| {
        std.debug.print("tf_h3_host_alloc: {t}\n", .{e});
        return null;
    };
}

export fn tf_h3_host_free(model: *h3.Model, ptr: *anyopaque) void {
    host.hostFree(model, ptr);
}

test {
    _ = @import("cuda_weights_test.zig");
}

test "h3 library: single-device C ABI has no rank inputs or exchange state" {
    try std.testing.expectEqual(@as(u32, 1), tf_h3_abi_version());
    try std.testing.expectEqual(@as(usize, 96), @sizeOf(h3.Model.Inputs));
    try std.testing.expectEqual(@as(usize, 88), @offsetOf(h3.Model.Inputs, "attention_out"));
    try std.testing.expect(!@hasField(h3.Model, "tp"));
    try std.testing.expect(!@hasField(h3.Model.Inputs, "tile0"));
    try std.testing.expect(!@hasField(h3.Model.Inputs, "tile1"));
    try std.testing.expect(!@hasField(h3.Model.Inputs, "cuts"));
    try std.testing.expectEqual(@as(usize, 9), h3.stages.len);
    var model: h3.Model = undefined;
    model.hidden = 128;
    model.inner = 128;
    model.heads = 1;
    model.mlp = 256;
    model.blocks = &.{};
    model.loaded_seconds = 0.25;
    var shape: [7]u32 = undefined;
    tf_h3_shape(&model, &shape);
    try std.testing.expectEqual([7]u32{ 128, 128, 1, 256, 0, 250, 16 }, shape);
    tf_h3_profile(&model, 1);
    try std.testing.expect(model.profile);
    var spent: [9]f64 = undefined;
    var names: [9][*:0]const u8 = undefined;
    try std.testing.expectEqual(@as(c_int, 9), tf_h3_spent(&model, &spent, &names));
    for (spent, names, h3.stages) |s, n, expected| {
        try std.testing.expectEqual(@as(f64, 0), s);
        try std.testing.expectEqualStrings(expected, std.mem.span(n));
    }
    try std.testing.expectEqual(@as(c_int, -1), tf_h3_set_weight(&model, 0, -1, 0, 0, 0));
}

test "h3 library: timing uses the host monotonic clock" {
    const started = h3.seconds();
    const ended = h3.seconds();
    try std.testing.expect(started > 0 and std.math.isFinite(started));
    try std.testing.expect(ended >= started);
}

test {
    _ = @import("cuda_runtime_test.zig");
}

test "h3 library: exported host and device refusals need no CUDA dependencies or caller reads" {
    var m: h3.Model = undefined;
    m.hidden = 128;
    m.inner = 128;
    m.heads = 1;
    m.mlp = 128;
    var block: [1]h3.Block = undefined;
    m.blocks = &block;
    if (@hasField(h3.Model, "tp")) m.tp = null;
    const valid: h3.Model.Inputs = .{ .x = 0x1000, .tables = 0x1000, .line = 0x1000, .cos = 0, .sin = 0, .slot = 0x1000, .sizes = 0x1000, .rows = 2, .tiles = 1, .lines = 1, .rot = 0, .prefix_tiles = 0, .keep = 0 };
    const Case = struct { field: []const u8, value: u32 };
    inline for (.{ Case{ .field = "rows", .value = 0 }, Case{ .field = "tiles", .value = 0 }, Case{ .field = "lines", .value = 0 }, Case{ .field = "rot", .value = 65 }, Case{ .field = "rot", .value = 128 }, Case{ .field = "prefix_tiles", .value = 2 }, Case{ .field = "keep", .value = 2 }, Case{ .field = "blocks", .value = 2 }, Case{ .field = "rows", .value = 0xffffffff }, Case{ .field = "lines", .value = 0xffffffff } }) |c| {
        var in = valid;
        @field(in, c.field) = c.value;
        const expected = if (std.mem.eql(u8, c.field, "rot")) error.BadH3Rotary else if (std.mem.eql(u8, c.field, "prefix_tiles") or std.mem.eql(u8, c.field, "keep")) error.BadH3Keep else if (std.mem.eql(u8, c.field, "blocks")) error.BadH3Blocks else if (c.value == 0xffffffff) error.BadH3Scalar else error.BadH3Dimensions;
        try std.testing.expectError(expected, m.plan(&in));
        try std.testing.expectEqual(@as(f64, -1), tf_h3_forward(&m, &in));
        try std.testing.expectEqual(@as(f64, -1), tf_h3_forward_host(&m, &in));
    }
    inline for (.{ "x", "tables", "line", "slot", "sizes" }) |field| {
        var in = valid;
        @field(in, field) = 0;
        try std.testing.expectError(error.BadH3Address, m.plan(&in));
        try std.testing.expectEqual(@as(f64, -1), tf_h3_forward(&m, &in));
        try std.testing.expectEqual(@as(f64, -1), tf_h3_forward_host(&m, &in));
    }
    var in = valid;
    in.attention_out = 0xfffffffffffffff0;
    try std.testing.expectError(error.BadH3Address, m.plan(&in));
    try std.testing.expectEqual(@as(f64, -1), tf_h3_forward_host(&m, &in));
    _ = try m.plan(&valid);
    in = valid;
    in.rot = 64;
    in.cos = 0x1000;
    in.sin = 0x1000;
    in.keep = 1;
    _ = try m.plan(&in);
    m.hidden = std.math.maxInt(usize);
    try std.testing.expectEqual(@as(f64, -1), tf_h3_forward_host(&m, &valid));
    try std.testing.expectError(error.BadH3Product, @import("cuda_validation.zig").product(&.{ std.math.maxInt(usize), 2 }));
}
