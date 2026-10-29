//! Checked per-stream device layout: shared KV prefixes and independent recurrent, convolution and lexical frames.
const std = @import("std");
const cfg = @import("config.zig");
pub const Range = struct { offset: usize = 0, bytes: usize = 0 };
pub const Layer = struct {
    conv: Range = .{},
    recurrent: Range = .{},
    ple: Range = .{},
    key: Range = .{},
    value: Range = .{},
    index: Range = .{},
    pooled: Range = .{},
    history: Range = .{},
};
fn reserve(end: *usize, count: usize, elem: usize) !Range {
    const bytes = try std.math.mul(usize, count, elem);
    const start = (try std.math.add(usize, end.*, 15)) & ~@as(usize, 15);
    end.* = try std.math.add(usize, start, bytes);
    return .{ .offset = start, .bytes = bytes };
}
fn product(values: []const usize) !usize {
    var out: usize = 1;
    for (values) |value| out = try std.math.mul(usize, out, value);
    return out;
}
pub const Layout = struct {
    layers: [cfg.max_layers]Layer = @splat(.{}),
    count: usize,
    capacity: usize,
    state_bytes: usize = 0,
    kv_bytes: usize = 0,
    history_bytes: usize = 0,
    ratio: usize,
    pub fn of(c: *const cfg.Config, capacity: usize) !Layout {
        if (capacity == 0 or capacity > c.context) return error.Qwen4StateCapacity;
        var out: Layout = .{ .count = c.layers, .capacity = capacity, .ratio = c.index_ratio };
        for (out.layers[0..c.layers], 0..) |*layer, i| {
            switch (c.kinds[i]) {
                .linear_attention => {
                    layer.conv = try reserve(&out.state_bytes, try product(&.{ c.conv - 1, c.convWidth() }), 2);
                    layer.recurrent = try reserve(&out.state_bytes, try product(&.{ c.value_heads, c.value_dim, c.key_dim }), 4);
                },
                .indexed_attention => {
                    layer.key = try reserve(&out.kv_bytes, try product(&.{ capacity, c.kv_heads, c.head_dim }), 2);
                    layer.value = try reserve(&out.kv_bytes, try product(&.{ capacity, c.kv_heads, c.head_dim }), 2);
                    layer.index = try reserve(&out.kv_bytes, try product(&.{ capacity, c.index_dim }), 2);
                    layer.pooled = try reserve(&out.kv_bytes, try product(&.{ @max(@as(usize, 1), capacity / c.index_ratio), c.index_dim }), 2);
                },
            }
            if (c.ple[i]) {
                layer.ple = try reserve(&out.state_bytes, try product(&.{ c.ple_conv - 1, c.ngram, c.wide() }), 2);
                layer.history = try reserve(&out.history_bytes, c.ngram - 1, 4);
            }
        }
        return out;
    }
    /// KV beyond the prefix and an incomplete pooled block never become part of a kept prompt state.
    pub fn savedBytes(l: Layout, c: *const cfg.Config, at: usize) !usize {
        if (at > l.capacity or c.layers != l.count) return error.Qwen4StateCapacity;
        var bytes = try std.math.add(usize, l.state_bytes, l.history_bytes);
        for (l.layers[0..l.count], 0..) |layer, i| {
            _ = i;
            if (layer.key.bytes == 0) continue;
            const keys = try product(&.{ at, c.kv_heads, c.head_dim, 4 });
            const raw = try product(&.{ at, c.index_dim, 2 });
            const pool = try product(&.{ at / l.ratio, c.index_dim, 2 });
            bytes = try std.math.add(usize, bytes, try std.math.add(usize, keys, try std.math.add(usize, raw, pool)));
        }
        return bytes;
    }
    pub fn deviceBytes(l: Layout, window: usize) !usize {
        if (window == 0 or window > 128) return error.Qwen4Window;
        // One mutable frame and base/row snapshots. KV storage is shared only inside this stream.
        return std.math.add(usize, l.kv_bytes, try std.math.mul(usize, window + 2, l.state_bytes));
    }
};
