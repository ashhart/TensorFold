//! Header-only spans for complete routed experts and PLE rows, retaining their stored component layouts.
const std = @import("std");
const arena = @import("arena.zig");
const rows = @import("rows.zig");
pub const suffixes = [_][]const u8{ "weight", "scales", "biases" };
pub const projections = [_][]const u8{ "gate_proj", "up_proj", "down_proj" };
pub const Experts = struct { spans: []arena.Expert, strides: [9]usize };

pub fn experts(a: std.mem.Allocator, source: anytype, layers: usize, count: usize, mtp: ?[]const u8) !Experts {
    if (layers == 0 or count == 0) return error.InvalidPoolShape;
    const blocks = try std.math.add(usize, layers, @intFromBool(mtp != null));
    const spans = try a.alloc(arena.Expert, try std.math.mul(usize, blocks, count));
    errdefer a.free(spans);
    var strides: [9]usize = @splat(0);
    var name: [256]u8 = undefined;
    for (0..blocks) |layer| {
        for (projections, 0..) |proj, pi| for (suffixes, 0..) |suffix, ci| {
            const path = if (layer < layers)
                try std.fmt.bufPrint(&name, "language_model.model.layers.{d}.mlp.switch_mlp.{s}.{s}", .{ layer, proj, suffix })
            else
                try std.fmt.bufPrint(&name, "{s}.layers.0.mlp.switch_mlp.{s}.{s}", .{ mtp.?, proj, suffix });
            const span = try source.span(path);
            if (span.length == 0 or span.length % count != 0) return error.BadExpertTensor;
            const stride = span.length / count;
            if (stride % @as(usize, if (ci == 0) 4 else 2) != 0) return error.InvalidExpertStride;
            strides[3 * pi + ci] = @max(strides[3 * pi + ci], stride);
            for (0..count) |expert| {
                var part = span;
                part.length = stride;
                part.offset = try std.math.add(u64, part.offset, try std.math.mul(u64, expert, stride));
                spans[layer * count + expert][3 * pi + ci] = part;
            }
        };
    }
    return .{ .spans = spans, .strides = strides };
}

/// Locality-only tracing needs the physical stride but does not allocate one descriptor per expert.
pub fn expertStrides(source: anytype, layers: usize, count: usize, mtp: ?[]const u8) ![9]usize {
    if (layers == 0 or count == 0) return error.InvalidPoolShape;
    var out: [9]usize = @splat(0);
    var name: [256]u8 = undefined;
    for (0..layers + @intFromBool(mtp != null)) |layer| {
        for (projections, 0..) |proj, pi| for (suffixes, 0..) |suffix, ci| {
            const path = if (layer < layers)
                try std.fmt.bufPrint(&name, "language_model.model.layers.{d}.mlp.switch_mlp.{s}.{s}", .{ layer, proj, suffix })
            else
                try std.fmt.bufPrint(&name, "{s}.layers.0.mlp.switch_mlp.{s}.{s}", .{ mtp.?, proj, suffix });
            const span = try source.span(path);
            if (span.length == 0 or span.length % count != 0) return error.BadExpertTensor;
            out[3 * pi + ci] = @max(out[3 * pi + ci], span.length / count);
        };
    }
    return out;
}

pub fn ple(a: std.mem.Allocator, source: anytype, spelling: []const u8, shards: usize, width: usize, bits: usize, group: usize) ![]rows.Shard {
    if (shards == 0 or width == 0 or group == 0 or width % group != 0) return error.InvalidPleRecipe;
    if (bits < 2 or bits > 8 or bits == 7 or width * bits % 32 != 0) return error.InvalidPleRecipe;
    const sizes = [3]usize{ width * bits / 8, width / group * 2, width / group * 2 };
    const out = try a.alloc(rows.Shard, shards);
    errdefer a.free(out);
    var name: [256]u8 = undefined;
    for (out, 0..) |*shard, i| {
        shard.rows = 0;
        for (suffixes, sizes, 0..) |suffix, size, j| {
            const path = try std.fmt.bufPrint(&name, "language_model.model.layers.1.ple.ple_embedding.ngram_embedding.{s}{d}.{s}", .{ spelling, i, suffix });
            const span = try source.span(path);
            if (span.length == 0 or span.length % size != 0) return error.InvalidRowTable;
            const n = span.length / size;
            if (j == 0) shard.rows = n else if (shard.rows != n) return error.InconsistentRowWidth;
            shard.parts[j] = span;
        }
    }
    return out;
}

test "FSR catalog slices every component and includes the draft layer without touching payload" {
    const Source = struct {
        count: usize = 4,
        pub fn span(s: @This(), name: []const u8) !@import("reader.zig").Span {
            if (std.mem.indexOf(u8, name, "switch_mlp") == null) return error.MissingTensor;
            return .{ .file = if (std.mem.startsWith(u8, name, "mtp.")) 1 else 0, .offset = 128, .length = s.count * 8 };
        }
    };
    const a = std.testing.allocator;
    const got = try experts(a, Source{}, 2, 4, "mtp");
    defer a.free(got.spans);
    try std.testing.expectEqual(@as(usize, 12), got.spans.len);
    for (got.spans, 0..) |expert, i| for (expert) |span| {
        try std.testing.expectEqual(@as(u64, 128 + i % 4 * 8), span.offset);
        try std.testing.expectEqual(@as(usize, 8), span.length);
        try std.testing.expectEqual(@as(usize, if (i < 8) 0 else 1), span.file);
    };
    try std.testing.expectEqual(@as([9]usize, @splat(8)), got.strides);
    try std.testing.expectEqual(got.strides, try expertStrides(Source{}, 2, 4, "mtp"));
    try std.testing.expectError(error.BadExpertTensor, experts(a, Source{ .count = 3 }, 2, 5, null));
}
