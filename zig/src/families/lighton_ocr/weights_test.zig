//! Tensor-map tests use a small asymmetric graph and a recording checkpoint, without a GPU or large fixture.
const std = @import("std");
const core = @import("core");
const config = @import("config.zig");
const weights = @import("weights.zig");
const a = std.testing.allocator;

fn small() !config.Config {
    var c = try config.Config.parse(a, config.sample);
    c.hidden = 8;
    c.intermediate = 12;
    c.layers = 1;
    c.vocab = 5;
    c.heads = 4;
    c.kv_heads = 2;
    c.head_dim = 4;
    c.vision.hidden = 8;
    c.vision.intermediate = 12;
    c.vision.layers = 1;
    c.vision.patch = 2;
    return c;
}

const Fake = struct {
    gpa: std.mem.Allocator,
    buffers: std.ArrayList([]u8) = .empty,
    missing_query: bool = false,
    short_bytes: bool = false,
    query: bool = false,
    merge: bool = false,

    fn deinit(f: *Fake) void {
        for (f.buffers.items) |p| f.gpa.free(p);
        f.buffers.deinit(f.gpa);
    }

    pub fn expect(f: *Fake, name: []const u8, dtype: core.checkpoint.DType, shape: []const usize) !core.checkpoint.Tensor {
        try std.testing.expectEqual(.bf16, dtype);
        try std.testing.expect(std.mem.indexOf(u8, name, "lm_head") == null);
        if (std.mem.eql(u8, name, "language_model.model.layers.0.self_attn.q_proj.weight")) {
            if (f.missing_query) return error.MissingTensor;
            try std.testing.expectEqualSlices(usize, &.{ 16, 8 }, shape);
            f.query = true;
        }
        if (std.mem.eql(u8, name, "model.vision_projection.patch_merger.merging_layer.weight")) {
            try std.testing.expectEqualSlices(usize, &.{ 8, 32 }, shape);
            f.merge = true;
        }
        var count: usize = 2;
        for (shape) |d| count *= d;
        const bytes = try f.gpa.alloc(u8, count - @intFromBool(f.short_bytes));
        errdefer f.gpa.free(bytes);
        try f.buffers.append(f.gpa, bytes);
        var t: core.checkpoint.Tensor = .{ .dtype = dtype, .rank = @intCast(shape.len), .shape = @splat(1), .bytes = bytes };
        @memcpy(t.shape[0..shape.len], shape);
        return t;
    }
};

test "LightOnOCR weights preserve wider queries and the learned merger without a second head" {
    var f: Fake = .{ .gpa = a };
    defer f.deinit();
    var l = try weights.Layout.init(a, &f, try small());
    defer l.deinit();
    try std.testing.expect(f.query and f.merge);
    try std.testing.expectEqual(@as(usize, 28), l.entries.items.len);
    try std.testing.expectEqual(@as(usize, 1), l.text.len);
    try std.testing.expectEqual(@as(usize, 1), l.vision.len);
    var end: usize = 0;
    for (l.entries.items) |e| {
        try std.testing.expectEqual(@as(u64, 0), e.offset % 256);
        try std.testing.expect(e.offset >= end);
        end = std.mem.alignForward(usize, e.offset + e.bytes.len, 256);
    }
    try std.testing.expectEqual(end, l.bytes);
    try std.testing.expectEqual(@as(u64, 0), l.embedding);
}

test "LightOnOCR weights reject missing projections and inconsistent tensor byte counts" {
    var f: Fake = .{ .gpa = a, .missing_query = true };
    defer f.deinit();
    try std.testing.expectError(error.MissingTensor, weights.Layout.init(a, &f, try small()));
    var short: Fake = .{ .gpa = a, .short_bytes = true };
    defer short.deinit();
    try std.testing.expectError(error.UnexpectedTensor, weights.Layout.init(a, &short, try small()));
}

test "LightOnOCR tensor mapping releases allocations after every allocation failure" {
    try std.testing.checkAllAllocationFailures(a, allocationFailures, .{try small()});
}

fn allocationFailures(gpa: std.mem.Allocator, c: config.Config) !void {
    var f: Fake = .{ .gpa = gpa };
    defer f.deinit();
    var l = try weights.Layout.init(gpa, &f, c);
    defer l.deinit();
}
