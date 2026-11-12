//! One aligned device allocation owns the checkpoint's unchanged BF16 tensors.
const std = @import("std");
const hip = @import("hip");
const schema = @import("schema.zig");
const checkpoint = @import("checkpoint.zig");
const Allocator = std.mem.Allocator;
pub const Placement = struct { offset: usize, bytes: usize };

pub const Layout = struct {
    arena: std.heap.ArenaAllocator,
    tensors: std.StringArrayHashMapUnmanaged(Placement) = .empty,
    bytes: usize = 0,

    pub fn init(a: Allocator) !Layout {
        var layout: Layout = .{ .arena = .init(a) };
        errdefer layout.deinit();
        try schema.each(&layout);
        return layout;
    }

    pub fn tensor(l: *Layout, name: []const u8, shape: []const usize) !void {
        var size: usize = 2;
        for (shape) |dim| size = try std.math.mul(usize, size, dim);
        const offset = std.mem.alignForward(usize, l.bytes, 256);
        const a = l.arena.allocator();
        try l.tensors.put(a, try a.dupe(u8, name), .{ .offset = offset, .bytes = size });
        l.bytes = try std.math.add(usize, offset, size);
    }

    pub fn deinit(l: *Layout) void {
        l.arena.deinit();
        l.* = undefined;
    }
};

pub const Weights = struct {
    layout: Layout,
    buffer: hip.DeviceBuffer,

    /// The caller reserves runtime, KV and scratch separately before it supplies the weight budget.
    pub fn load(a: Allocator, r: *const hip.Runtime, io: std.Io, dir: []const u8, budget: usize) !Weights {
        var layout = try Layout.init(a);
        errdefer layout.deinit();
        if (layout.bytes > budget) return error.GlmOcrWeightsDoNotFit;
        var ck = try checkpoint.Checkpoint.open(a, io, dir);
        defer ck.close();
        var buffer = try hip.DeviceBuffer.alloc(r, layout.bytes);
        errdefer buffer.free();
        for (layout.tensors.keys(), layout.tensors.values()) |name, place| {
            const source = try ck.get(name);
            if (source.bytes.len != place.bytes) return error.BadGlmOcrTensor;
            try buffer.upload(place.offset, source.bytes);
        }
        return .{ .layout = layout, .buffer = buffer };
    }

    pub fn get(w: *const Weights, name: []const u8) !u64 {
        const place = w.layout.tensors.get(name) orelse return error.MissingGlmOcrTensor;
        return w.buffer.address(place.offset);
    }

    /// The caller must complete every command that reads a tensor before releasing this owner.
    pub fn deinit(w: *Weights) void {
        w.buffer.free();
        w.layout.deinit();
        w.* = undefined;
    }
};

test "GLM-OCR HIP weights: full schema stays aligned and inside its admitted budget" {
    var layout = try Layout.init(std.testing.allocator);
    defer layout.deinit();
    try std.testing.expectEqual(@as(usize, 526), layout.tensors.count());
    var end: usize = 0;
    for (layout.tensors.values()) |place| {
        try std.testing.expectEqual(@as(usize, 0), place.offset % 256);
        try std.testing.expect(place.offset >= end);
        end = place.offset + place.bytes;
    }
    try std.testing.expectEqual(layout.bytes, end);
    const r: hip.Runtime = undefined;
    try std.testing.expectError(error.GlmOcrWeightsDoNotFit, Weights.load(std.testing.allocator, &r, std.testing.io, "absent", layout.bytes - 1));
    std.testing.refAllDecls(Weights);
}
