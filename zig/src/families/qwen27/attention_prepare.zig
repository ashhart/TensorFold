//! One ordered Q/K norm and RoPE launch, shared by full attention and keys-only prompt tails.
const glue = @import("glue.zig");
pub fn encode(comptime Size: type, e: anytype, pipeline: anytype, p: anytype, keys_only: bool) void {
    const g = p.geometry;
    e.setPipeline(pipeline);
    inline for (.{ p.qg, p.mixed, p.q_gain, p.k_gain, p.positions }, 0..) |r, i| e.setBuffer(r.buffer, r.offset, i);
    e.setValue(glue.QkNormRope{ .width = g.dim, .heads = if (keys_only) 0 else g.heads, .kv_heads = g.kv_heads, .rows = p.rows, .rotary = g.rotary, .key_stride = if (p.key_row_stride == 0) 2 * g.kv_heads * g.dim else p.key_row_stride, .eps = p.eps, .theta = @floatCast(p.theta), .h_end = p.h_end, .w_end = p.w_end, .plane = p.plane }, 5);
    inline for (.{ p.norm_q, p.norm_k, p.query, p.rotated_key }, 6..) |r, i| e.setBuffer(r.buffer, r.offset, i);
    e.dispatchGroups(Size.of((if (keys_only) @as(u32, 0) else g.heads) + g.kv_heads, p.rows, 1), Size.of(64, 1, 1));
    e.barrier();
}

const std = @import("std");
const Event = enum { q, kv, barrier, fused, append };
const Ref = struct { buffer: u32 = 1, offset: usize = 64 };
const RecordingSize = struct {
    width: usize,
    height: usize,
    depth: usize,
    pub fn of(x: usize, y: usize, z: usize) RecordingSize {
        return .{ .width = x, .height = y, .depth = z };
    }
};
const Recording = struct {
    events: [12]Event = undefined,
    n: usize = 0,
    args: glue.QkNormRope = undefined,
    grid: RecordingSize = undefined,
    threads: RecordingSize = undefined,
    fn add(s: *Recording, event: Event) void {
        s.events[s.n] = event;
        s.n += 1;
    }
    pub fn quant(s: *Recording, _: *Recording, weight: Event, _: Ref, _: ?Ref, _: Ref, _: Ref, _: u32) !void {
        s.add(weight);
    }
    pub fn get(_: *Recording, _: anytype) Ref {
        return .{};
    }
    pub fn barrier(s: *Recording) void {
        s.add(.barrier);
    }
    pub fn setPipeline(_: *Recording, _: u32) void {}
    pub fn setBuffer(_: *Recording, _: u32, _: usize, _: usize) void {}
    pub fn setValue(s: *Recording, args: glue.QkNormRope, _: usize) void {
        s.args = args;
    }
    pub fn dispatchGroups(s: *Recording, grid: RecordingSize, threads: RecordingSize) void {
        s.grid = grid;
        s.threads = threads;
        s.add(.fused);
    }
};
test "qwen27 mixed: one fused norm and RoPE dispatch after projections and before append" {
    try std.testing.expectEqual(@as(usize, 44), @sizeOf(glue.QkNormRope));
    for ([_]u32{ 1, 16, 2048 }) |rows| for ([_]bool{ false, true }) |keys_only| {
        var e = Recording{};
        try @import("attention_input.zig").project(&e, &e, .{ .q = Event.q, .kv = Event.kv }, &e, @as(?Ref, null), rows);
        const p = .{ .geometry = .{ .dim = @as(u32, 256), .heads = @as(u32, 24), .kv_heads = @as(u32, 4), .rotary = @as(u32, 64) }, .rows = rows, .qg = Ref{}, .mixed = Ref{}, .q_gain = Ref{}, .k_gain = Ref{}, .positions = Ref{}, .norm_q = Ref{}, .norm_k = Ref{}, .query = Ref{}, .rotated_key = Ref{}, .key_row_stride = @as(u32, 0), .eps = @as(f32, 1e-6), .theta = @as(f64, 10000000), .h_end = @as(u32, 12), .w_end = @as(u32, 10), .plane = rows };
        encode(RecordingSize, &e, @as(u32, 11), p, keys_only);
        e.add(.append);
        try std.testing.expectEqualSlices(Event, &.{ .q, .kv, .barrier, .fused, .barrier, .append }, e.events[0..e.n]);
        try std.testing.expectEqual(@as(usize, if (keys_only) 4 else 28), e.grid.width);
        try std.testing.expectEqual(@as(usize, rows), e.grid.height);
        try std.testing.expectEqual(@as(usize, 64), e.threads.width);
        try std.testing.expectEqual(@as(u32, 2048), e.args.key_stride);
        try std.testing.expectEqual(rows, e.args.plane);
    };
}
