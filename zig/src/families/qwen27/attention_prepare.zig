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

/// The pack barrier stays in the consumer. No intermediate barrier, append or pack follows a direct prepare.
pub fn direct(units: bool, fused: bool) bool {
    return units and fused;
}
pub fn cache(comptime Size: type, e: anytype, pipes: anytype, prepared: anytype, scratch: anytype, binding: anytype, plan: anytype, q: anytype, k: anytype, v: anytype, args: anytype) void {
    if (prepared) |p| {
        const g = p.geometry;
        e.setPipeline(pipes[12]);
        inline for (.{ p.qg, p.mixed, p.q_gain, p.k_gain, p.positions }, 0..) |r, i| e.setBuffer(r.buffer, r.offset, i);
        e.setValue(glue.QkNormRope{ .width = g.dim, .heads = g.heads, .kv_heads = g.kv_heads, .rows = p.rows, .rotary = g.rotary, .key_stride = 2 * g.kv_heads * g.dim, .eps = p.eps, .theta = @floatCast(p.theta), .h_end = p.h_end, .w_end = p.w_end, .plane = p.plane }, 5);
        inline for (.{ p.norm_q, p.norm_k, p.query, p.rotated_key, binding.get(.caches), binding.get(.meta), scratch.get(.nodes), scratch.get(.q_rows), scratch.get(.qa) }, 6..) |r, i| e.setBuffer(r.buffer, r.offset, i);
        e.dispatchGroups(Size.of(g.heads + 2 * g.kv_heads, p.rows, 1), Size.of(64, 1, 1));
    } else {
        e.setPipeline(pipes[2]);
        inline for (.{ k, v, binding.get(.caches), binding.get(.meta), scratch.get(.nodes) }, 0..) |r, i| e.setBuffer(r.buffer, r.offset, i);
        e.setValue(args, 5);
        e.dispatchThreads(Size.of(args.dim, args.kv_heads, plan.rows), Size.of(32, 4, 1));
        e.setPipeline(pipes[0]);
        inline for (.{ q, scratch.get(.q_rows), scratch.get(.qa) }, 0..) |r, i| e.setBuffer(r.buffer, r.offset, i);
        e.setValue(args, 3);
        e.dispatchThreads(Size.of(args.dim, args.packed_rows, args.kv_heads), Size.of(32, 4, 1));
    }
}

const std = @import("std");
const Event = enum { q, kv, barrier, fused, append, pack, cache };
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
    pipeline: u32 = 11,
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
    pub fn setPipeline(s: *Recording, p: u32) void {
        s.pipeline = p;
    }
    pub fn setBuffer(_: *Recording, _: u32, _: usize, _: usize) void {}
    pub fn setValue(s: *Recording, args: anytype, _: usize) void {
        if (@TypeOf(args) == glue.QkNormRope) s.args = args;
    }
    pub fn dispatchGroups(s: *Recording, grid: RecordingSize, threads: RecordingSize) void {
        s.grid = grid;
        s.threads = threads;
        s.add(if (s.pipeline == 12) .cache else .fused);
    }
    pub fn dispatchThreads(s: *Recording, _: RecordingSize, _: RecordingSize) void {
        s.add(if (s.pipeline == 2) .append else .pack);
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

test "qwen27 mixed: fused M5 prepare writes cache and packed Q without append pack or intermediate barrier" {
    const pipes = [_]u32{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
    for ([_]bool{ false, true }) |units| for ([_]bool{ false, true }) |fused| {
        var e = Recording{};
        const p = .{ .geometry = .{ .dim = @as(u32, 256), .heads = @as(u32, 24), .kv_heads = @as(u32, 4), .rotary = @as(u32, 64) }, .rows = @as(u32, 1), .qg = Ref{}, .mixed = Ref{}, .q_gain = Ref{}, .k_gain = Ref{}, .positions = Ref{}, .norm_q = Ref{}, .norm_k = Ref{}, .query = Ref{}, .rotated_key = Ref{}, .key_row_stride = @as(u32, 0), .eps = @as(f32, 1e-6), .theta = @as(f64, 10000000), .h_end = @as(u32, 12), .w_end = @as(u32, 10), .plane = @as(u32, 1) };
        const args = .{ .dim = @as(u32, 256), .kv_heads = @as(u32, 4), .packed_rows = @as(u32, 16) };
        try @import("attention_input.zig").project(&e, &e, .{ .q = Event.q, .kv = Event.kv }, &e, @as(?Ref, null), 1);
        if (!direct(units, fused)) encode(RecordingSize, &e, pipes[11], p, false);
        cache(RecordingSize, &e, pipes, if (direct(units, fused)) @as(?@TypeOf(p), p) else null, &e, &e, .{ .rows = @as(u32, 1) }, Ref{}, Ref{}, Ref{}, args);
        e.barrier(); // first consumer, after all cache and QA writes
        const want: []const Event = if (units and fused) &.{ .q, .kv, .barrier, .cache, .barrier } else &.{ .q, .kv, .barrier, .fused, .barrier, .append, .pack, .barrier };
        try std.testing.expectEqualSlices(Event, want, e.events[0..e.n]);
        if (units and fused) {
            try std.testing.expectEqual(@as(usize, 32), e.grid.width);
            try std.testing.expectEqual(@as(usize, 64), e.threads.width);
        }
    };
}
