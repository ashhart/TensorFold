//! The real trainer runs against native graph storage and nonzero scalar device instruments.
const std = @import("std");
const core = @import("core");
const n = @import("../../../tests/qwen4/numeric_gpu.zig");
const a = std.testing.allocator;
const E = @import("engine.zig").Engine(n.G);
pub const K = struct {
    const Ref = @import("train_tensor.zig").Ref(n.B);
    rope: ?n.B.Buffer = null,
    pub fn init(_: std.mem.Allocator) K {
        return .{};
    }
    pub fn deinitDrained(k: *K) void {
        if (k.rope) |r| a.free(r);
    }
    pub fn length(b: n.B.Buffer) usize {
        return b.len;
    }
    pub fn probabilityCut(_: *K, g: *n.G) bool {
        return g.probability_bf16;
    }
    pub fn read(_: *K, b: *n.B, r: Ref, out: []u8) !void {
        try b.drain();
        if (out.len > r.len) return error.Range;
        @memcpy(out, r.buffer[r.offset..][0..out.len]);
    }
    pub fn frequency(k: *K, g: *n.G, bytes: usize) !Ref {
        if (k.rope == null) {
            const r = try a.alignedAlloc(u8, .@"16", @max(4, bytes));
            const f = std.mem.bytesAsSlice(f32, r);
            for (0..g.c.rotary / 2) |i| f[i] = @floatCast(std.math.pow(f64, g.c.theta, -2 * @as(f64, @floatFromInt(i)) / @as(f64, @floatFromInt(g.c.rotary))));
            k.rope = r;
        }
        return .{ .buffer = k.rope.?, .len = @max(4, bytes) };
    }
    fn run(comptime spec: type, op: spec.Op, p: spec.Params, refs: [11]Ref) !void {
        var ptrs: [11][*]u8 = undefined;
        var ranges: [11]core.weight_only.Range = undefined;
        for (refs, &ptrs, &ranges) |r, *ptr, *range| {
            try std.testing.expect(r.offset <= r.buffer.len and r.len <= r.buffer.len - r.offset);
            ptr.* = r.buffer.ptr + r.offset;
            range.* = .{ .address = @intFromPtr(ptr.*), .len = r.len };
        }
        try spec.validate(op, p, ranges);
        const f = if (spec == core.living_rows) tf_living_rows_cpu else if (spec == core.living_gdn) tf_living_gdn_cpu else tf_living_qsa_cpu;
        f(@intCast(@backingInt(op)), p, &ptrs);
    }
    pub fn row(_: *K, _: *n.G, op: core.living_rows.Op, p: core.living_rows.Params, refs: [11]Ref) !void {
        try run(core.living_rows, op, p, refs);
    }
    pub fn gdn(_: *K, _: *n.G, op: core.living_gdn.Op, p: core.living_gdn.Params, refs: [11]Ref) !void {
        try run(core.living_gdn, op, p, refs);
    }
    pub fn qsa(_: *K, _: *n.G, op: core.living_qsa.Op, p: core.living_qsa.Params, refs: [11]Ref) !void {
        try run(core.living_qsa, op, p, refs);
    }
};
extern fn tf_living_rows_cpu(c_uint, core.living_rows.Params, [*][*]u8) void;
extern fn tf_living_gdn_cpu(c_uint, core.living_rows.Params, [*][*]u8) void;
extern fn tf_living_qsa_cpu(c_uint, core.living_rows.Params, [*][*]u8) void;
pub const T = @import("train.zig").Trainer(E, K);
pub const R = @import("learning_runtime.zig").Runtime(E, K);
test "Qwen4 learning: actual trainer declarations compile for native recording storage" {
    std.testing.refAllDecls(T);
}

test "Qwen4 learning: real native trainer records keys, loss and early-row pullbacks and restores hooks" {
    var f = try @import("learning_native_fixture.zig").Fixture.open();
    defer f.deinit();
    const r = try R.init(a, f.e);
    defer r.close() catch unreachable;
    f.e.learning = r.listener();
    const trainer = try T.init(a, std.testing.io, f.e, &r.pockets, r);
    defer trainer.deinit();
    const tokens = [_]u32{ 3, 5, 7, 2, 9 };
    const turn: @import("living").lesson_route.Turn = .{ .decide = 0, .whole = true };
    _ = try trainer.step(&tokens, 3, .keys, turn);
    try std.testing.expect(trainer.normed(0, 0).len == f.e.c.hidden);
    const before = try trainer.step(&tokens, 3, .loss, turn);
    trainer.nudge = .{ .layers = .{ 1, 0, 0 }, .rows = .{ 0, 0, 0 } };
    const target = try trainer.step(&tokens, 3, .target, turn);
    try std.testing.expectEqual(before.loss, target.loss);
    var mag: f64 = 0;
    for (trainer.pushAt(0, 0, 0)) |x| mag += @abs(x);
    try std.testing.expect(mag > 0);
    try std.testing.expect(f.e.trace == null and f.e.gpu.trace == null);
    try std.testing.expectEqual(@as(*anyopaque, @ptrCast(r)), f.e.learning.?.ctx);
    const base = f.b.held;
    f.b.fail_drain = true;
    trainer.drain();
    try std.testing.expect(!f.e.healthy());
    f.b.fail_drain = false;
    try std.testing.expectEqual(base, f.b.held);
}

test "Qwen4 learning: trainer gradients traverse GDN and QSA with independent frozen-store binary64 directions" {
    var f = try @import("learning_native_fixture.zig").Fixture.open();
    defer f.deinit();
    const r = try R.init(a, f.e);
    defer r.close() catch unreachable;
    const trainer = try T.init(a, std.testing.io, f.e, &r.pockets, r);
    defer trainer.deinit();
    const tokens = [_]u32{ 3, 5, 7, 2, 9, 11, 4, 12, 19, 6, 8, 17, 2, 3, 15, 20, 7, 12, 9, 2 };
    const turn: @import("living").lesson_route.Turn = .{ .decide = 0, .whole = true };
    const d = f.e.c.hidden;
    const direction = try a.alloc(f32, d);
    defer a.free(direction);
    for (direction, 0..) |*v, i| v.* = if ((i * 17 + i / 13) % 5 < 2) -1 else 1;
    for ([_]bool{ true, false }) |cut| {
        f.e.gpu.probability_bf16 = cut;
        for ([_]usize{ 0, 8, 17 }) |row| {
            trainer.nudge = .{ .layers = .{ 1, 0, 0 }, .rows = .{ row, 0, 0 } };
            @memset(trainer.delta, 0);
            const base = try trainer.step(&tokens, 17, .target, turn);
            var analytic: f64 = 0;
            for (trainer.pushAt(0, 0, 0), direction) |g, v| analytic += g * v;
            var point: @import("learning_reference.zig").Point(T) = .{ .a = a, .t = trainer };
            defer point.deinit();
            const ref_base = try point.loss(0, row, direction, 0);
            try std.testing.expectApproxEqAbs(@as(f64, base.loss), ref_base, 1e-6);
            for ([_]f64{ 0.0001, 0.00005 }) |epsilon| {
                const hi = try point.loss(0, row, direction, epsilon);
                const lo = try point.loss(0, row, direction, -epsilon);
                const slope = (hi - lo) / (2 * epsilon);
                std.debug.print("QWEN4_TRAIN_FROZEN cut={} row={d} analytic={d} fd={d}\n", .{ cut, row, analytic, slope });
                try @import("learning_fixture.zig").near(@floatCast(analytic), slope);
            }
        }
    }
}

pub fn selectionHash(t: *T) !u64 {
    var hash = std.hash.Wyhash.init(0);
    for (0..t.e.c.layers) |l| {
        if (t.e.c.kinds[l] != .indexed_attention) continue;
        const counts = try t.tape.indexed(l, "self_attn.selection_count", t.rows);
        const sparse = try t.tape.indexed(l, "self_attn.selection_sparse", t.rows);
        const ids = try t.tape.raw(l, "self_attn.selected_ids", t.rows);
        hash.update(counts.buffer[counts.offset..][0..counts.len]);
        const cc = n.view(i32, counts);
        const ss = n.view(i32, sparse);
        for (cc, ss, 0..) |count, sp, row| {
            if (sp == 0) continue;
            hash.update(ids.buffer[ids.offset + row * t.ids.len * 4 ..][0 .. @as(usize, @intCast(count)) * 4]);
        }
    }
    return hash.final();
}
