//! Recording devices verify untaught row isolation, empty-owner work, exact layouts and native parent custody.
const std = @import("std");
const core = @import("core");
const n = @import("learning_numeric.zig");
const hook = @import("learning_hook.zig");
const a = std.testing.allocator;
test "Gemma4 learning: empty and wholly untaught hooks allocate and launch nothing, mixed ranges stay isolated" {
    var f = try @import("learning_fixture.zig").Fixture.open();
    defer f.deinit();
    const r = f.e.r;
    const listener = r.listener();
    const input = try f.b.allocate(3 * 64 * 2);
    defer f.b.release(input);
    const output = try f.b.allocate(input.len);
    defer f.b.release(output);
    @memset(input, 0);
    @memset(output, 0);
    const x = std.mem.bytesAsSlice(u16, input);
    x[0] = n.bits(1);
    x[64] = n.bits(1);
    x[128] = n.bits(1);
    const before = f.b.allocations;
    try listener.begin(listener.ctx, &.{.{ .first = 0, .rows = 3 }}, 3);
    try listener.dense(listener.ctx, {}, 0, .{ .buffer = input, .len = input.len, .rows = 3, .width = 64 }, .{ .buffer = output, .len = output.len, .rows = 3, .width = 64 });
    try std.testing.expect(!listener.active(listener.ctx));
    try std.testing.expectEqual(before, f.b.allocations);
    try std.testing.expectEqual(@as(usize, 0), r.kernels.calls);
    const values = try a.alloc(f32, 3 * 64 * 64);
    defer a.free(values);
    @memset(values, 0);
    values[0] = 1;
    values[64 * 64] = 1;
    values[2 * 64 * 64] = 1;
    try r.pockets.set(0, values[0 .. 64 * 64], values[64 * 64 ..][0 .. 64 * 64], values[2 * 64 * 64 ..], 1);
    const allocated = f.b.allocations;
    try listener.begin(listener.ctx, &.{.{ .first = 0, .rows = 3, .untaught = true }}, 3);
    try listener.dense(listener.ctx, {}, 0, .{ .buffer = input, .len = input.len, .rows = 3, .width = 64 }, .{ .buffer = output, .len = output.len, .rows = 3, .width = 64 });
    try std.testing.expect(!listener.active(listener.ctx));
    try std.testing.expectEqual(allocated, f.b.allocations);
    try std.testing.expectEqual(@as(usize, 0), r.kernels.calls);
    try listener.begin(listener.ctx, &.{ .{ .first = 0, .rows = 1 }, .{ .first = 1, .rows = 1, .untaught = true }, .{ .first = 2, .rows = 1 } }, 3);
    try listener.dense(listener.ctx, {}, 0, .{ .buffer = input, .len = input.len, .rows = 3, .width = 64 }, .{ .buffer = output, .len = output.len, .rows = 3, .width = 64 });
    const out = std.mem.bytesAsSlice(u16, output);
    try std.testing.expect(n.f(out[0]) > 0 and n.f(out[128]) > 0);
    for (out[64..128]) |v| try std.testing.expectEqual(@as(u16, 0), v);
    try std.testing.expectEqual(@as(usize, 4), r.kernels.calls);
}
test "Gemma4 learning: tape gathers packed and head-major layouts and rejects short ranges and aliases" {
    const spec = core.living_gemma;
    const f = @import("../qwen4/learning_fixture.zig");
    for ([_]u32{ 0, 1, 2 }) |layout| {
        var p: spec.Params = .{};
        p.u[0] = 3;
        p.u[1] = 64;
        p.u[2] = if (layout == 2) 128 else 64;
        p.u[3] = layout;
        p.u[4] = 32;
        var c = try f.Case(spec).init(.copy, p);
        defer c.deinit();
        const in = std.mem.bytesAsSlice(u16, c.data[0]);
        for (in, 0..) |*v, i| v.* = @intCast(i);
        try c.run();
        const out = std.mem.bytesAsSlice(u16, c.data[8]);
        for (0..3 * 64) |i| {
            const row = i / 64;
            const col = i % 64;
            const want = if (layout == 1) (col / 32) * 3 * 32 + row * 32 + col % 32 else row * p.u[2] + (if (layout == 2) col / 16 * 32 + col % 16 else col);
            try std.testing.expectEqual(@as(u16, @intCast(want)), out[i]);
        }
    }
    var p: spec.Params = .{};
    p.u[0] = 1;
    p.u[1] = 64;
    const done = try spec.finish(.copy, p);
    var ranges: [11]core.weight_only.Range = @splat(.{ .address = 0x10000, .len = 4096 });
    try std.testing.expectError(error.LivingGemmaAlias, spec.validate(.copy, done, ranges));
    ranges[8].address = 0x20000;
    ranges[0].len = 127;
    try std.testing.expectError(error.LivingGemmaBuffer, spec.validate(.copy, done, ranges));
}
test "Gemma4 learning: failed completion retains the native parent, siblings and driver until recovery" {
    const Parent = struct {
        retained_next: ?*@This() = null,
        completed: bool = false,
        freed: bool = false,
        children: usize = 4,
        fn wait(p: *@This()) anyerror!void {
            if (!p.completed) return error.Pending;
        }
        fn finish(p: *@This()) void {
            std.debug.assert(p.completed);
            p.children = 0;
            p.freed = true;
        }
    };
    const Queue = @import("../../native/close_guard.zig").Queue(Parent, Parent.wait, Parent.finish);
    var parent: Parent = .{};
    Queue.keep(&parent);
    Queue.recover();
    try std.testing.expect(!parent.freed and parent.children == 4);
    parent.completed = true;
    Queue.recover();
    try std.testing.expect(parent.freed and parent.children == 0);
}

test "Gemma4 learning: impossible training geometry refuses before unchecked size arithmetic" {
    const P = @import("learning_buffers.zig").Plan(n.B);
    var s: @import("learning_profile.zig").Shape = .{ .hidden = 64, .intermediate = 64, .layers = 6, .heads = 2, .sliding_kv = 1, .global_kv = 1, .vocab = 64 };
    s.hidden = std.math.maxInt(usize);
    try std.testing.expectError(error.GemmaTrainingShape, P.make(s, 256));
    s.hidden = 64;
    s.heads = 0;
    try std.testing.expectError(error.GemmaTrainingShape, P.make(s, 256));
}

test "Gemma4 learning: high-low probability metadata cannot admit a zero tile-group divisor" {
    var p: core.living_qsa.Params = .{};
    p.u[0] = 1;
    p.u[1] = 1;
    p.u[2] = 1;
    p.u[3] = 64;
    p.u[4] = 1;
    p.u[5] = 2;
    p.f[0] = 1;
    try std.testing.expectError(error.LivingQsaShape, core.living_qsa.needs(.softmax, p));
}
