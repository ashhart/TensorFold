//! The actual native engine and owner run a nonzero checkpoint through recording arithmetic instruments.
const std = @import("std");
const rows = @import("core").living_rows;
const n = @import("../../../tests/qwen4/numeric_gpu.zig");
const E = @import("engine.zig").Engine(n.G);
const a = std.testing.allocator;
extern fn tf_living_rows_cpu(c_uint, rows.Params, [*][*]u8) void;
const K = struct {
    const Ref = @import("train_tensor.zig").Ref(n.B);
    calls: usize = 0,
    rows_seen: usize = 0,
    pub fn init(_: std.mem.Allocator) K {
        return .{};
    }
    pub fn deinitDrained(_: *K) void {}
    pub fn row(k: *K, _: *n.G, op: rows.Op, p: rows.Params, refs: [11]Ref) !void {
        var ranges: [11]@import("core").weight_only.Range = undefined;
        var ptrs: [11][*]u8 = undefined;
        for (refs, &ranges, &ptrs) |r, *range, *ptr| {
            try std.testing.expect(r.offset <= r.buffer.len and r.len <= r.buffer.len - r.offset);
            ptr.* = r.buffer.ptr + r.offset;
            range.* = .{ .address = @intFromPtr(ptr.*), .len = r.len };
        }
        try rows.validate(op, p, ranges);
        k.calls += 1;
        k.rows_seen += p.u[0];
        tf_living_rows_cpu(@intCast(@backingInt(op)), p, &ptrs);
    }
};
const Capture = struct {
    out: []u16,
    pub fn encode(c: Capture, _: *n.G, logits: E.Tensor, _: []const E.Segment) !void {
        try std.testing.expectEqual(c.out.len, n.bits(logits).len);
        @memcpy(c.out, n.bits(logits));
    }
};

test "Qwen4 learning: native mixed forward isolates untaught rows and an empty hook changes no launch" {
    var f = try @import("numeric_engine_test.zig").Fixture.open();
    defer f.deinit();
    const e = f.e;
    const left = try e.lease();
    const right = try e.lease();
    const count = 3 * e.c.vocab;
    const baseline = try a.alloc(u16, count);
    defer a.free(baseline);
    const candidate = try a.alloc(u16, count);
    defer a.free(candidate);
    const items = [_]E.Segment{ .{ .state = &left.state, .tokens = &.{ 3, 5 } }, .{ .state = &right.state, .tokens = &.{7} } };
    var problem: @import("config.zig").Problem = .{};
    try e.forward(&items, Capture{ .out = baseline }, &problem);
    try e.keep(left, 0);
    try e.keep(right, 0);
    const peak = e.scratch.peak;
    const R = @import("learning_runtime.zig").Runtime(E, K);
    const r = try R.init(a, e);
    defer r.close() catch unreachable;
    e.learning = r.listener();
    const allocations = f.b.allocs;
    try e.forward(&items, Capture{ .out = candidate }, &problem);
    try std.testing.expectEqualSlices(u16, baseline, candidate);
    try std.testing.expectEqual(@as(usize, 0), r.kernels.calls);
    try std.testing.expectEqual(allocations, f.b.allocs);
    try std.testing.expectEqual(peak, e.scratch.peak);
    try e.keep(left, 0);
    try e.keep(right, 0);
    const size = 64 * e.c.hidden;
    const values = try a.alloc(f32, size * 3);
    defer a.free(values);
    @memset(values, 0);
    values[0] = 2;
    values[size + 1] = 1;
    values[2 * size] = 64;
    try r.pockets.set(0, values[0..size], values[size..][0..size], values[2 * size ..], 1);
    right.state.untaught = true;
    try e.forward(&items, Capture{ .out = candidate }, &problem);
    try std.testing.expectEqual(@as(usize, 2), r.kernels.calls);
    try std.testing.expectEqual(@as(usize, 4), r.kernels.rows_seen);
    try std.testing.expectEqualSlices(u16, baseline[2 * e.c.vocab ..], candidate[2 * e.c.vocab ..]);
    try std.testing.expect(!std.mem.eql(u16, baseline[0 .. 2 * e.c.vocab], candidate[0 .. 2 * e.c.vocab]));
    try e.keep(left, 0);
    try e.keep(right, 0);
    left.state.untaught = true;
    const before = f.b.allocs;
    try e.forward(&items, Capture{ .out = candidate }, &problem);
    try std.testing.expectEqualSlices(u16, baseline, candidate);
    try std.testing.expectEqual(@as(usize, 2), r.kernels.calls);
    try std.testing.expectEqual(before, f.b.allocs);
    try e.keep(left, 0);
    try e.keep(right, 0);
    try e.release(left);
    try std.testing.expect(!(try e.lease()).state.untaught);
}
