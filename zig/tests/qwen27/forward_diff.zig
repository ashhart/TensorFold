const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const q = tf.qwen27;
const graph = @import("synthetic_graph.zig");
const frozen = @import("forward_copy_reference.zig");
const driver = @import("forward_diff_runner.zig");

fn all(buffer: mtl.Buffer) []u8 {
    return buffer.contents()[0..buffer.length()];
}
fn bytesEqual(a: []const u8, b: []const u8, comptime label: []const u8) !void {
    if (!std.mem.eql(u8, a, b)) {
        for (a, b, 0..) |x, y, i| if (x != y) {
            std.debug.print("forward difference {s} byte{d}:{d}/{d}\n", .{ label, i, x, y });
            break;
        };
        return error.ForwardDifference;
    }
}
fn sentinel(r: *q.decode_round.Runner) void {
    @memset(all(r.model.frame.get(.logits).buffer), 0x5a);
    @memset(all(r.taps.buffer), 0xa5);
    for (r.caches) |cache| {
        @memset(all(cache.keys.buffer), 0);
        @memset(all(cache.values.buffer), 0);
    }
}
fn outputs(a: *q.decode_round.Runner, b: *q.decode_round.Runner, rows: usize) !void {
    try bytesEqual(all(a.model.frame.get(.logits).buffer), all(b.model.frame.get(.logits).buffer), "head/canaries");
    try bytesEqual(all(a.taps.buffer), all(b.taps.buffer), "taps/canaries");
    try bytesEqual(all(a.model.frame.get(.input).buffer)[0 .. rows * a.model.config.hidden * 2], all(b.model.frame.get(.input).buffer)[0 .. rows * b.model.config.hidden * 2], "normalized hidden");
}
fn state(a: *q.decode_round.Runner, b: *q.decode_round.Runner) !void {
    const x = try (q.session.Session{ .runner = a }).fingerprint();
    const y = try (q.session.Session{ .runner = b }).fingerprint();
    try bytesEqual(&x, &y, "committed GDN/KV/logits");
    for (a.caches, b.caches) |ac, bc| {
        try bytesEqual(all(ac.keys.buffer), all(bc.keys.buffer), "all key bytes");
        try bytesEqual(all(ac.values.buffer), all(bc.values.buffer), "all value bytes");
    }
}
fn prefill(r: *q.decode_round.Runner, ids: []const u32, comptime reference: bool) !void {
    try r.reset(0);
    sentinel(r);
    if (!reference) return (q.session.Session{ .runner = r }).prefill(ids, 16);
    var round = try q.round_plan.Round.init(r.allocator, &.{.{ .slot = 0, .start = 0, .capacity = r.capacity, .ids = ids }}, @intCast(r.model.config.conv_kernel), 1, @intCast(r.model.config.vocab));
    defer round.deinit();
    try driver.verify(frozen.Stages, r, &round, .last, null, false);
    try r.keep(&round, &.{&.{ 0, 1, 2 }});
}
pub fn run(a: std.mem.Allocator, perturb: bool) !void {
    const cm = try graph.createLayers(a, 5);
    defer cm.deinit();
    const fm = try graph.createLayers(a, 5);
    defer fm.deinit();
    var candidate = try q.decode_round.Runner.init(a, cm, 1, 256);
    defer candidate.deinit();
    var reference = try q.decode_round.Runner.init(a, fm, 1, 256);
    defer reference.deinit();
    candidate.capture_taps = true;
    reference.capture_taps = true;
    candidate.taps.ids = .{ 0, 1, 2, 3, 4 };
    reference.taps.ids = candidate.taps.ids;
    var ids: [128]u32 = undefined;
    for (&ids, 0..) |*token, i| token.* = @intCast((i * 17 + 7) % 96);
    var cases: usize = 0;
    for (1..129) |rows| for ([_]bool{ false, true }) |tree| for ([_]q.forward.Head{ .all, .last, .none }) |head| {
        try prefill(&candidate, ids[0..3], false);
        try prefill(&reference, ids[0..3], true);
        try outputs(&candidate, &reference, 3);
        try state(&candidate, &reference);
        var parents: [128]i32 = undefined;
        var path: [128]u32 = undefined;
        parents[0] = -1;
        path[0] = 0;
        var kept: usize = 1;
        for (1..rows) |i| {
            parents[i] = @intCast(if (tree and i % 2 == 1) i - 1 else if (tree) i - 2 else i - 1);
            if (!tree or i % 2 == 0) {
                path[kept] = @intCast(i);
                kept += 1;
            }
        }
        var round = try q.round_plan.Round.init(a, &.{.{ .slot = 0, .start = 3, .capacity = 256, .ids = ids[0..rows], .parents = parents[0..rows] }}, @intCast(cm.config.conv_kernel), 1, @intCast(cm.config.vocab));
        defer round.deinit();
        @memset(all(cm.frame.get(.logits).buffer), 0x5a);
        @memset(all(fm.frame.get(.logits).buffer), 0x5a);
        @memset(all(candidate.taps.buffer), 0xa5);
        @memset(all(reference.taps.buffer), 0xa5);
        try driver.verify(q.forward.Stages, &candidate, &round, head, null, perturb);
        try driver.verify(frozen.Stages, &reference, &round, head, null, false);
        outputs(&candidate, &reference, rows) catch |err| {
            std.debug.print("rows{d} tree{} head{s}\n", .{ rows, tree, @tagName(head) });
            return err;
        };
        try candidate.keep(&round, &.{path[0..kept]});
        try reference.keep(&round, &.{path[0..kept]});
        try state(&candidate, &reference);
        if (head != .none and try (q.session.Session{ .runner = &candidate }).greedy() != try (q.session.Session{ .runner = &reference }).greedy()) return error.TokenDiffers;
        cases += 1;
    };
    for (0..17) |abort_at| {
        try prefill(&candidate, ids[0..3], false);
        try prefill(&reference, ids[0..3], true);
        try outputs(&candidate, &reference, 3);
        try state(&candidate, &reference);
        var round = try q.round_plan.Round.init(a, &.{.{ .slot = 0, .start = 3, .capacity = 256, .ids = ids[0..7] }}, @intCast(cm.config.conv_kernel), 1, @intCast(cm.config.vocab));
        defer round.deinit();
        const before = try (q.session.Session{ .runner = &candidate }).fingerprint();
        const taps = try a.dupe(u8, all(candidate.taps.buffer));
        defer a.free(taps);
        if (driver.verify(q.forward.Stages, &candidate, &round, .all, abort_at, false)) |_| return error.CancellationIgnored else |err| if (err != error.ControlCanceled) return err;
        if (driver.verify(frozen.Stages, &reference, &round, .all, abort_at, false)) |_| return error.CancellationIgnored else |err| if (err != error.ControlCanceled) return err;
        try bytesEqual(&before, &try (q.session.Session{ .runner = &candidate }).fingerprint(), "canceled publication");
        try bytesEqual(taps, all(candidate.taps.buffer), "canceled tap bytes");
        try driver.verify(q.forward.Stages, &candidate, &round, .all, null, false);
        try driver.verify(frozen.Stages, &reference, &round, .all, null, false);
        try outputs(&candidate, &reference, 7);
        try candidate.keep(&round, &.{&.{ 0, 1, 2, 3, 4, 5, 6 }});
        try reference.keep(&round, &.{&.{ 0, 1, 2, 3, 4, 5, 6 }});
        try state(&candidate, &reference);
    }
    std.debug.print("forward differential: {d} row/tree/head cases plus17 canceled-stage/reuse cases, all normalized/tap/head/GDN/KV/token bytes equal copy reference\n", .{cases});
}
