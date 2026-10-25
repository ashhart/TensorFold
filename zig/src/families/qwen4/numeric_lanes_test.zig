//! Shared verification sees nonzero model logits through the actual native adapter; the drawer is a CPU instrument.
const std = @import("std");
const n = @import("../../../tests/qwen4/numeric_gpu.zig");
const m = @import("../../../tests/qwen4/math.zig");
const E = @import("engine.zig").Engine(n.G);
const L = @import("core").lanes;
const t = std.testing;
const a = t.allocator;
const D = struct {
    rows: usize = 128,
    out: [256]u32 = @splat(0),
    draws: usize = 0,
    pub fn draw(d: *D, x: E.Tensor, offset: usize, positions: []const u64, sampling: ?L.Sampling, mask: []const u32) !void {
        try @import("draw_inputs.zig").check(x.width, positions, sampling, mask);
        if (sampling != null) return error.NumericInstrumentGreedyOnly;
        const words = (x.width + 31) / 32;
        for (0..x.rows) |r| {
            var best: f32 = -std.math.inf(f32);
            var picked: u32 = 0;
            for (n.row(x, r), 0..) |v, i| {
                if (mask.len != 0 and mask[r * words + i / 32] & (@as(u32, 1) << @intCast(i % 32)) == 0) continue;
                if (m.f(v) > best) {
                    best = m.f(v);
                    picked = @intCast(i);
                }
            }
            d.out[offset / 4 + r] = picked;
        }
        d.draws += x.rows;
    }
    pub fn read(d: *D, offset: usize, out: []u32) !void {
        @memcpy(out, d.out[offset / 4 ..][0..out.len]);
    }
};
test "Qwen4 native numeric: real shared core copied drafts equal plain with nonzero weights" {
    var f = try @import("numeric_engine_test.zig").Fixture.open();
    defer f.deinit();
    var drawer: D = .{};
    const A = @import("lanes_adapter.zig").Adapter(E, D, L);
    var adapter = try A.init(a, f.e, &drawer);
    defer adapter.deinitDrained();
    var config = try L.Config.init(a, adapter.facts(), 4, 3);
    defer config.deinit(a);
    config.timed = false;
    var clock: L.backend.WallClock = .{ .io = std.testing.io };
    var core = L.Engine.init(a, &config, adapter.backend(), clock.clock());
    defer core.deinit();
    const Proposal = struct {
        fn give(_: *anyopaque, _: []const u32, max: i64) ![]const u32 {
            const ids = &[_]u32{ 0, 7, 0 };
            return ids[0..@intCast(@max(0, @min(max, ids.len)))];
        }
        fn matched(_: *anyopaque) i64 {
            return 32;
        }
    };
    var id: u8 = 0;
    const proposer: L.proposer.Proposer = .{ .ptr = &id, .vtable = &.{ .propose = Proposal.give, .last_match = Proposal.matched, .priced = true } };
    var drafted = try L.Stream.init(a, .{ .id = "drafted", .prompt = f.doc.value.tokens, .max_new = 8, .drafts = true, .proposer = proposer });
    defer drafted.deinit(a);
    var plain = try L.Stream.init(a, .{ .id = "plain", .prompt = f.doc.value.tokens, .max_new = 8, .drafts = false });
    defer plain.deinit(a);
    try core.addStream(&drafted);
    try core.addStream(&plain);
    var steps: usize = 0;
    while (core.activeCount() > 0) {
        try t.expect(steps < 32);
        try core.step();
        steps += 1;
    }
    try t.expectEqualSlices(u32, plain.emitted(), drafted.emitted());
    try t.expectEqual(@as(usize, 8), plain.emitted().len);
    try t.expect(drafted.drafted > 0);
    try t.expect(drawer.draws > plain.emitted().len + drafted.emitted().len);
    var nonzero = false;
    for (plain.emitted()) |idv| nonzero = nonzero or idv != 0;
    try t.expect(nonzero);
}
