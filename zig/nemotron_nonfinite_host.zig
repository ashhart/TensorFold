//! Exercise the real nonfinite gate with class-preserving bytes and the attention's exact replacement check.
const std = @import("std");
const contract = @import("contract");
const c = contract.spec;
const g = contract.gates;
const exact = @import("tests/contract/nemotron_decode_nonfinite.zig");
const t = std.testing;
const baseline = [_]u16{ 0x3f80, 0x7fc1, 0x7f80, 0xff80, 0xc000, 0, 1, 0xffc3 };
const Mutation = enum { none, finite, nan_payload };
const Probe = struct {
    out: [baseline.len]u16 = baseline,
    mutation: Mutation,
    inner: struct { out: usize = 0, out_bytes: usize = baseline.len * 2 } = .{},
    rig: *Rig = undefined,
    const Rig = struct {
        p: *Probe,
        pub fn bytes(r: *@This(), _: usize, _: usize) []u8 {
            return std.mem.sliceAsBytes(&r.p.out);
        }
    };
    pub fn replaced(_: *@This()) ![]const u8 {
        return std.mem.sliceAsBytes(&baseline);
    }
    fn of(ctx: *anyopaque) *Probe {
        return @ptrCast(@alignCast(ctx));
    }
    fn prepare(ctx: *anyopaque, _: c.Shape, _: g.Fill, _: u64) !bool {
        of(ctx).out = baseline;
        return true;
    }
    fn run(ctx: *anyopaque) !void {
        const p = of(ctx);
        switch (p.mutation) {
            .none => {},
            .finite => p.out[0] = exact.flipFinite(p.out[0]),
            .nan_payload => p.out[1] ^= 2,
        }
    }
    fn output(ctx: *anyopaque) ![]const u8 {
        return std.mem.sliceAsBytes(&of(ctx).out);
    }
    fn inputs(_: *anyopaque) ![]const u8 {
        return &.{};
    }
    fn intact(_: *anyopaque) !bool {
        return true;
    }
    fn overlap(_: *anyopaque) bool {
        return false;
    }
    fn count(_: *anyopaque) usize {
        return baseline.len;
    }
    fn reference(_: *anyopaque, want: []f64, mag: []f64) !void {
        for (want, mag, baseline) |*w, *m, bits| {
            w.* = c.fromBf16(bits);
            m.* = @abs(w.*);
        }
    }
    fn check(ctx: *anyopaque, gate: g.Gate) !?[]const u8 {
        return exact.check(of(ctx), gate);
    }
    fn table(old: bool) g.VTable {
        return .{ .prepare = prepare, .run = run, .output = output, .inputs = inputs, .intact = intact, .overlap = overlap, .count = count, .reference = reference, .check = if (old) null else check };
    }
};
fn depth(_: c.Shape) usize {
    return 1;
}
const shapes = [_]c.Shape{.{ .rows = 5, .n = 1, .k = 1 }};
fn kernel(p: *Probe, vt: *const g.VTable) g.Kernel {
    return .{ .ctx = p, .vt = vt, .spec = .{ .name = "host nonfinite attention", .nan = .{ .stated = exact.rule }, .accum = .f32, .depth = depth, .ops = 1, .out = .bf16, .served = &shapes } };
}
test "attention nonfinite: finite-bit control preserves every BF16 class" {
    for (0..65536) |n| {
        const before: u16 = @intCast(n);
        const after = exact.flipFinite(before);
        try t.expectEqual(c.classOf(c.fromBf16(before)), c.classOf(c.fromBf16(after)));
        if (c.classOf(c.fromBf16(before)) == .finite) try t.expect(before != after) else try t.expectEqual(before, after);
    }
}
test "attention nonfinite: old classes miss finite-value and NaN-payload differences" {
    const old_quiet = g.quiet;
    g.quiet = true;
    defer g.quiet = old_quiet;
    for ([_]Mutation{ .finite, .nan_payload }) |mutation| {
        var p: Probe = .{ .mutation = mutation };
        var rig: Probe.Rig = .{ .p = &p };
        p.rig = &rig;
        const old = Probe.table(true);
        try t.expectEqual(g.Verdict.pass, try g.nonfinite(t.allocator, kernel(&p, &old), 7));
        const next = Probe.table(@import("nonfinite_test_options").class_only);
        try t.expectEqual(g.Verdict.fail, try g.nonfinite(t.allocator, kernel(&p, &next), 7));
    }
}
test "attention nonfinite: equal plain bytes pass and the nonfinite control is caught" {
    const old_quiet = g.quiet;
    g.quiet = true;
    defer g.quiet = old_quiet;
    var p: Probe = .{ .mutation = .none };
    var rig: Probe.Rig = .{ .p = &p };
    p.rig = &rig;
    const vt = Probe.table(@import("nonfinite_test_options").class_only);
    var k = kernel(&p, &vt);
    try t.expectEqual(g.Verdict.pass, try g.nonfinite(t.allocator, k, 7));
    p.mutation = .finite;
    k.control = .nonfinite;
    var tally: g.Tally = .{};
    try g.runAll(t.allocator, &.{k}, 7, &tally);
    try t.expectEqual(@as(usize, 1), tally.caught);
    try t.expectEqual(@as(usize, 0), tally.missed);
}
test "attention nonfinite: control edits only the actual output pack and refuses absent or duplicate sites" {
    const source = @embedFile("kernels/metal/nemotron_glue.metal");
    const changed = try exact.controlSource(t.allocator, source);
    defer t.allocator.free(changed);
    try t.expect(std.mem.indexOf(u8, changed, "if ((bits & 0x7f80u) != 0x7f80u) bits ^= 1u;") != null);
    try t.expectError(error.ControlEditMissing, exact.controlSource(t.allocator, changed));
    const duplicate = try std.mem.concat(t.allocator, u8, &.{ source, source });
    defer t.allocator.free(duplicate);
    try t.expectError(error.ControlEditMissing, exact.controlSource(t.allocator, duplicate));
}
