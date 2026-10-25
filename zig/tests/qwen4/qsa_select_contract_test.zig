//! Real selection source checks sparse membership and both radix branches against an independent numeric sort.
const std = @import("std");
const ct = @import("core").contract;
const common = @import("../../src/families/qwen4/qsa_select_contract.zig");
const a = std.testing.allocator;
extern fn tf_q4_qsa_select_cpu(c_int, [*]const f32, [*]const i32, [*]const i32, [*]i32, c_int, c_int, c_uint) void;
const Backend = struct {
    pub const name = "host actual MSL";
    pub const Rig = @import("contract_rig.zig").HostRig;
    pub const served: []const ct.spec.Shape = &.{ .{ .rows = 8, .n = 129, .k = 1 }, .{ .rows = 1, .n = 0, .k = 2 }, .{ .rows = 16, .n = 128, .k = 1 }, .{ .rows = 1, .n = 127, .k = 8 }, .{ .rows = 1, .n = 513, .k = 8 }, .{ .rows = 1, .n = 4097, .k = 8 }, .{ .rows = 1, .n = 8192, .k = 8 } };
    scores: []const f32 = undefined,
    complete: []const i32 = undefined,
    ends: []const i32 = undefined,
    ids: []i32 = undefined,
    control: common.Control = .production,
    pub fn bind(b: *@This(), r: *Rig, f: common.Fixture, control: common.Control) !void {
        b.control = control;
        b.scores = @alignCast(std.mem.bytesAsSlice(f32, try r.add(std.mem.sliceAsBytes(f.scores), f.scores.len * 4)));
        b.complete = @alignCast(std.mem.bytesAsSlice(i32, try r.add(std.mem.sliceAsBytes(f.complete), f.rows * 4)));
        b.ends = @alignCast(std.mem.bytesAsSlice(i32, try r.add(std.mem.sliceAsBytes(f.ends), f.rows * 4)));
        b.ids = @alignCast(std.mem.bytesAsSlice(i32, try r.add(null, f.want.len * 4)));
    }
    pub fn launch(b: *@This(), _: *Rig, f: common.Fixture, repeat: u32) !void {
        tf_q4_qsa_select_cpu(@intCast(@backingInt(b.control)), b.scores.ptr, b.complete.ptr, b.ends.ptr, b.ids.ptr, @intCast(f.rows), @intCast(f.stride), repeat);
    }
};
test "Qwen4 QSA native contract: selection source includes sampled path signed IEEE order tails and controls" {
    const Case = common.Case(ct, Backend);
    const g = ct.gates;
    const old = g.quiet;
    g.quiet = true;
    defer g.quiet = old;
    var good: Case = .{ .a = a, .rig = .{}, .backend = .{} };
    defer good.deinit();
    for ([_]g.Gate{ .nonfinite, .repeat, .shapes, .sums }) |gate| {
        const got = try g.one(a, gate, good.kernel(), 0x367);
        if (got != (if (gate == .sums) g.Verdict.na else g.Verdict.pass)) {
            g.quiet = false;
            _ = try g.one(a, gate, good.kernel(), 0x367);
            g.quiet = true;
        }
        try std.testing.expectEqual(if (gate == .sums) g.Verdict.na else g.Verdict.pass, got);
    }
    for ([_]common.Control{ .nan, .repeat, .tail }) |control| {
        var bad: Case = .{ .a = a, .rig = .{}, .backend = .{}, .control = control };
        defer bad.deinit();
        const got = try g.one(a, bad.kernel().control.?, bad.kernel(), 0x367);
        if (got != .fail) std.debug.print("QSA_SELECT_CONTROL {t}: {t}\n", .{ control, got });
        try std.testing.expectEqual(g.Verdict.fail, got);
    }
    for ([_]usize{ 4097, 8192 }) |blocks| {
        try std.testing.expect(try good.prepare(.{ .rows = 1, .n = blocks, .k = 8 }, .stress, 0x367));
        try good.run();
        try std.testing.expectEqual(@as(?[]const u8, null), try good.check(.shapes));
    }
}
