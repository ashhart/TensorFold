//! Actual QSA score source checks FP32 integer cancellation, max-number exceptional rules and guards.
const std = @import("std");
const ct = @import("core").contract;
const common = @import("../../src/families/qwen4/qsa_scores_contract.zig");
const a = std.testing.allocator;
extern fn tf_q4_qsa_scores_cpu(c_int, [*]const u16, [*]const u16, [*]const i32, [*]f32, c_int, c_int, c_uint) void;
const Backend = struct {
    pub const name = "host actual MSL";
    pub const Rig = @import("contract_rig.zig").HostRig;
    pub const served: []const ct.spec.Shape = &.{ .{ .rows = 1, .n = 129, .k = 128 }, .{ .rows = 8, .n = 257, .k = 128 }, .{ .rows = 1, .n = 4097, .k = 128 } };
    query: []const u16 = undefined,
    pooled: []const u16 = undefined,
    complete: []const i32 = undefined,
    output: []f32 = undefined,
    control: common.Control = .production,
    pub fn bind(b: *@This(), r: *Rig, f: common.Fixture, control: common.Control) !void {
        b.control = control;
        b.query = @alignCast(std.mem.bytesAsSlice(u16, try r.add(std.mem.sliceAsBytes(f.query), f.query.len * 2)));
        b.pooled = @alignCast(std.mem.bytesAsSlice(u16, try r.add(std.mem.sliceAsBytes(f.pooled), f.pooled.len * 2)));
        b.complete = @alignCast(std.mem.bytesAsSlice(i32, try r.add(std.mem.sliceAsBytes(f.complete), f.complete.len * 4)));
        b.output = @alignCast(std.mem.bytesAsSlice(f32, try r.add(null, f.rows * f.blocks * 4)));
    }
    pub fn launch(b: *@This(), _: *Rig, f: common.Fixture, repeat: u32) !void {
        tf_q4_qsa_scores_cpu(@intCast(@backingInt(b.control)), b.query.ptr, b.pooled.ptr, b.complete.ptr, b.output.ptr, @intCast(f.rows), @intCast(f.blocks), repeat);
    }
};
test "Qwen4 QSA native contract: scores actual source and nonfinite repeat tail and sum controls" {
    const Case = common.Case(ct, Backend);
    const g = ct.gates;
    const old = g.quiet;
    g.quiet = true;
    defer g.quiet = old;
    var good: Case = .{ .a = a, .rig = .{}, .backend = .{} };
    defer good.deinit();
    for ([_]g.Gate{ .nonfinite, .repeat, .shapes, .sums }) |gate| {
        const got = try g.one(a, gate, good.kernel(), 0x367);
        if (got != .pass) {
            g.quiet = false;
            _ = try g.one(a, gate, good.kernel(), 0x367);
            g.quiet = true;
        }
        try std.testing.expectEqual(g.Verdict.pass, got);
    }
    for ([_]common.Control{ .nan, .repeat, .tail, .sums }) |control| {
        var bad: Case = .{ .a = a, .rig = .{}, .backend = .{}, .control = control };
        defer bad.deinit();
        const got = try g.one(a, bad.kernel().control.?, bad.kernel(), 0x367);
        if (got != .fail) std.debug.print("QSA_SCORES_CONTROL {t}: {t}\n", .{ control, got });
        try std.testing.expectEqual(g.Verdict.fail, got);
    }
}
