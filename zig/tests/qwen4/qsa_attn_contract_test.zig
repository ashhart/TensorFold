//! Actual split-softmax bodies face independent binary64 and four planted faults through the maintained gates.
const std = @import("std");
const ct = @import("core").contract;
const common = @import("../../src/families/qwen4/qsa_attn_contract.zig");
const a = std.testing.allocator;
extern fn tf_q4_qsa_attn_cpu(c_int, [*]const [*]const u8, [*]f32, [*]f32, [*]u16, c_int, c_int, c_uint) void;
const Backend = struct {
    pub const name = "host actual MSL";
    pub const Rig = @import("contract_rig.zig").HostRig;
    pub const served: []const ct.spec.Shape = &.{ .{ .rows = 1, .n = 1, .k = 1024 }, .{ .rows = 1, .n = 2, .k = 17 }, .{ .rows = 2, .n = 2, .k = 33 } };
    input: [6][*]const u8 = undefined,
    po: []f32 = undefined,
    pm: []f32 = undefined,
    output: []u16 = undefined,
    control: common.Control = .production,
    pub fn bind(b: *@This(), r: *Rig, f: common.Fixture, control: common.Control) !void {
        b.control = control;
        for ([_][]const u8{ std.mem.sliceAsBytes(f.query), std.mem.sliceAsBytes(f.key), std.mem.sliceAsBytes(f.value), std.mem.sliceAsBytes(f.ids), std.mem.sliceAsBytes(f.nk), std.mem.sliceAsBytes(f.spr) }, 0..) |bytes, i| b.input[i] = (try r.add(bytes, bytes.len)).ptr;
        b.po = @alignCast(std.mem.bytesAsSlice(f32, try r.add(null, f.rows * 24 * 16 * 256 * 4)));
        b.pm = @alignCast(std.mem.bytesAsSlice(f32, try r.add(null, f.rows * 24 * 16 * 2 * 4)));
        b.output = @alignCast(std.mem.bytesAsSlice(u16, try r.add(null, f.rows * 24 * 256 * 2)));
    }
    pub fn launch(b: *@This(), _: *Rig, f: common.Fixture, repeat: u32) !void {
        tf_q4_qsa_attn_cpu(@intCast(@backingInt(b.control)), &b.input, b.po.ptr, b.pm.ptr, b.output.ptr, @intCast(f.rows), @intCast(f.keys), repeat);
    }
};
test "Qwen4 QSA native contract: attention actual parts and merge catch nonfinite repeat tail and reduced sums" {
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
        if (got != .fail) std.debug.print("QSA_ATTN_CONTROL {t}: {t}\n", .{ control, got });
        try std.testing.expectEqual(g.Verdict.fail, got);
    }
}
