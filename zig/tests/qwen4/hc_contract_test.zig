//! Registered HC cases and planted controls execute the actual shared scalar header on guarded CPU memory.
const std = @import("std");
const core = @import("core");
const common = core.hc_activation_contract;
const h = core.hc_activation;
const a = std.testing.allocator;
extern fn tf_hc_activation_cpu([*]const u16, [*]u16, c_int, c_int, c_int, c_int, c_int) void;
const B = struct {
    pub const name = "host shared source";
    pub const Rig = @import("contract_rig.zig").HostRig;
    x: []u8 = undefined,
    out: []u8 = undefined,
    pub fn bind(b: *B, r: *Rig, f: common.Fixture) !void {
        b.x = try r.add(std.mem.sliceAsBytes(f.x), f.x.len * 2);
        b.out = try r.add(null, f.x.len * 2);
    }
    pub fn launch(b: *B, _: *Rig, p: h.Params, control: common.Control, repeat: c_int) !void {
        tf_hc_activation_cpu(@ptrCast(@alignCast(b.x.ptr)), @ptrCast(@alignCast(b.out.ptr)), p.rows * p.cols, p.streams, p.injection, @intCast(@backingInt(control)), repeat);
    }
};
test "HC contract: every stream count and role, all finite gates, actual-source controls" {
    const g = core.contract.gates;
    const old = g.quiet;
    g.quiet = true;
    defer g.quiet = old;
    const Case = common.Case(core.contract, B);
    for (2..9) |streams| for ([_]bool{ false, true }) |injection| {
        var good: Case = .{ .a = a, .rig = .{}, .backend = .{}, .streams = @intCast(streams), .injection = injection };
        defer good.deinit();
        for ([_]g.Gate{ .nonfinite, .repeat, .shapes }) |gate| try std.testing.expectEqual(g.Verdict.pass, try g.one(a, gate, good.kernel(), 0x367));
        try std.testing.expectEqual(g.Verdict.na, try g.one(a, .sums, good.kernel(), 0x367));
        for ([_]common.Control{ .nan, .repeat, .tail }, [_]g.Gate{ .nonfinite, .repeat, .shapes }) |control, gate| {
            var bad: Case = .{ .a = a, .rig = .{}, .backend = .{}, .streams = @intCast(streams), .injection = injection, .control = control };
            defer bad.deinit();
            try std.testing.expectEqual(g.Verdict.fail, try g.one(a, gate, bad.kernel(), 0x367));
        }
    };
}
