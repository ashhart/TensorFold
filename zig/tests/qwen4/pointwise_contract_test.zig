//! Both rounding policies use the actual shared scalar body in their registered gate fixtures and controls.
const std = @import("std");
const core = @import("core");
const h = core.hc_pointwise;
const common = core.hc_pointwise_contract;
const a = std.testing.allocator;
extern fn tf_hc_pointwise_strided_cpu([*]const u16, [*]const u16, [*]const u16, [*]u16, c_int, c_int, c_int, c_int, c_int, c_int, c_int, c_int, c_int) void;
const B = struct {
    pub const name = "host shared source";
    pub const Rig = @import("contract_rig.zig").HostRig;
    views: [4][]u8 = undefined,
    pub fn bind(b: *B, r: *Rig, f: common.Fixture) !void {
        for ([_][]const u8{ std.mem.sliceAsBytes(f.x), std.mem.sliceAsBytes(f.b), std.mem.sliceAsBytes(f.g) }, 0..) |bytes, i| b.views[i] = try r.add(bytes, bytes.len);
        b.views[3] = try r.add(null, f.x.len * 2);
    }
    pub fn launch(b: *B, _: *Rig, p: h.Params, control: common.Control, repeat: c_int) !void {
        tf_hc_pointwise_strided_cpu(@ptrCast(@alignCast(b.views[0].ptr)), @ptrCast(@alignCast(b.views[1].ptr)), @ptrCast(@alignCast(b.views[2].ptr)), @ptrCast(@alignCast(b.views[3].ptr)), p.rows, p.hidden, p.streams, @backingInt(p.op), @backingInt(p.round_mid), p.gate_stride, p.gate_dim, @intCast(@backingInt(control)), repeat);
    }
};
test "HC pointwise contract: both policies pass finite gates, every actual-source control fails" {
    const g = core.contract.gates;
    const old = g.quiet;
    g.quiet = true;
    defer g.quiet = old;
    const Case = common.Case(core.contract, B);
    for (common.variants) |variant| {
        var good: Case = .{ .a = a, .rig = .{}, .backend = .{}, .variant = variant };
        defer good.deinit();
        for ([_]g.Gate{ .nonfinite, .repeat, .shapes }) |gate| {
            const verdict = try g.one(a, gate, good.kernel(), 0x367);
            if (verdict != .pass) std.debug.print("pointwise {t} {t} {t}\\n", .{ variant.op, variant.rounding, gate });
            try std.testing.expectEqual(g.Verdict.pass, verdict);
        }
        try std.testing.expectEqual(g.Verdict.na, try g.one(a, .sums, good.kernel(), 0x367));
        for ([_]common.Control{ .nan, .repeat, .tail }, [_]g.Gate{ .nonfinite, .repeat, .shapes }) |control, gate| {
            var bad: Case = .{ .a = a, .rig = .{}, .backend = .{}, .variant = variant, .control = control };
            defer bad.deinit();
            try std.testing.expectEqual(g.Verdict.fail, try g.one(a, gate, bad.kernel(), 0x367));
        }
    }
}
