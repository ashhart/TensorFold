//! The registered five-gate fixtures execute the shared source body on owned, guarded recording buffers.
const std = @import("std");
const core = @import("core");
const wo = core.weight_only;
const contract = core.contract;
const fixture = core.weight_only_contract;
const a = std.testing.allocator;
extern fn tf_weight_only_cpu(x: [*]const u16, w: [*]const u8, s: [*]const u8, b: [*]const u8, out: [*]u16, p: wo.Params, control: c_int, repetition: c_int) void;
const HostRig = @import("contract_rig.zig").HostRig;
const Backend = struct {
    pub const name = "host shared source";
    pub const Rig = HostRig;
    views: [5][]u8 = undefined,
    pub fn bind(b: *Backend, r: *HostRig, f: fixture.Fixture) !void {
        b.views[0] = try r.add(std.mem.sliceAsBytes(f.x), f.x.len * 2);
        b.views[1] = try r.add(f.weight, f.weight.len);
        b.views[2] = try r.add(f.scales, f.scales.len);
        b.views[3] = try r.add(f.biases, f.biases.len);
        b.views[4] = try r.add(null, (try wo.validate(f.p)).out);
    }
    pub fn launch(b: *Backend, _: *HostRig, p: wo.Params, control: fixture.Control, repetition: c_int) !void {
        tf_weight_only_cpu(@ptrCast(@alignCast(b.views[0].ptr)), b.views[1].ptr, b.views[2].ptr, b.views[3].ptr, @ptrCast(@alignCast(b.views[4].ptr)), p, @intCast(@backingInt(control)), repetition);
    }
};
test "weight-only contract: all formats pass gates 1-4 and every actual-source control fails its own gate" {
    const g = contract.gates;
    const previous = g.quiet;
    g.quiet = true;
    defer g.quiet = previous;
    const Case = fixture.Case(contract, Backend);
    for (fixture.variants) |variant| {
        var good: Case = .{ .a = a, .rig = .{}, .backend = .{}, .format = variant.format, .group = variant.group, .scale_kind = variant.scale_kind };
        defer good.deinit();
        for ([_]g.Gate{ .nonfinite, .repeat, .shapes, .sums }) |gate| {
            const result = try g.one(a, gate, good.kernel(), 0x367);
            if (result != .pass) std.debug.print("weight-only {t} {t} failed\n", .{ variant.format, gate });
            try std.testing.expectEqual(g.Verdict.pass, result);
        }
        for ([_]fixture.Control{ .nan, .repeat, .tail, .sums }, [_]g.Gate{ .nonfinite, .repeat, .shapes, .sums }) |control, gate| {
            var bad: Case = .{ .a = a, .rig = .{}, .backend = .{}, .format = variant.format, .group = variant.group, .scale_kind = variant.scale_kind, .control = control };
            defer bad.deinit();
            try std.testing.expectEqual(g.Verdict.fail, try g.one(a, gate, bad.kernel(), 0x367));
        }
    }
}
