//! Actual custom checkers with only stream synchronization and output readback recorded on the host.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const act = @import("contract_act.zig");
const gdn = @import("contract_gdn.zig");
const qk = @import("contract_qk.zig");
const rig = @import("contract_rig.zig");
const gate = core.contract.gates;
export fn act_contract_hooks(x: *act.Act) *const gate.VTable {
    return x.kernel().vt;
}
export fn gdn_contract_hooks(x: *gdn.Gdn) *const gate.VTable {
    return x.kernel().vt;
}
export fn qk_contract_hooks(x: *qk.Qk) *const gate.VTable {
    return x.kernel().vt;
}
fn sync(_: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
    return 0;
}
fn copy(dst: ?*anyopaque, src: u64, n: usize) callconv(.c) cuda.abi.Result {
    const bytes: [*]const u8 = @ptrFromInt(src);
    const out: [*]u8 = @ptrCast(dst.?);
    @memcpy(out[0..n], bytes[0..n]);
    return 0;
}
fn output(d: *cuda.Driver, bytes: []u8) !rig.Rig {
    d.api.cuStreamSynchronize = sync;
    d.api.cuMemcpyDtoH_v2 = copy;
    var r: rig.Rig = .{ .gpu = .{ .d = d, .ctx = undefined, .gpa = std.testing.allocator, .io = undefined }, .s = .{ .d = d, .handle = null } };
    try r.slots.append(std.testing.allocator, .{ .buf = .{ .d = d, .ptr = @intFromPtr(bytes.ptr) - rig.guard, .len = bytes.len + 2 * rig.guard }, .len = bytes.len, .kind = .output });
    return r;
}
fn close(r: *rig.Rig) void {
    const a = std.testing.allocator;
    r.slots.deinit(a);
    r.seen.deinit(a);
    r.outs.deinit(a);
    r.host.deinit(a);
}
fn legacyRow(symbol: []const u8) bool {
    inline for (.{
        "tf_swiglu_rows", "tf_gate_rows",    "tf_group_sums_rows", "tf_dconv_rows",
        "tf_gdn_prep_64", "tf_gdn_prep_128", "tf_gdn_prep_256",    "tf_qk_rows",
        "tf_qk_rows_w",
    }) |name| if (std.mem.eql(u8, symbol, name)) return true;
    return false;
}
test "all fifteen row launches have measured 13.0 resources" {
    var count: usize = 0;
    for (core.contract.registry.all_cuda) |r| {
        const image = r.launch.image;
        if (!std.mem.eql(u8, image, "act_rows") and !std.mem.eql(u8, image, "gdn_prep") and !std.mem.eql(u8, image, "qk_rows")) continue;
        // New launches in the same images have their own records; this check retains the original fifteen.
        if (!legacyRow(r.launch.symbol)) continue;
        count += 1;
        for (r.usage) |u| try std.testing.expect(u.regs > 0);
        try std.testing.expectEqual(@as(usize, 0), try core.contract.registry.checkCuda(std.testing.allocator, &.{r}, true));
        try std.testing.expectEqualDeep(core.contract.registry.none, core.contract.registry.usageFor(r, .{ .major = 13, .minor = 2 }));
    }
    try std.testing.expectEqual(@as(usize, 15), count);
}
test "bf16 stress non-distinction is explicit and retains declared fp32 arithmetic" {
    for ([_]usize{ 64, 128, 256 }) |dim| for ([_]gdn.Part{ .qk, .v }) |part| {
        const s = gdn.spec(dim, part);
        try std.testing.expectEqual(core.contract.spec.Accum.f32, s.accum);
        try std.testing.expectEqual(@as(f64, 0), s.sums.bf16);
        try std.testing.expect(s.sums.why.len > 0 and s.sums.f16 > 0);
    };
    for (qk.forms) |form| if (form.norm != .none) {
        const s = qk.spec(form);
        try std.testing.expectEqual(core.contract.spec.Accum.f32, s.accum);
        try std.testing.expectEqual(@as(f64, 0), s.sums.bf16);
        try std.testing.expect(s.sums.why.len > 0 and s.sums.f16 > 0);
    };
    var a: act.Act = .{ .rig = undefined, .which = .dconv_res, .k = undefined, .arena = undefined };
    var g: gdn.Gdn = .{ .rig = undefined, .dk = 64, .part = .gb, .k = undefined, .arena = undefined };
    try std.testing.expect(a.kernel().vt.check != null);
    try std.testing.expect(g.kernel().vt.check != null);
}
test "actual residual checker accepts its formula and rejects every wrong output class or bound" {
    var d: cuda.Driver = undefined;
    var got: [16]u16 = @splat(0x4000);
    var x: [16]u16 = @splat(0x3f80);
    var base: [64]u16 = @splat(0x3f80);
    var dyn: [4]u16 = @splat(0);
    var res: [16]u16 = @splat(0x3f80);
    var f: act.Act = .{ .rig = try output(&d, std.mem.sliceAsBytes(&got)), .which = .dconv_res, .k = undefined, .arena = .init(std.testing.allocator), .shape = .{ .rows = 1, .n = 16, .k = 1 }, .x = &x, .base = &base, .dyn = &dyn, .res = &res };
    defer close(&f.rig);
    defer f.arena.deinit();
    const check = f.kernel().vt.check.?;
    try std.testing.expectEqual(@as(?[]const u8, null), try check(&f, .shapes));
    for (0..got.len) |i| {
        got[i] = 0x4040;
        try std.testing.expect((try check(&f, .shapes)) != null);
        got[i] = 0x7fc0;
        try std.testing.expect((try check(&f, .shapes)) != null);
        got[i] = 0x4000;
    }
    for ([_]u16{ 0x7fc0, 0x7f80, 0xff80 }) |word| {
        x[0] = word;
        got[0] = word;
        try std.testing.expectEqual(@as(?[]const u8, null), try check(&f, .nonfinite));
        got[0] = 0x4000;
        try std.testing.expect((try check(&f, .nonfinite)) != null);
    }
}
test "actual g beta checker bounds both projections and refuses nonfinite erasure" {
    var d: cuda.Driver = undefined;
    var got: [6]f32 = @splat(0.5);
    var av: [6]u16 = @splat(0);
    var bv: [6]u16 = @splat(0);
    var logs: [3]f32 = @splat(0);
    var bias: [3]f32 = @splat(0);
    var f: gdn.Gdn = .{ .rig = try output(&d, std.mem.sliceAsBytes(&got)), .dk = 64, .part = .gb, .k = undefined, .arena = .init(std.testing.allocator), .shape = .{ .rows = 1, .n = 1, .k = 2 }, .sh = .{ .kh = 1, .vh = 3, .dk = 64, .keep = 1 }, .av = &av, .bv = &bv, .a_log = &logs, .dtb = &bias };
    defer close(&f.rig);
    defer f.arena.deinit();
    const check = f.kernel().vt.check.?;
    try std.testing.expectEqual(@as(?[]const u8, null), try check(&f, .shapes));
    for (0..got.len) |i| {
        got[i] = 0.6;
        try std.testing.expect((try check(&f, .shapes)) != null);
        got[i] = std.math.nan(f32);
        try std.testing.expect((try check(&f, .shapes)) != null);
        got[i] = 0.5;
    }
    av[0] = 0x7fc0;
    got[0] = std.math.nan(f32);
    try std.testing.expectEqual(@as(?[]const u8, null), try check(&f, .nonfinite));
    got[0] = 0.5;
    try std.testing.expect((try check(&f, .nonfinite)) != null);
}
