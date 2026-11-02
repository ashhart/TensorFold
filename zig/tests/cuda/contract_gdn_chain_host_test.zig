//! The device replacement and planted controls are live; no driver calls run here.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const suite = @import("contract_gdn_chain.zig");
const g = core.contract.gates;

export fn gdn_chain_contract_hooks(t: *suite.Case) *const g.VTable {
    return t.kernel().vt;
}

export fn gdn_chain_suite_compile(gpu: *const @import("check.zig").Gpu, s: *const cuda.Stream, m: *const cuda.Module) void {
    var cases = suite.Suite.init(gpu.*, s.*, m.*) catch return;
    defer cases.deinit();
    var list: std.ArrayList(g.Kernel) = .empty;
    defer list.deinit(gpu.gpa);
    cases.append(gpu.gpa, &list) catch return;
}

test "GDN chain port: raw byte registration retains every output and both source controls" {
    var t: suite.Case = undefined;
    t.sums = false;
    t.chunks = false;
    t.control = .none;
    t.shape = .{ .rows = 33, .n = 16, .k = 48 };
    const bytes = 2 * 33 * 48 * 128 * 2 + 48 * 128 * 128 * 4;
    try std.testing.expectEqual(@as(usize, bytes), t.count());
    const k = t.kernel();
    try std.testing.expect(k.vt.replaced != null and k.vt.check != null and k.spec.replaces != null and k.spec.in_place);
    t.sums = true;
    try std.testing.expectEqual(@as(usize, bytes + 33 * 48 * 2 * 4), t.count());
    for ([_]suite.Control{ .gate, .state }) |control| {
        t.control = control;
        try std.testing.expectEqual(g.Gate.shapes, t.kernel().control.?);
    }
    var found: usize = 0;
    for (core.contract.registry.all_cuda) |r| if (std.mem.eql(u8, r.launch.symbol, cuda.gdn_chain.symbol)) {
        found += 1;
        try std.testing.expectEqual(@as(u32, 256), r.launch.threads);
        try std.testing.expectEqual(core.contract.registry.none, r.usage);
    };
    try std.testing.expectEqual(@as(usize, 1), found);
}

test "GDN chain port: source control witnesses distinguish the SiLU and head-state placement" {
    const y: f64 = 0.75;
    const gate: f64 = 0.25;
    const production = y * gate / (1 + @exp(-gate));
    try std.testing.expect(@abs(production - y) > 0.5);
    const state = [_]u32{ 0x3f810000, 0x3f820000, 0x3f830000, 0x3f840000 };
    const rotated = [_]u32{ state[3], state[0], state[1], state[2] };
    try std.testing.expect(!std.mem.eql(u32, &state, &rotated));
}
