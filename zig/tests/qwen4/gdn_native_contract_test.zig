//! Actual generated MSL recurrence bodies execute only as CPU coroutines; this does not qualify a GPU.
const std = @import("std");
const core = @import("core");
const common = @import("../../src/families/qwen4/gdn_contract.zig");
const a = std.testing.allocator;
extern fn tf_q4_gdn_native_cpu(c_int, c_int, [*]const [*]const u8, [*]u8, [*]u8, [*]u8, c_int, c_int, c_uint) void;
extern fn tf_q4_cuda_gdn_native_cpu(c_int, c_int, [*]const [*]const u8, [*]u8, [*]u8, c_int, c_int, c_uint) void;
fn Backend(comptime use_cuda: bool) type {
    return struct {
        const B = @This();
        pub const name = if (use_cuda) "host actual CUDA source" else "host actual MSL source";
        pub const Rig = @import("contract_rig.zig").HostRig;
        pub const served: []const core.contract.spec.Shape = &.{ .{ .rows = 1, .n = 1, .k = 2 }, .{ .rows = 16, .n = 1, .k = 2 } };
        inputs: [7][*]const u8 = undefined,
        output: []u8 = undefined,
        state: []u8 = undefined,
        tail: []u8 = undefined,
        control: common.Control = .production,
        pub fn bind(b: *B, r: *Rig, f: common.Fixture, control: common.Control) !void {
            b.control = control;
            for (f.inputs, 0..) |bytes, i| b.inputs[i] = (try r.add(bytes, bytes.len)).ptr;
            const sizes = try @import("../../src/families/qwen4/cuda_gdn.zig").sizes(f.p);
            if (f.part != .conv) b.tail = try r.add(null, f.p.rows * sizes.conv_state);
            if (f.part == .readout) {
                b.state = try r.add(null, sizes.snapshots);
                b.output = try r.add(null, sizes.output);
            } else {
                b.output = try r.add(null, sizes.output);
                b.state = try r.add(null, sizes.snapshots);
            }
            if (f.part == .conv) b.tail = try r.add(null, f.p.rows * sizes.conv_state);
        }
        pub fn launch(b: *B, _: *Rig, f: common.Fixture, repeat: u32) !void {
            std.debug.assert(f.p.keys == 1 and f.p.values == 2);
            if (use_cuda) tf_q4_cuda_gdn_native_cpu(@intCast(@backingInt(f.p.gate)), @intCast(@backingInt(b.control)), &b.inputs, b.output.ptr, b.state.ptr, @intCast(f.p.rows), f.p.weight_flags, repeat) else tf_q4_gdn_native_cpu(@intCast(@backingInt(f.p.gate)), @intCast(@backingInt(b.control)), &b.inputs, b.output.ptr, b.state.ptr, b.tail.ptr, @intCast(f.p.rows), f.p.weight_flags, repeat);
        }
    };
}
fn exercise(comptime use_cuda: bool) !void {
    const g = core.contract.gates;
    const Case = common.Case(core.contract, Backend(use_cuda));
    const old = g.quiet;
    g.quiet = true;
    defer g.quiet = old;
    for ([_]@import("../../src/families/qwen4/cuda_gdn.zig").Gate{ .silu, .sigmoid }) |gate_type| for ([_]u4{ 3, 7, 11, 15 }) |flags| for (std.enums.values(common.Part)) |part| {
        if (part == .state and flags != 3 and flags != 11) continue;
        if (part == .conv and (use_cuda or flags != 11)) continue;
        var good: Case = .{ .a = a, .rig = .{}, .backend = .{}, .gate = gate_type, .flags = flags, .part = part };
        defer good.deinit();
        for ([_]g.Gate{ .nonfinite, .repeat, .shapes, .sums }) |which| {
            const got = try g.one(a, which, good.kernel(), 0x367);
            if (got != (if (part == .conv and which == .sums) g.Verdict.na else g.Verdict.pass)) {
                std.debug.print("NATIVE_GDN {t} flags{d} {t} {t}: {t}\n", .{ gate_type, flags, part, which, got });
                g.quiet = false;
                _ = try g.one(a, which, good.kernel(), 0x367);
                g.quiet = true;
            }
            try std.testing.expectEqual(if (part == .conv and which == .sums) g.Verdict.na else g.Verdict.pass, got);
        }
        for ([_]common.Control{ .nan, .repeat, .tail, .sums }) |control| {
            if (control == .sums and (flags != 11 or part == .conv)) continue;
            var bad: Case = .{ .a = a, .rig = .{}, .backend = .{}, .gate = gate_type, .flags = flags, .part = part, .control = control };
            defer bad.deinit();
            const which = bad.kernel().control.?;
            const got = try g.one(a, which, bad.kernel(), 0x367);
            if (got != .fail) std.debug.print("NATIVE_GDN control {t} flags{d} {t} {t}: {t}\n", .{ gate_type, flags, part, control, got });
            try std.testing.expectEqual(g.Verdict.fail, got);
        }
    };
}
test "Qwen4 GDN native contract: real MSL readout and state bodies with source controls and f64 bounds" {
    try exercise(false);
}
test "Qwen4 GDN native contract: real CUDA readout and state bodies with source controls and f64 bounds" {
    try exercise(true);
}
fn negativeSilu(comptime use_cuda: bool) !void {
    const Case = common.Case(core.contract, Backend(use_cuda));
    var probe: Case = .{ .a = a, .rig = .{}, .backend = .{}, .gate = .sigmoid };
    defer probe.deinit();
    probe.arena = .init(a);
    probe.fixture = try common.Fixture.negativeSilu(probe.arena.?.allocator());
    try probe.backend.bind(&probe.rig, probe.fixture, .production);
    try probe.run();
    const bytes = try probe.rig.outputBytes();
    for (probe.fixture.want, probe.fixture.error_bound, 0..) |expected, limit, i| {
        const actual = core.contract.spec.read(.bf16, bytes, i);
        if (@abs(actual - expected) > limit) std.debug.print("GDN_NEGATIVE_SILU cuda{} element{d}: got{e}, expected{e}, bound{e}\n", .{ use_cuda, i, actual, expected, limit });
        try std.testing.expect(@abs(actual - expected) <= limit);
    }
    try std.testing.expect(probe.fixture.want[0] != 0);
}
test "Qwen4 GDN native contract: negative SiLU keeps nonzero BF16 MSL activations" {
    try negativeSilu(false);
}
test "Qwen4 GDN native contract: negative SiLU keeps nonzero BF16 CUDA activations" {
    try negativeSilu(true);
}
