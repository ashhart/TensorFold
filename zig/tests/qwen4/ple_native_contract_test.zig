//! Real generated PLE MSL runs on CPU coroutines with the same independent Cases as its device collector.
const std = @import("std");
const ct = @import("core").contract;
const common = @import("../../src/families/qwen4/ple_contract.zig");
const a = std.testing.allocator;
extern fn tf_q4_ple_native_cpu(c_int, c_int, [*]const [*]const u8, [*]u8, [*]u8, [*]u8, [*]u8, c_int, c_uint) void;
extern fn tf_q4_cuda_ple_native_cpu(c_int, [*]const [*]const u8, [*]u8, [*]u8, [*]u8, [*]u8, c_int, c_int, c_int, c_uint) void;
fn Backend(comptime use_cuda: bool) type {
    return struct {
        pub const name = if (use_cuda) "host actual CUDA" else "host actual MSL";
        pub const Rig = @import("contract_rig.zig").HostRig;
        pub const served: []const ct.spec.Shape = &.{ .{ .rows = 1, .n = 256, .k = 2 }, .{ .rows = 16, .n = 256, .k = 2 }, .{ .rows = 1, .n = 5120, .k = 4 } };
        input: [7][*]const u8 = undefined,
        output: [4][]u8 = undefined,
        control: common.Control = .production,
        pub fn bind(b: *@This(), r: *Rig, f: common.Fixture, control: common.Control) !void {
            b.control = control;
            for (f.inputs, 0..) |bytes, i| b.input[i] = (try r.add(bytes, bytes.len)).ptr;
            const w = f.p.hidden * f.p.streams;
            for (0..4) |i| if (i != @backingInt(f.part)) {
                b.output[i] = try r.add(null, f.p.rows * w * 2 * @as(usize, if (i == 3) 9 else 1));
            };
            b.output[@backingInt(f.part)] = try r.add(null, f.want.len * 2);
        }
        pub fn launch(b: *@This(), _: *Rig, f: common.Fixture, repeat: u32) !void {
            if (use_cuda) tf_q4_cuda_ple_native_cpu(@intCast(@backingInt(b.control)), &b.input, b.output[0].ptr, b.output[1].ptr, b.output[2].ptr, b.output[3].ptr, @intCast(f.p.rows), @intCast(f.p.hidden), @intCast(f.p.streams), repeat) else tf_q4_ple_native_cpu(@intFromBool(f.p.hidden == 5120), @intCast(@backingInt(b.control)), &b.input, b.output[0].ptr, b.output[1].ptr, b.output[2].ptr, b.output[3].ptr, @intCast(f.p.rows), repeat);
        }
    };
}
fn exercise(comptime use_cuda: bool) !void {
    const Case = common.Case(ct, Backend(use_cuda));
    const g = ct.gates;
    const old = g.quiet;
    g.quiet = true;
    defer g.quiet = old;
    for (std.enums.values(common.Part)) |part| {
        var good: Case = .{ .a = a, .rig = .{}, .backend = .{}, .part = part };
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
            var bad: Case = .{ .a = a, .rig = .{}, .backend = .{}, .part = part, .control = control };
            defer bad.deinit();
            const got = try g.one(a, bad.kernel().control.?, bad.kernel(), 0x367);
            if (got != .fail) std.debug.print("PLE_CONTROL {t} {t}: {t}\n", .{ part, control, got });
            try std.testing.expectEqual(g.Verdict.fail, got);
        }
    }
}

fn denominator(comptime use_cuda: bool) !void {
    const Case = common.Case(ct, Backend(use_cuda));
    var probe: Case = .{ .a = a, .rig = .{}, .backend = .{}, .part = .gated };
    defer probe.deinit();
    probe.arena = .init(a);
    probe.fixture = try common.Fixture.denominator(probe.arena.?.allocator());
    try probe.backend.bind(&probe.rig, probe.fixture, .production);
    try probe.run();
    const bytes = try probe.rig.outputBytes();
    for (probe.fixture.want, probe.fixture.error_bound, 0..) |want, bound, i| {
        const got = ct.spec.read(.bf16, bytes, i);
        if (@abs(got - want) > bound) std.debug.print("PLE_DENOMINATOR element{d}: got{e}, want{e}, bound{e}\n", .{ i, got, want, bound });
        try std.testing.expect(@abs(got - want) <= bound);
    }
}

test "Qwen4 PLE native contract: generated Metal gate convolution and saved tails follow binary64 nodes" {
    try exercise(false);
}
test "Qwen4 PLE native contract: generated CUDA gate convolution and saved tails follow binary64 nodes" {
    try exercise(true);
}
test "Qwen4 PLE native contract: Metal 5120-wide gate uses the full scalar square root" {
    try denominator(false);
}
test "Qwen4 PLE native contract: CUDA 5120-wide gate uses the full scalar square root" {
    try denominator(true);
}

fn negativeSilu(comptime use_cuda: bool) !void {
    const Case = common.Case(ct, Backend(use_cuda));
    var probe: Case = .{ .a = a, .rig = .{}, .backend = .{}, .part = .output };
    defer probe.deinit();
    probe.arena = .init(a);
    probe.fixture = try common.Fixture.negativeSilu(probe.arena.?.allocator());
    try probe.backend.bind(&probe.rig, probe.fixture, .production);
    try probe.run();
    const bytes = try probe.rig.outputBytes();
    for (probe.fixture.want, probe.fixture.error_bound, 0..) |want, bound, i| {
        const got = ct.spec.read(.bf16, bytes, i);
        if (@abs(got - want) > bound) std.debug.print("PLE_NEGATIVE_SILU cuda{} element{d}: got{e}, want{e}, bound{e}\n", .{ use_cuda, i, got, want, bound });
        try std.testing.expect(@abs(got - want) <= bound);
    }
    try std.testing.expect(probe.fixture.want[0] != 0);
}
test "Qwen4 PLE native contract: Metal preserves a representable negative SiLU" {
    try negativeSilu(false);
}
test "Qwen4 PLE native contract: CUDA preserves a representable negative SiLU" {
    try negativeSilu(true);
}
