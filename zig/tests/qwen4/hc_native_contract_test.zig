//! The actual generated Metal HC functions run as CPU coroutines with recorded BF16 and SIMD operations.
const std = @import("std");
const core = @import("core");
const common = @import("../../src/families/qwen4/hc_native_contract.zig");
const h = @import("../../src/families/qwen4/hc_layout.zig");
const a = std.testing.allocator;
extern fn tf_q4_hc_native_cpu(c_int, c_int, [*]const [*]const u8, [*]u8, c_int, c_int, c_uint) void;
const B = struct {
    pub const Rig = @import("contract_rig.zig").HostRig;
    inputs: [3][*]const u8 = undefined,
    out: []u8 = undefined,
    control: common.Control,
    pub fn bind(b: *B, r: *Rig, f: common.Fixture) !void {
        for (f.inputs[0..f.count], 0..) |bytes, i| b.inputs[i] = (try r.add(bytes, bytes.len)).ptr;
        b.out = try r.add(null, f.output_bytes);
    }
    pub fn launch(b: *B, _: *Rig, op: h.Op, p: h.Call, repeat: u32) !void {
        tf_q4_hc_native_cpu(@intCast(@backingInt(op)), @intCast(@backingInt(b.control)), &b.inputs, b.out.ptr, @intCast(p.rows), @intCast(p.width), repeat);
    }
};
test "Qwen4 HC native contract: actual source and every planted control" {
    const g = core.contract.gates;
    const Case = common.Case(core.contract, B);
    const old = g.quiet;
    g.quiet = true;
    defer g.quiet = old;
    for (std.enums.values(h.Op)) |op| {
        var good: Case = .{ .a = a, .rig = .{}, .backend = .{ .control = .production }, .op = op };
        defer good.deinit();
        for ([_]g.Gate{ .nonfinite, .repeat, .shapes, .sums }) |gate| {
            const got = try g.one(a, gate, good.kernel(), 0x367);
            if (got != (if (op == .act and gate == .sums) g.Verdict.na else .pass)) {
                std.debug.print("NATIVE_HC {t} {t}: {t}\n", .{ op, gate, got });
            }
            try std.testing.expectEqual(if (op == .act and gate == .sums) g.Verdict.na else .pass, got);
        }
        for ([_]common.Control{ .nan, .repeat, .tail, .sums }) |control| {
            if (control == .sums and op != .sum) continue;
            var bad: Case = .{ .a = a, .rig = .{}, .backend = .{ .control = control }, .op = op, .control = control };
            defer bad.deinit();
            const gate = bad.kernel().control.?;
            try std.testing.expectEqual(g.Verdict.fail, try g.one(a, gate, bad.kernel(), 0x367));
        }
    }
}

fn negativeDense(comptime Backend: type) !void {
    var arena: std.heap.ArenaAllocator = .init(a);
    defer arena.deinit();
    const f = try common.Fixture.init(arena.allocator(), .{ .hidden = 5120, .streams = 4 }, .act, .{ .rows = 1, .n = 257, .k = 1 }, .random, 0x367);
    @memset(std.mem.bytesAsSlice(u16, f.inputs[0]), core.contract.spec.toBf16(-89));
    @memset(std.mem.bytesAsSlice(u16, f.inputs[1]), core.contract.spec.toBf16(0x1p120));
    var rig: Backend.Rig = .{};
    defer rig.deinit();
    var backend: Backend = .{ .control = .production };
    try backend.bind(&rig, f);
    try backend.launch(&rig, .act, f.p, 0);
    const want = try arena.allocator().alloc(f64, 257);
    const mag = try arena.allocator().alloc(f64, 257);
    f.reference(want, mag, .none, .sequential);
    const out = try rig.outputBytes();
    try std.testing.expect(want[0] != 0);
    for (want, 0..) |v, i| {
        const got = core.contract.spec.read(.bf16, out, i);
        if (v != got) std.debug.print("DENSE_NEGATIVE_SILU element{d}: got{e}, want{e}\n", .{ i, got, v });
        try std.testing.expectEqual(v, got);
    }
}

extern fn tf_q4_cuda_dense_native_cpu(c_int, [*]const u16, [*]const u16, [*]u16, c_int, c_int, c_uint) void;
const DenseCuda = struct {
    pub const name = "host actual CUDA";
    pub const Rig = B.Rig;
    control: common.Control,
    gate: []const u16 = undefined,
    up: []const u16 = undefined,
    out: []u16 = undefined,
    pub fn bind(b: *@This(), r: *Rig, f: common.Fixture) !void {
        b.gate = @alignCast(std.mem.bytesAsSlice(u16, try r.add(f.inputs[0], f.inputs[0].len)));
        b.up = @alignCast(std.mem.bytesAsSlice(u16, try r.add(f.inputs[1], f.inputs[1].len)));
        b.out = @alignCast(std.mem.bytesAsSlice(u16, try r.add(null, f.output_bytes)));
    }
    pub fn launch(b: *@This(), _: *Rig, op: h.Op, p: h.Call, repeat: u32) !void {
        std.debug.assert(op == .act and b.control != .sums);
        tf_q4_cuda_dense_native_cpu(@intCast(@backingInt(b.control)), b.gate.ptr, b.up.ptr, b.out.ptr, @intCast(p.rows), @intCast(p.width), repeat);
    }
};
test "Qwen4 HC native contract: Metal dense SiLU keeps a representable up product" {
    try negativeDense(B);
}
test "Qwen4 HC native contract: CUDA dense SiLU keeps a representable up product" {
    try negativeDense(DenseCuda);
}
test "Qwen4 HC native contract: actual CUDA dense source and every planted control" {
    const g = core.contract.gates;
    const Case = common.Case(core.contract, DenseCuda);
    const old = g.quiet;
    g.quiet = true;
    defer g.quiet = old;
    var good: Case = .{ .a = a, .rig = .{}, .backend = .{ .control = .production }, .op = .act };
    defer good.deinit();
    for ([_]g.Gate{ .nonfinite, .repeat, .shapes, .sums }) |gate| try std.testing.expectEqual(if (gate == .sums) g.Verdict.na else g.Verdict.pass, try g.one(a, gate, good.kernel(), 0x367));
    for ([_]common.Control{ .nan, .repeat, .tail }) |control| {
        var bad: Case = .{ .a = a, .rig = .{}, .backend = .{ .control = control }, .op = .act, .control = control };
        defer bad.deinit();
        try std.testing.expectEqual(g.Verdict.fail, try g.one(a, bad.kernel().control.?, bad.kernel(), 0x367));
    }
}
