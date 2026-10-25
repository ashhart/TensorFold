//! Real generated QSA frontend bodies execute with CPU collectives and a binary64 node reference.
const std = @import("std");
const ct = @import("core").contract;
const common = @import("../../src/families/qwen4/qsa_norm_contract.zig");
const a = std.testing.allocator;
extern fn tf_q4_qsa_norm_native_cpu(c_int, c_int, [*]const [*]const u8, [*]u8, [*]u8, [*]u8, c_int, c_uint) void;
const Backend = struct {
    pub const name = "host actual MSL";
    pub const Rig = @import("contract_rig.zig").HostRig;
    pub const served: []const ct.spec.Shape = &.{ .{ .rows = 1, .n = 24, .k = 4 }, .{ .rows = 16, .n = 24, .k = 4 } };
    input: [6][*]const u8 = undefined,
    output: [3][*]u8 = undefined,
    control: common.Control = .production,
    pub fn bind(b: *@This(), r: *Rig, f: common.Fixture, control: common.Control) !void {
        b.control = control;
        for (f.inputs[0..f.count], 0..) |bytes, i| b.input[i] = (try r.add(bytes, bytes.len)).ptr;
        if (f.part == .pool) {
            b.output[0] = (try r.add(null, f.want.len * 2)).ptr;
            b.output[1] = b.output[0];
            b.output[2] = b.output[0];
            return;
        }
        const selected = @backingInt(f.part);
        const lengths = [_]usize{ f.rows * 24 * 256 * 2, f.rows * 4 * 256 * 2, f.rows * 4 * 128 * 2 };
        for (lengths, 0..) |n, i| if (i != selected) {
            b.output[i] = (try r.add(null, n)).ptr;
        };
        b.output[selected] = (try r.add(null, f.want.len * 2)).ptr;
    }
    pub fn launch(b: *@This(), _: *Rig, f: common.Fixture, repeat: u32) !void {
        tf_q4_qsa_norm_native_cpu(@intFromBool(f.part == .pool), @intCast(@backingInt(b.control)), &b.input, b.output[0], b.output[1], b.output[2], @intCast(f.rows), repeat);
    }
};
test "Qwen4 QSA native contract: query key index and pooling actual source meets binary64 nodes" {
    const Case = common.Case(ct, Backend);
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
            if (got != .fail) std.debug.print("QSA_NORM_CONTROL {t} {t}: {t}\n", .{ part, control, got });
            try std.testing.expectEqual(g.Verdict.fail, got);
        }
    }
}
