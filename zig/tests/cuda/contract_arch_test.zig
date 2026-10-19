//! Host-only checks of the CUDA contract's architecture selection and its real gate-3 routing.
const std = @import("std");
const mm = @import("contract_mm.zig");
const contract = @import("core").contract;
const g = contract.gates;

test "contract architecture: older SMs omit the standalone cluster kernels" {
    for ([_]u32{ 80, 86, 89 }) |sm| {
        var count: usize = 0;
        for (std.enums.values(mm.Kind)) |kind| {
            var m: mm.Mm = .{ .rig = undefined, .kind = kind, .sm = sm };
            const k = m.kernel();
            try std.testing.expectEqual((kind == .qmm_group or kind == .verify32_bits or kind == .verify32_fp64), k == null);
            if (k != null) count += 1;
        }
        try std.testing.expectEqual(@as(usize, 6), count);
    }
}

test "contract architecture: SM90 and Blackwell retain every kernel and the GEMV bit comparisons" {
    for ([_]u32{ 90, 120, 121 }) |sm| {
        var count: usize = 0;
        for (std.enums.values(mm.Kind)) |kind| {
            var m: mm.Mm = .{ .rig = undefined, .kind = kind, .sm = sm };
            const k = m.kernel().?;
            count += 1;
            try std.testing.expectEqual(kind == .lane_gemv or kind == .lane_gemv_split or kind == .verify32_bits, k.spec.replaces != null);
        }
        try std.testing.expectEqual(@as(usize, 9), count);
    }
}

test "contract architecture: legacy GEMV keeps every field except the unavailable replacement" {
    for ([_]u32{ 80, 86, 89 }) |sm| for ([_]mm.Kind{ .lane_gemv, .lane_gemv_split }) |kind| {
        var modern: mm.Mm = .{ .rig = undefined, .kind = kind, .sm = 121 };
        var legacy: mm.Mm = .{ .rig = undefined, .kind = kind, .sm = sm };
        var want = modern.kernel().?.spec;
        want.replaces = null;
        try std.testing.expectEqualDeep(want, legacy.kernel().?.spec);
    };
}

test "contract architecture: the legacy FP4 lane names the reduction fallback it actually selects" {
    for ([_]u32{ 80, 86, 89 }) |sm| {
        var m: mm.Mm = .{ .rig = undefined, .kind = .lane_f32, .sm = sm };
        const k = m.kernel().?;
        try std.testing.expect(std.mem.indexOf(u8, k.spec.name, "reduce_kernel") != null);
    }
}

const Cpu = struct {
    bytes: [2]u8 = .{ 0x80, 0x3f }, // bf16 1
    baseline: [2]u8 = .{ 0x80, 0x3f },
    references: usize = 0,
    replacements: usize = 0,
    const vt: g.VTable = .{ .prepare = prepare, .run = run, .output = output, .inputs = inputs, .intact = intact, .overlap = overlap, .count = count, .reference = reference, .replaced = replaced };

    fn self(ctx: *anyopaque) *Cpu {
        return @ptrCast(@alignCast(ctx));
    }
    fn prepare(_: *anyopaque, _: contract.spec.Shape, _: g.Fill, _: u64) anyerror!bool {
        return true;
    }
    fn run(_: *anyopaque) anyerror!void {}
    fn output(ctx: *anyopaque) anyerror![]const u8 {
        return &self(ctx).bytes;
    }
    fn inputs(_: *anyopaque) anyerror![]const u8 {
        return &.{};
    }
    fn intact(_: *anyopaque) anyerror!bool {
        return true;
    }
    fn overlap(_: *anyopaque) bool {
        return false;
    }
    fn count(_: *anyopaque) usize {
        return 1;
    }
    fn reference(ctx: *anyopaque, want: []f64, mag: []f64) anyerror!void {
        self(ctx).references += 1;
        want[0] = 1;
        mag[0] = 1;
    }
    fn replaced(ctx: *anyopaque) anyerror![]const u8 {
        self(ctx).replacements += 1;
        return &self(ctx).baseline;
    }
};

test "contract architecture: legacy gate 3 really checks FP64 and rejects a planted bad output" {
    const old = g.quiet;
    g.quiet = true;
    defer g.quiet = old;
    for ([_]mm.Kind{ .lane_gemv, .lane_gemv_split }) |kind| {
        var m: mm.Mm = .{ .rig = undefined, .kind = kind, .sm = 89 };
        var cpu: Cpu = .{};
        const k: g.Kernel = .{ .spec = m.kernel().?.spec, .ctx = &cpu, .vt = &Cpu.vt };
        try std.testing.expectEqual(g.Verdict.pass, try g.shapes(std.testing.allocator, k, 1));
        try std.testing.expect(cpu.references > 0);
        try std.testing.expectEqual(@as(usize, 0), cpu.replacements);
        cpu.bytes = .{ 0, 0 };
        try std.testing.expectEqual(g.Verdict.fail, try g.shapes(std.testing.allocator, k, 1));
        try std.testing.expectEqual(@as(usize, 0), cpu.replacements);
    }
}

test "contract architecture: modern gate 3 retains bit equality and rejects a planted baseline difference" {
    const old = g.quiet;
    g.quiet = true;
    defer g.quiet = old;
    for ([_]mm.Kind{ .lane_gemv, .lane_gemv_split }) |kind| {
        var m: mm.Mm = .{ .rig = undefined, .kind = kind, .sm = 121 };
        var cpu: Cpu = .{};
        const k: g.Kernel = .{ .spec = m.kernel().?.spec, .ctx = &cpu, .vt = &Cpu.vt };
        try std.testing.expectEqual(g.Verdict.pass, try g.shapes(std.testing.allocator, k, 1));
        try std.testing.expect(cpu.replacements > 0);
        try std.testing.expectEqual(@as(usize, 0), cpu.references);
        cpu.baseline = .{ 0, 0 };
        try std.testing.expectEqual(g.Verdict.fail, try g.shapes(std.testing.allocator, k, 1));
        try std.testing.expectEqual(@as(usize, 0), cpu.references);
    }
}
