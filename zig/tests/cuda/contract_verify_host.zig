//! Registration and real numerical gate routing for the wider affine-4 verify kernel.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const mm = @import("contract_mm.zig");
const c = core.contract;
const g = c.gates;
const t = std.testing;
fn find(fp64: bool) !c.spec.Spec {
    for (std.enums.values(mm.Kind)) |kind| {
        const s = mm.spec(kind);
        const wanted = if (fp64) "cuda affine4 verify32 FP64" else "cuda affine4 verify32 one-row bits";
        if (std.mem.eql(u8, s.name, wanted)) return s;
    }
    return error.VerifyContractMissing;
}

test "verify contract: numerical and exact one-row comparisons both cover the new row tile" {
    for ([_]bool{ false, true }) |fp64| {
        const s = try find(fp64);
        try t.expectEqual(c.spec.Accum.f32, s.accum);
        try t.expectEqual(c.spec.Elem.bf16, s.out);
        try t.expectEqual(!fp64, s.replaces != null);
        var widest: usize = 0;
        for (s.served) |shape| widest = @max(widest, shape.rows);
        try t.expectEqual(@as(usize, 32), widest);
        try t.expectEqual(@as(usize, 32), s.tiles[0].size);
    }
}

test "verify contract: the one-CTA and clustered launch remain explicitly unmeasured" {
    var found: usize = 0;
    for (c.registry.all_cuda) |r| if (std.mem.eql(u8, r.launch.symbol, cuda.qlinear.symbols.verify32)) {
        found += 1;
        try t.expectEqual(@as(u32, 128), r.launch.threads);
        try t.expectEqual(@as(u32, 32640), r.launch.dynamic);
        try t.expect(r.launch.opt_in);
        for (r.usage) |u| try t.expectEqual(@as(u32, 0), u.regs);
        try t.expect(try c.registry.checkCuda(t.allocator, &.{r}, true) > 0);
    };
    try t.expectEqual(@as(usize, 2), found);
}

/// qmm_frag.cuh's LaneTile<64, rows, 64, 1, 4, stages>::SMEM: the stages' inputs, weights, scales and sums, or the
/// parked K-slice partials when those are larger.
fn laneTileSmem(rows: u32, stages: u32) u32 {
    const stage = rows * 128 + 64 * 64 / 2 + 2 * 64 * 2 + rows * 4;
    return @max(stages * stage, rows / 16 * 2 * 4 * 128 * 4);
}

test "verify contract: 64- and 128-row tiles cover the drafter's shapes and stay unmeasured until receipts" {
    const Wide = struct { tile: usize, bits: mm.Kind, fp64: mm.Kind, symbol: []const u8, shared: u32 };
    try t.expectEqual(laneTileSmem(32, 5), cuda.qlinear.verify_smem);
    for ([_]Wide{
        .{ .tile = 64, .bits = .verify64_bits, .fp64 = .verify64_fp64, .symbol = cuda.qlinear.symbols.verify64, .shared = laneTileSmem(64, 4) },
        .{ .tile = 128, .bits = .verify128_bits, .fp64 = .verify128_fp64, .symbol = cuda.qlinear.symbols.verify128, .shared = laneTileSmem(128, 3) },
    }) |w| {
        for ([_]mm.Kind{ w.bits, w.fp64 }) |kind| {
            const s = mm.spec(kind);
            try t.expectEqual(kind == w.bits, s.replaces != null);
            var widest: usize = 0;
            var drafter_k = false;
            for (s.served) |shape| {
                widest = @max(widest, shape.rows);
                drafter_k = drafter_k or shape.k == 5120;
            }
            try t.expectEqual(w.tile, widest);
            try t.expect(drafter_k);
            try t.expectEqual(w.tile, s.tiles[0].size);
            try t.expectEqual(w.tile, mm.verifyTile(kind));
        }
        var found: usize = 0;
        for (c.registry.all_cuda) |r| if (std.mem.eql(u8, r.launch.symbol, w.symbol)) {
            found += 1;
            try t.expectEqual(w.shared, r.launch.dynamic);
            try t.expect(r.launch.opt_in);
            for (r.usage) |u| try t.expectEqual(@as(u32, 0), u.regs);
        };
        try t.expectEqual(@as(usize, 2), found);
    }
    try t.expectEqual(laneTileSmem(64, 4), cuda.qlinear.verify64_smem);
    try t.expectEqual(laneTileSmem(128, 3), cuda.qlinear.verify128_smem);
}

const Cpu = struct {
    bits: [2]u8 = .{ 0x80, 0x3f },
    base: [2]u8 = .{ 0x80, 0x3f },
    references: usize = 0,
    replacements: usize = 0,
    const vt: g.VTable = .{ .prepare = prepare, .run = run, .output = output, .inputs = inputs, .intact = intact, .overlap = overlap, .count = count, .reference = reference, .replaced = replaced };
    fn self(p: *anyopaque) *Cpu {
        return @ptrCast(@alignCast(p));
    }
    fn prepare(_: *anyopaque, _: c.spec.Shape, _: g.Fill, _: u64) anyerror!bool {
        return true;
    }
    fn run(_: *anyopaque) anyerror!void {}
    fn output(p: *anyopaque) anyerror![]const u8 {
        return &self(p).bits;
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
    fn reference(p: *anyopaque, want: []f64, mag: []f64) anyerror!void {
        self(p).references += 1;
        want[0] = 1;
        mag[0] = 1;
    }
    fn replaced(p: *anyopaque) anyerror![]const u8 {
        self(p).replacements += 1;
        return &self(p).base;
    }
};

test "verify contract: gate 3 uses FP64 and exact bits separately and rejects each planted error" {
    const quiet = g.quiet;
    g.quiet = true;
    defer g.quiet = quiet;
    for ([_]bool{ false, true }) |fp64| {
        var cpu: Cpu = .{};
        const k: g.Kernel = .{ .spec = try find(fp64), .ctx = &cpu, .vt = &Cpu.vt };
        try t.expectEqual(g.Verdict.pass, try g.shapes(t.allocator, k, 1234));
        try t.expectEqual(fp64, cpu.references > 0);
        try t.expectEqual(!fp64, cpu.replacements > 0);
        if (fp64) cpu.bits = .{ 0, 0 } else cpu.base = .{ 0, 0 };
        try t.expectEqual(g.Verdict.fail, try g.shapes(t.allocator, k, 1234));
    }
}

const Baseline = struct {
    var calls: usize = 0;
    var all_single = true;
    var rows: c_int = 0;
    fn launch(_: cuda.abi.Function, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: cuda.abi.Stream, args: ?[*]?*anyopaque, _: ?[*]?*anyopaque) callconv(.c) cuda.abi.Result {
        calls += 1;
        rows = @as(*const c_int, @ptrCast(@alignCast(args.?[3].?))).*;
        all_single = all_single and @as(*const c_int, @ptrCast(@alignCast(args.?[3].?))).* == 1;
        return 0;
    }
    fn sync(_: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
        return 0;
    }
    fn copy(dst: ?*anyopaque, src: u64, len: usize) callconv(.c) cuda.abi.Result {
        const p: [*]const u8 = @ptrFromInt(src);
        const out: [*]u8 = @ptrCast(dst.?);
        @memcpy(out[0..len], p[0..len]);
        return 0;
    }
};

test "verify contract: the actual replacement producer really runs 32 separate original one-row calls" {
    var d: cuda.Driver = undefined;
    d.api.cuLaunchKernel = Baseline.launch;
    d.api.cuStreamSynchronize = Baseline.sync;
    d.api.cuMemcpyDtoH_v2 = Baseline.copy;
    var a: cuda.qlinear.Affine4Kernels = undefined;
    a.pdl = false;
    a.split = null;
    a.group_fn = .{ .d = &d, .handle = @ptrFromInt(2) };
    var data: [32 * 64]u16 = @splat(0x3f80);
    var product: mm.Mm = .{ .rig = .{ .gpu = .{ .d = &d, .ctx = undefined, .gpa = t.allocator, .io = undefined }, .s = .{ .d = &d, .handle = null } }, .kind = .verify32_bits, .sm = 121, .affine4 = &a, .base = @intFromPtr(&data), .q = .{ .w = 1, .s = 2, .b = 3, .n = 64, .k = 512, .npad = 128 }, .f = .{ .a = t.allocator, .shape = .{ .rows = 32, .n = 64, .k = 512 }, .bits = 4, .group = 64, .x = undefined, .codes = undefined, .scales = undefined, .biases = undefined } };
    defer product.rig.host.deinit(t.allocator);
    Baseline.calls = 0;
    Baseline.all_single = true;
    const got = try product.replaced();
    try t.expectEqual(@as(usize, 32), Baseline.calls);
    try t.expect(Baseline.all_single);
    try t.expectEqualSlices(u8, std.mem.sliceAsBytes(&data), got);
}

test "verify contract: the actual FP64 producer covers every row including the second row tile" {
    var f = try c.affine.Affine.init(t.allocator, .{ .rows = 32, .n = 1, .k = 64 }, 4, 64);
    defer f.deinit();
    for (0..32) |row| @memset(f.x[row * 64 ..][0..64], c.spec.toBf16(@floatFromInt(row + 1)));
    @memset(f.codes, 3);
    f.scales[0] = c.spec.toBf16(0.5);
    f.biases[0] = c.spec.toBf16(-1);
    var product: mm.Mm = .{ .rig = undefined, .kind = .verify32_fp64, .sm = 121, .f = f };
    var want: [32]f64 = undefined;
    var mag: [32]f64 = undefined;
    try product.reference(&want, &mag);
    for (want, 0..) |v, row| try t.expectEqual(@as(f64, @floatFromInt(32 * (row + 1))), v);
}

test "verify contract: both compiled device controls really truncate the new launch at row sixteen" {
    var d: cuda.Driver = undefined;
    d.api.cuLaunchKernel = Baseline.launch;
    d.api.cuStreamSynchronize = Baseline.sync;
    var a: cuda.qlinear.Affine4Kernels = undefined;
    a.pdl = false;
    a.verify_fn = .{ .d = &d, .handle = @ptrFromInt(2) };
    for ([_]mm.Kind{ .verify32_bits, .verify32_fp64 }) |kind| {
        var product: mm.Mm = .{ .rig = .{ .gpu = .{ .d = &d, .ctx = undefined, .gpa = t.allocator, .io = undefined }, .s = .{ .d = &d, .handle = null } }, .kind = kind, .sm = 121, .affine4 = &a, .truncate_verify = true, .q = .{ .w = 1, .s = 2, .b = 3, .n = 64, .k = 512, .npad = 128 }, .f = .{ .a = t.allocator, .shape = .{ .rows = 32, .n = 64, .k = 512 }, .bits = 4, .group = 64, .x = undefined, .codes = undefined, .scales = undefined, .biases = undefined } };
        try t.expectEqual(c.spec.Gate.shapes, product.kernel().?.control.?);
        try t.expectEqual(@as(usize, 32), product.f.?.shape.rows);
        try product.run();
        try t.expectEqual(@as(c_int, 16), Baseline.rows);
    }
}
