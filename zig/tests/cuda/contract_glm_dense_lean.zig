//! GLM dense lean tile replacement bytes, guarded repeats, nonfinite inputs, controls and focused profiling.
const std = @import("std");
const cuda = @import("cuda");
const glm = @import("glm_cuda");
const core = @import("core");
const rig_ = @import("contract_rig.zig");
const check_ = @import("check.zig");
const c = core.contract.spec;
const g = core.contract.gates;
const Prompt = cuda.nvfp4.Prompt;
pub const lengths = [_]usize{ 1, 17, 63, 64, 65, 127, 128, 129, 256, 512, 1024, 2048, 4096, 8192, 32768 };
fn shapes(comptime wide: bool) [lengths.len * 2]c.Shape {
    var out: [lengths.len * 2]c.Shape = undefined;
    for (lengths, 0..) |rows, i| for (0..2) |rank| {
        const width: usize = 12288 / (rank + 1);
        out[2 * i + rank] = .{ .rows = rows, .n = if (wide) 4096 else width, .k = if (wide) width else 4096 };
    };
    return out;
}
const up_shapes = shapes(false);
const down_shapes = shapes(true);
fn noSums(_: c.Shape) usize {
    return 0;
}
const rule = "every output byte equals the original 128x256 lean product on identical inputs";
pub const Product = struct {
    rig: rig_.Rig,
    prompt: *const Prompt,
    wide: bool,
    control: bool = false,
    rows: usize = 0,
    stride: usize = 0,
    w: cuda.qmmf.Weight = undefined,
    x: u64 = 0,
    out: u64 = 0,
    base: u64 = 0,
    pub fn kernel(p: *Product) g.Kernel {
        return .{ .spec = .{
            .name = if (p.control) (if (p.wide) "control: GLM dense lean down wrong scale" else "control: GLM dense lean up wrong scale") else (if (p.wide) "cuda GLM dense lean down replacement" else "cuda GLM dense lean up replacement"),
            .accum = .none,
            .depth = noSums,
            .out = .raw,
            .nan = .{ .stated = rule },
            .replaces = if (p.control) null else "prompt_lean<0,128,256,1,8,3>",
            .served = if (p.wide) &down_shapes else &up_shapes,
        }, .ctx = p, .vt = &g.hooks(Product), .control = if (p.control) .shapes else null };
    }
    pub fn deinit(p: *Product) void {
        p.rig.deinit();
    }
    pub fn prepare(p: *Product, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        p.rig.reset();
        if (shape.k % 64 != 0 or shape.n % 64 != 0 or shape.rows == 0) return false;
        p.rows = shape.rows;
        p.stride = shape.k + 8;
        const a = p.rig.gpu.gpa;
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        const x = try a.alloc(u16, p.rows * p.stride);
        defer a.free(x);
        @memset(x, 0x7fc1);
        for (0..p.rows) |r| {
            for (0..shape.k) |k| x[r * p.stride + k] = c.toBf16(rng.float(f32) * 4 - 2);
        }
        if (fill == .nonfinite) {
            for ([_]u16{ 0x7fc1, 0x7f80, 0xff80 }, 0..) |bits, i| x[(i % p.rows) * p.stride + i + 3] = bits;
            if (p.rows >= 4) {
                x[3 * p.stride + 3] = 0x7f80;
                x[3 * p.stride + 4] = 0xff80;
            }
        }
        p.x = try p.rig.input(std.mem.sliceAsBytes(x));
        const codes = try a.alloc(u8, shape.n * shape.k / 2);
        defer a.free(codes);
        const scales = try a.alloc(u8, shape.n * shape.k / 16);
        defer a.free(scales);
        for (codes) |*v| v.* = rng.int(u8);
        for (scales) |*v| v.* = 0x20 + rng.uintLessThan(u8, 0x29);
        if (fill == .nonfinite) scales[scales.len / 3] = 0x7f;
        const words = try a.alloc(u32, cuda.nvfp4.padded(shape.n) * shape.k / 8);
        defer a.free(words);
        const packed_scales = try a.alloc(u8, cuda.nvfp4.padded(shape.n) * shape.k / 16);
        defer a.free(packed_scales);
        cuda.nvfp4.packWords(words, codes, shape.n, shape.k);
        cuda.nvfp4.packScales(packed_scales, scales, shape.n, shape.k);
        p.w = cuda.nvfp4.weight(try p.rig.input(std.mem.sliceAsBytes(words)), try p.rig.input(packed_scales), 0.0123, shape.n, shape.k);
        p.out = try p.rig.output(p.rows * shape.n * @as(usize, if (p.wide) 4 else 2));
        p.base = try p.rig.scratch(p.count());
        return true;
    }
    pub fn run(p: *Product) !void {
        var w = p.w;
        if (p.control) w.scale *= 2;
        try p.prompt.denseMatmulOut(p.rig.s, p.x, p.stride, p.rows, w, p.out, if (p.wide) .f32 else .bf16);
        try p.rig.s.synchronize();
    }
    pub fn replaced(p: *Product) ![]const u8 {
        const b: cuda.DeviceBuffer = .{ .d = p.rig.gpu.d, .ptr = p.base, .len = p.count() };
        try b.fill8(0xff, p.rig.s.handle);
        try p.prompt.matmulOut(p.rig.s, p.x, p.stride, p.rows, p.w, p.base, if (p.wide) .f32 else .bf16);
        return p.rig.bytes(p.base, p.count());
    }
    pub fn check(p: *Product, _: g.Gate) !?[]const u8 {
        const a = p.rig.gpu.gpa;
        const got = try a.dupe(u8, try p.rig.outputBytes());
        defer a.free(got);
        return if (std.mem.eql(u8, got, try p.replaced())) null else rule;
    }
    pub fn count(p: *Product) usize {
        return p.rows * p.w.n * @as(usize, if (p.wide) 4 else 2);
    }
    pub fn reference(_: *Product, _: []f64, _: []f64) !void {}
};
pub fn run(gpu: check_.Gpu) !void {
    var s = try cuda.Stream.init(gpu.d, false);
    defer s.deinit();
    var prompt = try Prompt.load(gpu.d);
    defer prompt.unload();
    var products: [4]Product = undefined;
    for (&products, 0..) |*p, i| {
        p.* = .{ .rig = .{ .gpu = gpu, .s = s }, .prompt = &prompt, .wide = i % 2 == 1, .control = i >= 2 };
    }
    defer for (&products) |*p| p.deinit();
    try runPrepared(&products);
}
pub fn runPrepared(products: *[4]Product) !void {
    var kernels: [4]g.Kernel = undefined;
    for (products, &kernels) |*p, *kernel| kernel.* = p.kernel();
    var tally: g.Tally = .{};
    try g.runAll(products[0].rig.gpu.gpa, &kernels, 0x487, &tally);
    std.debug.print("SUMMARY glm_dense_lean: {d} pass, {d} fail; {d} controls red, {d} missed\n", .{ tally.passed, tally.failed, tally.caught, tally.missed });
    try check_.expect(tally.problems() == 0 and tally.caught == 2, "dense lean replacement gates failed", .{});
}
pub fn profile(gpu: check_.Gpu, kind: []const u8, rows: usize) !void {
    const wide = std.mem.eql(u8, kind, "dense-down");
    if (!wide and !std.mem.eql(u8, kind, "dense-up")) return error.BadProfileKind;
    if (rows == 0 or rows > 32768) return error.BadProfileRows;
    var s = try cuda.Stream.init(gpu.d, false);
    defer s.deinit();
    var prompt = try Prompt.load(gpu.d);
    defer prompt.unload();
    var p: Product = .{ .rig = .{ .gpu = gpu, .s = s }, .prompt = &prompt, .wide = wide };
    defer p.deinit();
    _ = try p.prepare(.{ .rows = rows, .n = if (wide) 4096 else 12288, .k = if (wide) 12288 else 4096 }, .random, 0x487);
    const enabled = try @import("sys").environment.read(bool, "TF_GLM_DENSE_LEAN_TILE", false);
    try profilePrepared(&p, enabled, kind);
}
pub fn profilePrepared(p: *Product, enabled: bool, kind: []const u8) !void {
    const gpu = p.rig.gpu;
    const s = p.rig.s;
    const prompt = p.prompt;
    const rows = p.rows;
    const wide = p.wide;
    const k: glm.mlp.Kernels = .{ .set = undefined, .linear = .{ .nvfp4_prompt = prompt }, .copy_rows = undefined, .dense_lean = enabled };
    for (0..3) |_| try glm.mlp.mm(k, s, p.w, p.x, p.stride, rows, p.out, wide, true);
    try s.synchronize();
    var begin = try cuda.Event.init(gpu.d, true);
    defer begin.deinit();
    var end = try cuda.Event.init(gpu.d, true);
    defer end.deinit();
    try begin.record(s);
    for (0..10) |_| try glm.mlp.mm(k, s, p.w, p.x, p.stride, rows, p.out, wide, true);
    try end.record(s);
    try end.synchronize();
    const ms = try begin.elapsedMs(end);
    const f = if (enabled) prompt.dense_lean[if (wide) 1 else 0] else if (wide) prompt.lean_f32 else prompt.lean;
    const regs = try f.attribute(.num_regs);
    const blocks = try f.occupancy(256, if (enabled) Prompt.denseLeanSharedBytes() else Prompt.leanSharedBytes());
    // Check profile output against the original too; timing is not a qualification gate.
    try check_.expect((try p.check(.shapes)) == null, "profile output differs from parent", .{});
    const bm: usize = if (enabled) 64 else 128;
    const bn: usize = if (enabled) 128 else 256;
    const grid = @min(prompt.sms * @as(usize, if (enabled) 2 else 1), ((rows + bm - 1) / bm) * ((p.w.n + bn - 1) / bn));
    std.debug.print("PROFILE {s} rows={d} dense_lean={d} grid={d} threads=256 registers={d} dynamic_shared={d} active_blocks_per_sm={d} ms={d:.6}\n", .{ kind, rows, @intFromBool(enabled), grid, regs, if (enabled) Prompt.denseLeanSharedBytes() else Prompt.leanSharedBytes(), blocks, ms / 10 });
}
