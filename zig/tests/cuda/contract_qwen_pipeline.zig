//! Whole BF16 prompt attention replacement: identical bytes, long/short/causal tiles and adversarial maxima.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const prompt = @import("qwen35").prompt;
const check = @import("check.zig");
const rig_ = @import("contract_rig.zig");
const c = core.contract.spec;
const g = core.contract.gates;
const D = 256;
const H = 6;
const served = [_]c.Shape{
    .{ .rows = 1, .n = D, .k = 1 },         .{ .rows = 1, .n = D, .k = 63 },
    .{ .rows = 63, .n = D, .k = 63 },       .{ .rows = 64, .n = D, .k = 64 },
    .{ .rows = 65, .n = D, .k = 65 },       .{ .rows = 127, .n = D, .k = 128 },
    .{ .rows = 129, .n = D, .k = 257 },     .{ .rows = 64, .n = D, .k = 2112 },
    .{ .rows = 65, .n = D, .k = 28737 },    .{ .rows = 64, .n = D, .k = 127040 },
    .{ .rows = 4096, .n = D, .k = 32768 },  .{ .rows = 4096, .n = D, .k = 131072 },
    .{ .rows = 1024, .n = D, .k = 132096 },
};
fn noSums(_: c.Shape) usize {
    return 0;
}

pub const Product = struct {
    rig: rig_.Rig,
    old: cuda.Function,
    new: cuda.Function,
    shape: c.Shape = undefined,
    q: u64 = 0,
    k: u64 = 0,
    v: u64 = 0,
    out: u64 = 0,
    baseline: u64 = 0,
    adversarial: bool = false,
    shifted: bool = false,

    pub fn deinit(p: *Product) void {
        p.rig.deinit();
    }
    pub fn prepare(p: *Product, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.n != D or shape.rows == 0 or shape.rows > shape.k or shape.k > 132096) return false;
        p.rig.reset();
        p.shape = shape;
        const a = p.rig.gpu.gpa;
        const q = try a.alloc(u16, shape.rows * H * D);
        defer a.free(q);
        const k = try a.alloc(u16, shape.k * D);
        defer a.free(k);
        const v = try a.alloc(u16, shape.k * D);
        defer a.free(v);
        var rng = std.Random.DefaultPrng.init(seed);
        for (q) |*x| x.* = c.toBf16(rng.random().float(f32) * 2 - 1);
        for (k) |*x| x.* = c.toBf16(rng.random().float(f32) * 2 - 1);
        for (v) |*x| x.* = c.toBf16(rng.random().float(f32) * 2 - 1);
        if (p.adversarial) {
            @memset(q, 0);
            @memset(k, 0);
            for (0..shape.rows * H) |r| q[r * D] = c.toBf16(1);
            const maxima = [_]f32{ 0, 0, 64, -64, 128, 128, -128, 192 };
            for (0..shape.k) |r| {
                k[r * D] = c.toBf16(maxima[(r / 64) % maxima.len]);
                for (0..D) |d| v[r * D + d] = c.toBf16(if (r % 4 < 2) (if (d % 2 == 0) @as(f32, 32768) else 1) else (if (d % 2 == 0) @as(f32, -32768) else -1));
            }
        }
        if (fill == .nonfinite) {
            q[0] = 0x7fc1;
            q[D + 1] = 0x7f80;
            q[2 * D + 2] = 0xff80;
            k[0] = 0x7fc7;
            v[v.len - 1] = 0x7f80;
        }
        p.q = try p.rig.input(std.mem.sliceAsBytes(q));
        p.k = try p.rig.input(std.mem.sliceAsBytes(k));
        p.v = try p.rig.input(std.mem.sliceAsBytes(v));
        p.out = try p.rig.output(p.count());
        p.baseline = try p.rig.scratch(p.count());
        return true;
    }
    pub fn count(p: *Product) usize {
        return p.shape.rows * H * D * 2;
    }
    pub fn reference(_: *Product, _: []f64, _: []f64) !void {}
    fn launch(p: *Product, f: cuda.Function, out: u64) !void {
        var args: cuda.Args = .{};
        for ([_]u64{ p.q, p.k, p.v, out }) |ptr| args.add(ptr);
        for ([_]usize{ p.shape.k - p.shape.rows, p.shape.rows, H, 1, H }) |n| args.add(@as(c_int, @intCast(n)));
        args.add(@as(f32, 1.0 / 16.0));
        args.add(@as(c_int, @intCast(p.shape.k)));
        try cuda.launch.launch(f, .{ .grid = .{ .x = @intCast((p.shape.rows + 63) / 64), .y = H / 2 }, .block = .{ .x = 256 }, .shared = prompt.smem }, p.rig.s, &args);
        try p.rig.s.synchronize();
    }
    pub fn run(p: *Product) !void {
        try p.launch(p.new, p.out + @as(u64, if (p.shifted) 4 else 0));
    }
    pub fn replaced(p: *Product) ![]const u8 {
        try p.launch(p.old, p.baseline);
        return p.rig.bytes(p.baseline, p.count());
    }
    pub fn check(p: *Product, _: c.Gate) !?[]const u8 {
        const got = try p.rig.gpu.gpa.dupe(u8, try p.rig.bytes(p.out, p.count()));
        defer p.rig.gpu.gpa.free(got);
        return if (std.mem.eql(u8, got, try p.replaced())) null else "prompt pipeline differs from the original output bytes";
    }
    const vt = g.hooks(Product);
    pub fn kernel(p: *Product) g.Kernel {
        return .{ .spec = .{ .name = if (p.shifted) "control: Qwen prompt pipeline shifts aligned output" else if (p.adversarial) "cuda Qwen prompt pipeline adversarial exact bytes" else "cuda Qwen prompt pipeline exact bytes", .out = .raw, .accum = .none, .depth = noSums, .served = &served, .replaces = if (p.shifted) null else "hpattn_whole_kernel<256,8,2,2>", .nan = .{ .stated = "all output bytes including exceptional payloads equal the original kernel" } }, .ctx = p, .vt = &vt, .control = if (p.shifted) .shapes else null };
    }
};

pub const Set = struct {
    products: [3]Product = undefined,
    active: bool = false,
    pub fn append(set: *Set, gpu: check.Gpu, s: cuda.Stream, module: cuda.Module, kernels: *std.ArrayList(g.Kernel)) !void {
        if (try gpu.ctx.capability() != 121) return;
        const old = try module.function(prompt.symbols.bf16_whole);
        var f: prompt.Fns = .{ .bf16 = old, .fp8 = old, .bf16_whole = old, .fp8_whole = old, .bf16_p256 = old, .bf16_p256_ids = old, .chain = old };
        try old.allowDynamicShared(prompt.smem);
        try prompt.enablePipeline(&f, module, true);
        const new = f.pipeline.?;
        std.debug.print("RESOURCES prompt_pipeline old/new regs={d}/{d} static={d}/{d} local={d}/{d} CTAs={d}/{d} dynamic={d}\n", .{ try old.attribute(.num_regs), try new.attribute(.num_regs), try old.attribute(.shared_size_bytes), try new.attribute(.shared_size_bytes), try old.attribute(.local_size_bytes), try new.attribute(.local_size_bytes), try old.occupancy(256, prompt.smem), try new.occupancy(256, prompt.smem), prompt.smem });
        for (&set.products, 0..) |*p, i| p.* = .{ .rig = .{ .gpu = gpu, .s = s }, .old = old, .new = new, .adversarial = i == 1, .shifted = i == 2 };
        set.active = true;
        for (&set.products) |*p| try kernels.append(gpu.gpa, p.kernel());
    }
    pub fn deinit(set: *Set) void {
        if (set.active) for (&set.products) |*p| p.deinit();
    }
};

pub fn run(gpu: check.Gpu) !void {
    try check.expect(try gpu.ctx.capability() == 121, "Qwen prompt pipeline check needs GB10 sm121", .{});
    var module = try cuda.Module.load(gpu.d, cuda.kernels.prefill_attention);
    defer module.unload();
    var s = try cuda.Stream.init(gpu.d, false);
    defer s.deinit();
    var set: Set = .{};
    defer set.deinit();
    var kernels: std.ArrayList(g.Kernel) = .empty;
    defer kernels.deinit(gpu.gpa);
    try set.append(gpu, s, module, &kernels);
    var tally: g.Tally = .{};
    try g.runAll(gpu.gpa, kernels.items, 0x466, &tally);
    std.debug.print("SUMMARY qwen_prompt_pipeline: {d} pass, {d} fail, {d} controls red, {d} missed\n", .{ tally.passed, tally.failed, tally.caught, tally.missed });
    try check.expect(tally.problems() == 0, "Qwen prompt pipeline contract failed", .{});
}
