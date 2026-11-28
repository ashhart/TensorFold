//! Shared prompt expert: whole-chunk preparation versus the original per-piece shared launches, raw FP32 output.
const std = @import("std");
const cuda = @import("cuda");
const glm = @import("glm_cuda");
const core = @import("core");
const rig_ = @import("contract_rig.zig");
const check_ = @import("check.zig");
const c = core.contract.spec;
const g = core.contract.gates;
fn depth(_: c.Shape) usize {
    return 0;
}
const shapes = [_]c.Shape{ .{ .rows = 2048, .n = 4096, .k = 1024 }, .{ .rows = 1537, .n = 4096, .k = 1024 }, .{ .rows = 129, .n = 4096, .k = 1024 }, .{ .rows = 17, .n = 4096, .k = 1024 }, .{ .rows = 1, .n = 4096, .k = 1024 } };
const Probe = struct {
    rig: rig_.Rig,
    mm: *const glm.mm.Kernels,
    glue: glm.glue.Kernels,
    put: cuda.Function,
    control: bool,
    l: glm.moe.Layer = undefined,
    sc: glm.moe.Scratch = undefined,
    x: u64 = 0,
    out: u64 = 0,
    base: u64 = 0,
    rows: usize = 0,
    const vt = g.hooks(Probe);
    fn kernel(p: *Probe) g.Kernel {
        return .{ .ctx = p, .vt = &vt, .control = if (p.control) .shapes else null, .spec = .{ .name = if (p.control) "control: GLM prefill shared expert missing first piece" else "cuda GLM prefill shared expert once", .accum = .none, .out = .raw, .depth = depth, .replaces = if (p.control) null else "shared_gu, clamped SwiGLU, shared_down, put_slot and combine per piece", .nan = .{ .stated = "every FP32 output byte equals the original per-piece path on identical BF16 input and weights" }, .served = &shapes } };
    }
    fn deinit(p: *Probe) void {
        p.rig.deinit();
    }
    fn input(p: *Probe, n: usize, rng: std.Random, fill: g.Fill) !u64 {
        const a = p.rig.gpu.gpa;
        const words = try a.alloc(u16, n);
        defer a.free(words);
        for (words) |*v| v.* = c.toBf16((rng.float(f32) * 2 - 1) * (if (fill == .stress) @as(f32, 1.0e10) else 1));
        if (fill == .nonfinite) words[n / 3] = c.toBf16(std.math.nan(f32));
        return p.rig.input(std.mem.sliceAsBytes(words));
    }
    pub fn prepare(p: *Probe, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        p.rig.reset();
        p.rows = shape.rows;
        const d = shape.n;
        const w = shape.k;
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        p.x = try p.input(shape.rows * d, rng, fill);
        const gu = try p.input(2 * w * d, rng, fill);
        const down = try p.input(d * w, rng, fill);
        const tiled_gu = try p.rig.scratch(2 * w * d * 2);
        const tiled_down = try p.rig.scratch(d * w * 2);
        try glm.mm.tile(p.mm, p.rig.s, gu, tiled_gu, 2 * w, d);
        try glm.mm.tile(p.mm, p.rig.s, down, tiled_down, d, w);
        p.l = .{ .router = 0, .bias = 0, .routed = .{ .format = .nvfp4, .up = 0, .down = 0, .width = w, .dims = d, .experts = 288, .limit = 10 }, .count = 288, .shared_gu = .{ .bf16 = tiled_gu }, .shared_down = .{ .bf16 = tiled_down }, .shared = w, .top_k = 8, .scale = 1, .norm = true };
        const weights = try p.rig.gpu.gpa.alloc(f32, shape.rows * 9);
        defer p.rig.gpu.gpa.free(weights);
        for (weights) |*v| v.* = rng.float(f32);
        p.sc = .{ .mlog = 0, .router_part = 0, .pick = 0, .wts = try p.rig.input(std.mem.sliceAsBytes(weights)), .plan = undefined, .eact = 0, .ey = try p.rig.scratch(shape.rows * 9 * d * 2), .sgu = try p.rig.scratch(shape.rows * 2 * w * 2), .sact = try p.rig.scratch(shape.rows * w * 2), .sxs = try p.rig.scratch(shape.rows * (w / 64) * 4), .sy = try p.rig.scratch(shape.rows * d * 4), .mm_part = 0 };
        p.out = try p.rig.output(p.count());
        p.base = try p.rig.scratch(p.count());
        return true;
    }
    fn path(p: *Probe, fused: bool, dst: u64) !void {
        const slots: cuda.DeviceBuffer = .{ .d = p.rig.gpu.d, .ptr = p.sc.ey, .len = p.rows * 9 * p.l.routed.dims * 2 };
        try slots.fill8(0x3f, p.rig.s.handle); // same nonzero routed slots before each old/new run
        const output: cuda.DeviceBuffer = .{ .d = p.rig.gpu.d, .ptr = dst, .len = p.count() };
        try output.fill8(0xff, p.rig.s.handle);
        const k = glm.moe.Kernels{ .shared_once = fused, .set = .{ .glue = &p.glue, .mm = p.mm, .prompt = true }, .experts = undefined, .exl3 = undefined, .put_slot = p.put };
        try glm.moe.prepareShared(k, p.rig.s, p.l, p.sc, p.x, p.l.routed.dims, p.rows, true);
        var cuts: [glm.forward.max_pieces][2]usize = undefined;
        const n = glm.forward.ranges(p.rows, 3, &cuts);
        for (cuts[0..n], 0..) |cut, i| {
            if (fused and p.control and i == 0) continue;
            try glm.moe.fill(k, p.rig.s, p.l, p.sc, p.x, p.l.routed.dims, cut[0], cut[1], true, dst + cut[0] * p.l.routed.dims * 4);
        }
        try p.rig.s.synchronize();
    }
    pub fn run(p: *Probe) !void {
        try p.path(true, p.out);
    }
    pub fn replaced(p: *Probe) ![]const u8 {
        try p.path(false, p.base);
        return p.rig.bytes(p.base, p.count());
    }
    pub fn count(p: *Probe) usize {
        return p.rows * p.l.routed.dims * 4;
    }
    pub fn check(p: *Probe, _: g.Gate) !?[]const u8 {
        const got = try p.rig.gpu.gpa.dupe(u8, try p.rig.outputBytes());
        defer p.rig.gpu.gpa.free(got);
        const want = try p.replaced();
        return if (std.mem.eql(u8, got, want)) null else "shared expert or piece-combine bytes differ";
    }
    pub fn reference(_: *Probe, _: []f64, _: []f64) !void {}
};
pub const Suite = struct {
    mod: cuda.Module,
    mm: glm.mm.Kernels,
    probes: [2]Probe,
    pub fn create(gpu: check_.Gpu, s: cuda.Stream, glue: glm.glue.Kernels, ops: cuda.Module, kernels: *std.ArrayList(g.Kernel)) !*Suite {
        const a = gpu.gpa;
        const one = try a.create(Suite);
        errdefer a.destroy(one);
        one.mod = try cuda.Module.load(gpu.d, cuda.kernels.glm_mm);
        errdefer one.mod.unload();
        one.mm = try glm.mm.Kernels.load(one.mod);
        one.mm.prompt_fixed_k = false;
        const put = try ops.function("tf_glm_put_slot");
        for (&one.probes, 0..) |*p, i| p.* = .{ .rig = .{ .gpu = gpu, .s = s }, .mm = &one.mm, .glue = glue, .put = put, .control = i == 1 };
        for (&one.probes) |*p| try kernels.append(a, p.kernel());
        return one;
    }
    pub fn destroy(one: *Suite) void {
        const a = one.probes[0].rig.gpu.gpa;
        for (&one.probes) |*p| p.deinit();
        one.mod.unload();
        a.destroy(one);
    }
};
