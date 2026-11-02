//! Whole-head GDN chain, raw rows, state, gated rows and sums byte-equal to the original two launches.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const rig_ = @import("contract_rig.zig");
const c = core.contract.spec;
const g = core.contract.gates;
const chain = cuda.gdn_chain;
const Geo = extern struct { hid: c_int = 2048, nk: c_int, nv: c_int, qh: c_int = 16, kvh: c_int = 2, nl: c_int = 1 };
const eps: f32 = 1e-6;

pub const Control = enum { none, gate, state };
fn depth(_: c.Shape) usize {
    return 0;
}
pub const served = [_]c.Shape{
    .{ .rows = 1, .n = 16, .k = 32 },   .{ .rows = 17, .n = 16, .k = 48 },
    .{ .rows = 33, .n = 16, .k = 64 },  .{ .rows = 65, .n = 16, .k = 128 },
    .{ .rows = 512, .n = 16, .k = 32 }, .{ .rows = 2048, .n = 16, .k = 48 },
};
pub const Case = struct {
    rig: rig_.Rig,
    f: chain.Chain,
    post: cuda.Function,
    sums: bool = false,
    chunks: bool = false,
    control: Control = .none,
    shape: c.Shape = .{ .rows = 0, .n = 0, .k = 0 },
    ins: [8]u64 = @splat(0),
    output: u64 = 0,
    base: u64 = 0,
    seen: std.ArrayList(u8) = .empty,

    pub fn deinit(t: *Case) void {
        t.seen.deinit(t.rig.gpu.gpa);
        t.rig.deinit();
    }
    pub fn kernel(t: *Case) g.Kernel {
        return .{ .ctx = t, .vt = &g.hooks(Case), .control = if (t.control == .none) null else .shapes, .spec = .{
            .name = switch (t.control) {
                .gate => "control: GDN chain without SiLU gate",
                .state => "control: GDN chain rotating final state",
                .none => if (t.chunks) (if (t.sums) "cuda GDN chain norm chunk seams and sums" else "cuda GDN chain norm chunk seams") else if (t.sums) "cuda GDN chain norm with sums" else "cuda GDN chain norm",
            },
            .accum = .none,
            .depth = depth,
            .out = .raw,
            .nan = .{ .stated = "every raw row, gated row, state and optional group sum equals the old chain then post bytes" },
            .replaces = "chain_kernel<bf16,bf16,64,32> then tf_q35_gdn_post",
            .tiles = .{ .{ .size = 16 }, .{}, .{} },
            .served = &served,
        } };
    }
    fn input(t: *Case, i: usize, len: usize, floats: bool, fill: g.Fill, rng: std.Random) !void {
        const a = t.rig.gpu.gpa;
        const bytes = try a.alloc(u8, len * @as(usize, if (floats) 4 else 2));
        defer a.free(bytes);
        for (0..len) |j| {
            var v = (rng.float(f32) - 0.5) * (if (i == 5) @as(f32, 0.05) else 0.5);
            if (i == 3) v = 0.95 + rng.float(f32) * 0.04;
            if (i == 4) v = 0.25 + rng.float(f32) * 0.5;
            if (i == 7) v = 0.5 + rng.float(f32);
            if (fill == .nonfinite and j < 3) v = switch (j) {
                0 => std.math.nan(f32),
                1 => std.math.inf(f32),
                else => -std.math.inf(f32),
            };
            if (floats) std.mem.writeInt(u32, bytes[j * 4 ..][0..4], @bitCast(v), .little) else std.mem.writeInt(u16, bytes[j * 2 ..][0..2], c.toBf16(v), .little);
        }
        t.ins[i] = try t.rig.input(bytes);
    }
    pub fn prepare(t: *Case, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.n == 0 or shape.k == 0 or shape.k % shape.n != 0) return false;
        t.rig.reset();
        t.shape = shape;
        var prng = std.Random.DefaultPrng.init(seed);
        const r = prng.random();
        const rows = shape.rows;
        const hk = shape.n;
        const hv = shape.k;
        const lens = [_]usize{ rows * hk * 128, rows * hk * 128, rows * hv * 128, rows * hv, rows * hv, hv * 128 * 128, rows * hv * 128, 128 };
        for (lens, 0..) |len, i| try t.input(i, len, i == 3 or i == 5, fill, r);
        t.output = try t.rig.output(t.count());
        t.base = try t.rig.scratch(t.count());
        return true;
    }
    pub fn count(t: *Case) usize {
        const s = t.shape;
        return 2 * s.rows * s.k * 128 * 2 + s.k * 128 * 128 * 4 + (if (t.sums) s.rows * s.k * 2 * 4 else 0);
    }
    fn buffers(t: *Case, ptr: u64) chain.Buffers {
        const width = t.shape.rows * t.shape.k * 128 * 2;
        const state = ptr + 2 * width;
        return .{ .q = t.ins[0], .k = t.ins[1], .v = t.ins[2], .g = t.ins[3], .beta = t.ins[4], .state = state, .last = state, .y = ptr, .z = t.ins[6], .norm = t.ins[7], .mix = ptr + width, .sums = if (t.sums) state + t.shape.k * 128 * 128 * 4 else 0 };
    }
    fn resetState(t: *Case, b: chain.Buffers) !void {
        const len = t.shape.k * 128 * 128 * 4;
        const state: cuda.DeviceBuffer = .{ .d = t.rig.gpu.d, .ptr = b.state, .len = len };
        try state.copyFrom(0, t.ins[5], len, t.rig.s.handle);
    }
    pub fn run(t: *Case) !void {
        var b = t.buffers(t.output);
        try t.resetState(b);
        var at: usize = 0;
        while (at < t.shape.rows) {
            const n = @min(t.shape.rows - at, if (t.chunks) @as(usize, 17) else t.shape.rows);
            if (!try t.f.run(t.rig.s, b, n, t.shape.n, t.shape.k, eps)) return error.FusedKernelNotBound;
            at += n;
            b.q += n * t.shape.n * 128 * 2;
            b.k += n * t.shape.n * 128 * 2;
            b.v += n * t.shape.k * 128 * 2;
            b.g += n * t.shape.k * 4;
            b.beta += n * t.shape.k * 2;
            b.y += n * t.shape.k * 128 * 2;
            b.z += n * t.shape.k * 128 * 2;
            b.mix += n * t.shape.k * 128 * 2;
            if (b.sums != 0) b.sums += n * t.shape.k * 2 * 4;
        }
        try t.rig.s.synchronize();
    }
    pub fn replaced(t: *Case) ![]const u8 {
        const b = t.buffers(t.base);
        try t.resetState(b);
        var old = t.f;
        old.fused = null;
        _ = try old.run(t.rig.s, b, t.shape.rows, t.shape.n, t.shape.k, eps);
        var args: cuda.Args = .{};
        args.add(Geo{ .nk = @intCast(t.shape.n), .nv = @intCast(t.shape.k) });
        for ([_]u64{ b.y, b.z, b.norm }) |v| args.add(v);
        args.add(eps);
        args.add(b.mix);
        args.add(@as(u64, 0));
        args.add(b.sums);
        try cuda.launch.launch(t.post, .{ .grid = .{ .y = @intCast(t.shape.k), .z = @intCast(t.shape.rows) }, .block = .{ .x = 32 } }, t.rig.s, &args);
        return t.rig.bytes(t.base, t.count());
    }
    pub fn check(t: *Case, _: g.Gate) !?[]const u8 {
        t.seen.clearRetainingCapacity();
        try t.seen.appendSlice(t.rig.gpu.gpa, try t.rig.outputBytes());
        return if (std.mem.eql(u8, t.seen.items, try t.replaced())) null else "GDN bytes differ from the original chain and post";
    }
    pub fn reference(_: *Case, _: []f64, _: []f64) !void {
        return error.ReplacementOnly;
    }
};

pub const Suite = struct {
    gdn: cuda.Module,
    qwen: cuda.Module,
    cases: [6]Case,
    pub fn init(gpu: @import("check.zig").Gpu, s: cuda.Stream, controls: cuda.Module) !Suite {
        var gd = try cuda.Module.load(gpu.d, cuda.kernels.gdn_prefill);
        errdefer gd.unload();
        var qw = try cuda.Module.load(gpu.d, cuda.kernels.qwen35);
        errdefer qw.unload();
        const f = try chain.Chain.resolveMode(gd, true);
        const post = try qw.function("tf_q35_gdn_post");
        var cases: [6]Case = undefined;
        for (&cases, 0..) |*case, i| {
            var fns = f;
            if (i >= 4) fns.fused = try controls.function(if (i == 4) "tf_gdn_chain_norm_bad_gate" else "tf_gdn_chain_norm_bad_state");
            case.* = .{ .rig = .{ .gpu = gpu, .s = s }, .f = fns, .post = post, .sums = i % 2 == 1 or i >= 4, .chunks = i == 2 or i == 3, .control = if (i == 4) .gate else if (i == 5) .state else .none };
        }
        return .{ .gdn = gd, .qwen = qw, .cases = cases };
    }
    pub fn append(t: *Suite, a: std.mem.Allocator, list: *std.ArrayList(g.Kernel)) !void {
        for (&t.cases) |*case| try list.append(a, case.kernel());
    }
    pub fn deinit(t: *Suite) void {
        for (&t.cases) |*case| case.deinit();
        t.qwen.unload();
        t.gdn.unload();
    }
};
