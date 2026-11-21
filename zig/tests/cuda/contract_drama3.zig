//! Drama 3 under the CUDA contract: RMS norm rows (plain gamma and gated form), compact router and CFG mix.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const check = @import("check.zig");
const rig_ = @import("contract_rig.zig");
const router_k = @import("contract_drama3_router.zig");

const contract = core.contract;
const c = contract.spec;
const g = contract.gates;
const rows = contract.rows;

/// ceil(k / 256) squares a thread, then a warp's tree and the block's tree of 8 warps.
fn depth(s: c.Shape) usize {
    return (s.k + 255) / 256 + 10;
}

const why = "a bf16 output follows the sum of squares as ss^-1/2: a bf16 sum's error is a few output ulps";

pub fn spec() c.Spec {
    return .{ .name = "cuda drama3 d3_rmsnorm (bf16 rows, plain gamma)", .accum = .f32, .depth = depth, .ops = 8, .out = .bf16, .tiles = .{ .{}, .{}, .{ .size = 256 } }, .served = &.{ .{ .rows = 1, .n = 2048, .k = 2048 }, .{ .rows = 168, .n = 2048, .k = 2048 } }, .sums = .{ .bf16 = 0, .why = why } };
}

pub const Norm = struct {
    rig: rig_.Rig,
    f: cuda.Function,
    r: ?rows.Rows = null,
    x: u64 = 0,
    w: u64 = 0,
    out: u64 = 0,

    pub fn init(gpu: check.Gpu, s: cuda.Stream, mod: cuda.Module) !Norm {
        return .{ .rig = .{ .gpu = gpu, .s = s }, .f = try mod.function("d3_rmsnorm") };
    }

    pub fn deinit(self: *Norm) void {
        if (self.r) |r| r.deinit();
        self.rig.deinit();
    }

    pub fn kernel(self: *Norm) g.Kernel {
        return .{ .spec = spec(), .ctx = self, .vt = &vt };
    }

    const vt = g.hooks(Norm);

    pub fn prepare(self: *Norm, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.n != shape.k) return false;
        if (self.r) |r| r.deinit();
        self.r = null;
        self.rig.reset();
        const r = try rows.Rows.init(self.rig.gpu.gpa, .rms, shape);
        self.r = r;
        r.fill(fill, seed);
        self.x = try self.rig.input(std.mem.sliceAsBytes(r.x));
        self.w = try self.rig.input(std.mem.sliceAsBytes(r.w));
        self.out = try self.rig.output(r.outputs() * c.size(spec().out));
        return true;
    }

    pub fn run(self: *Norm) !void {
        const r = self.r.?;
        const k: c_int = @intCast(r.shape.k);
        var a: cuda.Args = .{};
        a.add(self.x);
        a.add(k);
        a.add(self.w);
        a.add(self.out);
        a.add(k);
        a.add(k);
        a.add(r.eps);
        a.add(@as(c_int, 0));
        try cuda.launch.launch(self.f, .{ .grid = .{ .x = @intCast(r.shape.rows), .y = 1, .z = 1 }, .block = .{ .x = 256 } }, self.rig.s, &a);
        try self.rig.s.synchronize();
    }

    pub fn count(self: *Norm) usize {
        return self.r.?.outputs();
    }

    pub fn reference(self: *Norm, want: []f64, mag: []f64) !void {
        return self.r.?.reference(want, mag);
    }

    pub fn simulate(self: *Norm, acc: c.Accum, order: g.Order, out: []f64) !void {
        return self.r.?.simulate(acc, order, out);
    }
};

fn noSums(_: c.Shape) usize {
    return 0;
}

const mix_gamma: f32 = 2.5;

pub fn mixSpec() c.Spec {
    // Drama 3's candidates: 4,096 semantic tokens and EOS
    return .{ .name = "cuda drama3 d3_cfg_mix (semantic CFG, gamma 2.5)", .accum = .none, .depth = noSums, .out = .f32, .tiles = .{ .{}, .{ .size = 256 }, .{} }, .served = &.{.{ .rows = 1, .n = 4097, .k = 1 }} };
}

/// d3_cfg_mix updates the conditional logits in place: each run starts from a fresh copy of them.
pub const Mix = struct {
    rig: rig_.Rig,
    f: cuda.Function,
    cond: []f32 = &.{},
    uncond: []f32 = &.{},
    u: u64 = 0,
    out: u64 = 0,

    pub fn init(gpu: check.Gpu, s: cuda.Stream, mod: cuda.Module) !Mix {
        return .{ .rig = .{ .gpu = gpu, .s = s }, .f = try mod.function("d3_cfg_mix") };
    }

    pub fn deinit(self: *Mix) void {
        self.free();
        self.rig.deinit();
    }

    fn free(self: *Mix) void {
        self.rig.gpu.gpa.free(self.cond);
        self.rig.gpu.gpa.free(self.uncond);
        self.cond = &.{};
        self.uncond = &.{};
    }

    pub fn kernel(self: *Mix) g.Kernel {
        return .{ .spec = mixSpec(), .ctx = self, .vt = &mix_vt };
    }

    const mix_vt = g.hooks(Mix);

    pub fn prepare(self: *Mix, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        self.free();
        self.rig.reset();
        const a = self.rig.gpu.gpa;
        self.cond = try a.alloc(f32, shape.n);
        self.uncond = try a.alloc(f32, shape.n);
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        for (self.cond, self.uncond) |*x, *y| {
            x.* = (rng.float(f32) - 0.5) * 64;
            y.* = (rng.float(f32) - 0.5) * 64;
        }
        if (fill == .nonfinite) {
            const planted = [_]f32{ std.math.nan(f32), std.math.inf(f32), -std.math.inf(f32) };
            for (planted, 0..) |v, i| if (i < shape.n) {
                (if (i == 1) self.uncond else self.cond)[i] = v;
            };
        }
        // the inputs as the gates see them; each run copies the conditional logits into the output it mixes in place
        _ = try self.rig.input(std.mem.sliceAsBytes(self.cond));
        self.u = try self.rig.input(std.mem.sliceAsBytes(self.uncond));
        self.out = try self.rig.output(shape.n * 4);
        return true;
    }

    pub fn run(self: *Mix) !void {
        const n = self.cond.len;
        const out: cuda.DeviceBuffer = .{ .d = self.rig.gpu.d, .ptr = self.out, .len = n * 4 };
        try out.upload(0, std.mem.sliceAsBytes(self.cond));
        var a: cuda.Args = .{};
        a.add(self.out);
        a.add(self.u);
        a.add(@as(c_int, @intCast(n)));
        a.add(mix_gamma);
        try cuda.launch.launch(self.f, .{ .grid = .{ .x = @intCast((n + 255) / 256), .y = 1, .z = 1 }, .block = .{ .x = 256 } }, self.rig.s, &a);
        try self.rig.s.synchronize();
    }

    pub fn count(self: *Mix) usize {
        return self.cond.len;
    }

    pub fn reference(self: *Mix, want: []f64, mag: []f64) !void {
        const gm = @as(f64, mix_gamma) - 1;
        for (self.cond, self.uncond, want, mag) |x, y, *w, *m| {
            w.* = @as(f64, x) + gm * (@as(f64, x) - @as(f64, y));
            m.* = @abs(@as(f64, x)) + gm * (@abs(@as(f64, x)) + @abs(@as(f64, y)));
        }
    }

    /// Three fp32 roundings, each at most 2^-24 of the terms they touch, so the bound follows `mag`, not the result.
    pub fn bounds(_: *Mix, _: []const f64, mag: []const f64, allow: []f64) !void {
        for (allow, mag) |*l, m| l.* = 0x1p-22 * m + 0x1p-149;
    }
};

/// The drama3 image's registrations onto `kernels`: norm rows, compact router with unmasked control, CFG mix.
pub const Group = struct {
    a: std.mem.Allocator,
    mod: cuda.Module,
    norm: Norm,
    routers: [2]router_k.Router,
    mix: Mix,

    pub fn create(a: std.mem.Allocator, gpu: check.Gpu, s: cuda.Stream, kernels: *std.ArrayList(g.Kernel)) !*Group {
        const self = try a.create(Group);
        errdefer a.destroy(self);
        self.a = a;
        self.mod = try cuda.Module.load(gpu.d, cuda.kernels.drama3);
        errdefer self.mod.unload();
        self.norm = try Norm.init(gpu, s, self.mod);
        self.mix = try Mix.init(gpu, s, self.mod);
        self.routers = .{ try .init(gpu, s, self.mod, false), try .init(gpu, s, self.mod, true) };
        try kernels.append(a, self.norm.kernel());
        for (&self.routers) |*r| try kernels.append(a, r.kernel());
        try kernels.append(a, self.mix.kernel());
        return self;
    }

    pub fn destroy(self: *Group) void {
        self.norm.deinit();
        for (&self.routers) |*r| r.deinit();
        self.mix.deinit();
        self.mod.unload();
        self.a.destroy(self);
    }
};
