//! Nemotron's routers under the contract on CUDA: fp32 logits in K slices, then each row's top-k by its stated rule.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const nemotron = @import("nemotron");
const check = @import("check.zig");
const rig_ = @import("contract_rig.zig");

const contract = core.contract;
const c = contract.spec;
const g = contract.gates;
const Dense = contract.dense.Dense;
const glue = nemotron.glue;

/// The K slices the engine cuts a router into: Super's latent router (4,096 inputs) its own, Nano's the glue's.
fn slices(k: usize) usize {
    return if (k == 4096) nemotron.latent.routerSlices(k) else glue.routerShape(k).sk;
}

/// One fma chain a K slice (two a bf16 word pair).
fn sliceChain(s: c.Shape) usize {
    return s.k / slices(s.k) + 2;
}

pub fn logitsSpec() c.Spec {
    return .{ .name = "cuda nemotron router logits tf_nemo_router", .accum = .f32, .depth = sliceChain, .out = .f32, .tiles = .{ .{ .size = 16 }, .{ .size = 16 }, .{} }, .served = &.{ .{ .rows = 16, .n = 128, .k = 2688 }, .{ .rows = 33, .n = 128, .k = 2688 } } };
}

/// Nano's router and Super's (512 experts) past 32 rows.
pub fn tilesSpec() c.Spec {
    return .{ .name = "cuda nemotron router logits tf_nemo_router_tiles", .accum = .f32, .depth = sliceChain, .out = .f32, .replaces = "tf_nemo_router", .tiles = .{ .{ .size = 128 }, .{ .size = 128, .step = 16, .min = 16 }, .{} }, .served = &.{ .{ .rows = 300, .n = 128, .k = 2688 }, .{ .rows = 129, .n = 512, .k = 4096 }, .{ .rows = 4100, .n = 512, .k = 4096 } } };
}

/// x [rows, D] bf16 times the router's W [E, D] bf16: fp32 partials [slices][rows][E]; `tiles` runs in place of `f`.
pub const Logits = struct {
    rig: rig_.Rig,
    f: cuda.Function,
    tiles: ?cuda.Function = null,
    d: ?Dense = null,
    x: u64 = 0,
    w: u64 = 0,
    out: u64 = 0,
    base: u64 = 0,

    pub fn deinit(self: *Logits) void {
        if (self.d) |d| d.deinit();
        self.rig.deinit();
    }

    pub fn kernel(self: *Logits) g.Kernel {
        return .{ .spec = if (self.tiles != null) tilesSpec() else logitsSpec(), .ctx = self, .vt = &vt };
    }

    const vt = g.hooks(Logits);

    pub fn prepare(self: *Logits, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.k % 64 != 0 or (self.tiles == null and shape.n > 256) or shape.n == 0 or shape.rows == 0) return false;
        if (self.d) |d| d.deinit();
        self.d = null;
        self.rig.reset();
        const a = self.rig.gpu.gpa;
        const d = try Dense.init(a, shape, slices(shape.k));
        self.d = d;
        d.fillX(fill, seed, 1);
        const bits = try a.alloc(u16, d.w.len);
        defer a.free(bits);
        var prng = std.Random.DefaultPrng.init(seed ^ 0x77);
        const rng = prng.random();
        const k = shape.k;
        for (bits) |*b| b.* = c.toBf16(if (fill == .stress) 0 else rng.float(f32) * 2 - 1);
        if (fill == .stress) for (0..shape.n) |j| for (0..k / 2) |i| {
            const v: f32 = @floatFromInt(1 + rng.uintLessThan(u32, 7));
            bits[j * k + i] = c.toBf16(v);
            bits[j * k + k / 2 + i] = c.toBf16(-v);
        };
        for (d.w, bits) |*w, b| w.* = c.fromBf16(b);
        self.x = try self.rig.input(std.mem.sliceAsBytes(d.x));
        self.w = try self.rig.input(std.mem.sliceAsBytes(bits));
        self.out = try self.rig.output(d.outputs() * 4);
        self.base = try self.rig.scratch(d.outputs() * 4);
        return true;
    }

    pub fn run(self: *Logits) !void {
        try self.launch(self.tiles != null, self.out);
        try self.rig.s.synchronize();
    }

    /// tf_nemo_router's 16-row tiles, or tf_nemo_router_tiles' 128, on the prepared rows.
    fn launch(self: *Logits, tiles: bool, out: u64) !void {
        const d = self.d.?;
        const sk = slices(d.shape.k);
        var a: cuda.Args = .{};
        for ([_]u64{ self.x, self.w, out }) |p| a.add(p);
        for ([_]usize{ d.shape.rows, d.shape.k, d.shape.n, sk }) |v| a.add(@as(c_int, @intCast(v)));
        const t: usize = if (tiles) 128 else 16;
        const grid: cuda.Dim3 = .{ .x = @intCast((d.shape.rows + t - 1) / t), .y = @intCast((d.shape.n + t - 1) / t), .z = @intCast(sk) };
        const f = if (tiles) self.tiles.? else self.f;
        try cuda.launch.launch(f, .{ .grid = grid, .block = .{ .x = 256 }, .shared = if (tiles) glue.tiles_shared else glue.routerShared(d.shape.k / sk) }, self.rig.s, &a);
    }

    pub fn replaced(self: *Logits) ![]const u8 {
        try self.launch(false, self.base);
        return self.rig.bytes(self.base, self.count() * 4);
    }

    pub fn count(self: *Logits) usize {
        return self.d.?.outputs();
    }

    pub fn reference(self: *Logits, want: []f64, mag: []f64) !void {
        return self.d.?.reference(want, mag);
    }

    pub fn simulate(self: *Logits, acc: c.Accum, order: g.Order, out: []f64) !void {
        return self.d.?.simulate(acc, order, out);
    }
};

fn noSums(_: c.Shape) usize {
    return 0;
}

const scaling: f32 = 2.5;

const latent_rule = "tf_super_topk's bytes, and its host twin's picks: a NaN never ahead of a number, ties to the lower " ++
    "expert, every pick past the numbers expert 0";
const latent_picks = 22;
const latent_shares = 2;

/// Super's 512 experts summed from 8 slices at verify windows' calls; experts at whole warps and either side.
pub fn latentSpec() c.Spec {
    const served = &[_]c.Shape{ .{ .rows = 1, .n = 512, .k = 8 }, .{ .rows = 16, .n = 512, .k = 8 }, .{ .rows = 32, .n = 512, .k = 8 } };
    return .{ .name = "cuda nemotron latent top-k tf_super_topk_few", .accum = .none, .depth = noSums, .out = .raw, .replaces = "tf_super_topk", .nan = .{ .stated = latent_rule }, .tiles = .{ .{}, .{ .size = 32, .min = latent_picks }, .{} }, .served = served };
}

/// The latent router's picks, a thread an expert: pair ids by pick and share, weights by pick; shape as TopK's.
pub const LatentTopK = struct {
    rig: rig_.Rig,
    fns: *const nemotron.latent.Fns,
    shape: c.Shape = undefined,
    part: []f32 = &.{},
    bias: []f32 = &.{},
    in: u64 = 0,
    b: u64 = 0,
    ids: u64 = 0,
    wts: u64 = 0,
    base: u64 = 0,
    msg: [160]u8 = undefined,

    pub fn deinit(self: *LatentTopK) void {
        self.rig.gpu.gpa.free(self.part);
        self.rig.gpu.gpa.free(self.bias);
        self.rig.deinit();
    }

    pub fn kernel(self: *LatentTopK) g.Kernel {
        return .{ .spec = latentSpec(), .ctx = self, .vt = &vt };
    }

    const vt = g.hooks(LatentTopK);

    fn idBytes(self: *const LatentTopK) usize {
        return self.shape.rows * latent_picks * latent_shares * 4;
    }

    fn outBytes(self: *const LatentTopK) usize {
        return self.idBytes() + self.shape.rows * latent_picks * 4;
    }

    /// Sixteenths of logits (ties), biases in 64ths; non-finite: NaN and infinities in rows 0-3, numbers in 12 experts.
    pub fn prepare(self: *LatentTopK, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.n < latent_picks or shape.n > nemotron.latent.max_experts or shape.k == 0 or shape.rows == 0) return false;
        self.rig.reset();
        const a = self.rig.gpu.gpa;
        self.shape = shape;
        a.free(self.part);
        self.part = &.{};
        a.free(self.bias);
        self.bias = &.{};
        self.part = try a.alloc(f32, shape.k * shape.rows * shape.n);
        self.bias = try a.alloc(f32, shape.n);
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        for (self.part) |*p| p.* = @as(f32, @floatFromInt(rng.uintLessThan(u32, 64))) / 16 - 2;
        for (self.bias) |*p| p.* = @as(f32, @floatFromInt(rng.uintLessThan(u32, 8))) / 64;
        if (fill == .nonfinite) {
            const n = shape.n;
            const r = shape.rows;
            const put = [_]f32{ std.math.nan(f32), std.math.inf(f32), -std.math.inf(f32), std.math.nan(f32) };
            for (put, 0..) |v, row| if (row < r) for (0..4) |e| {
                self.part[row * n + 3 * e + row] = v;
            };
            @memset(self.bias[12..], -std.math.inf(f32));
        }
        self.in = try self.rig.input(std.mem.sliceAsBytes(self.part));
        self.b = try self.rig.input(std.mem.sliceAsBytes(self.bias));
        self.ids = try self.rig.output(self.idBytes());
        self.wts = try self.rig.output(self.outBytes() - self.idBytes());
        self.base = try self.rig.scratch(self.outBytes());
        return true;
    }

    pub fn run(self: *LatentTopK) !void {
        try self.launch(true, self.ids, self.wts);
        try self.rig.s.synchronize();
    }

    fn launch(self: *LatentTopK, few: bool, ids: u64, wts: u64) !void {
        const x = self.shape;
        try self.fns.picks(self.rig.s, few, self.in, self.b, ids, wts, x.rows, scaling, x.n, x.k, latent_picks, latent_shares, true);
    }

    /// tf_super_topk on the same inputs, ids then weights as the outputs gather.
    pub fn replaced(self: *LatentTopK) ![]const u8 {
        try self.launch(false, self.base, self.base + self.idBytes());
        return self.rig.bytes(self.base, self.outBytes());
    }

    pub fn count(self: *LatentTopK) usize {
        return self.outBytes();
    }

    pub fn reference(_: *LatentTopK, want: []f64, mag: []f64) !void {
        @memset(want, 0);
        @memset(mag, 0);
    }

    /// tf_super_topk's bytes, then the host twin's ids and weights (a NaN weight for a NaN).
    pub fn check(self: *LatentTopK, _: g.Gate) !?[]const u8 {
        const a = self.rig.gpu.gpa;
        const x = self.shape;
        const got = try a.dupe(u8, try self.rig.outputBytes());
        defer a.free(got);
        if (std.mem.indexOfDiff(u8, got, try self.replaced())) |at| return try std.fmt.bufPrint(&self.msg, "byte {d} of {d} differs from tf_super_topk's", .{ at, got.len });
        const idx = try a.alloc(i32, x.rows * latent_picks * latent_shares);
        defer a.free(idx);
        const wt = try a.alloc(f32, x.rows * latent_picks);
        defer a.free(wt);
        nemotron.glue_ref.superTopk(self.part, self.bias, idx, wt, x.rows, scaling, x.n, x.k, latent_picks, latent_shares, true);
        if (std.mem.indexOfDiff(u8, got[0..self.idBytes()], std.mem.sliceAsBytes(idx))) |at| return try std.fmt.bufPrint(&self.msg, "id byte {d} differs from the host twin's", .{at});
        for (wt, 0..) |w, i| {
            const v: f32 = @bitCast(std.mem.readInt(u32, got[self.idBytes() + 4 * i ..][0..4], .little));
            if (v != w and !(std.math.isNan(v) and std.math.isNan(w))) return try std.fmt.bufPrint(&self.msg, "weight {d} is {e}, the host twin's {e}", .{ i, v, w });
        }
        return null;
    }
};
