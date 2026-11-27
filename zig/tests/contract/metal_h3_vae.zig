//! MiniMax-H3 video VAE's fp32 Metal kernels under the contract: each against its formula in float64.
const std = @import("std");
const mtl = @import("metal");
const contract = @import("contract");
const h3k = @import("h3_kernels");
const rig_ = @import("rig.zig");
const bolt = @import("metal_bolt.zig");

const c = contract.spec;
const g = contract.gates;

pub const Which = enum { matmul, swiglu, rmsnorm, layernorm, postquant };

/// The decode ViT's dimensions at their extremes: dim 2,048, ffn 8,192, 24 latent channels.
pub const dim = 2048;
pub const ffn = 8192;
pub const chans = 24;
/// Rows of a real piece at its cap and at its smallest edge tile.
pub const most_rows = 108 * 60 + 65;
pub const few_rows = 108 * 2 + 5;

fn kChain(s: c.Shape) usize {
    return s.k + 1;
}

/// A thread's strided terms, then the threadgroup's 8-level tree.
fn tgChain(s: c.Shape) usize {
    return (s.k + 255) / 256 + 8;
}

/// The mean pass, the variance pass, then rsqrt: each pass a strided sum and its tree.
fn lnChain(s: c.Shape) usize {
    // f32's mean error lives in every subtracted term: double the chain's room.
    return 8 * (tgChain(s) + 2 + tgChain(s)) + 24;
}

fn none(_: c.Shape) usize {
    return 0;
}

fn qChain(_: c.Shape) usize {
    return 24 + 1;
}

pub fn spec(w: Which) c.Spec {
    return switch (w) {
        .matmul => .{ .name = "h3 vae matmul h3_vae_matmul", .accum = .f32, .depth = kChain, .out = .f32, .served = &.{ .{ .rows = most_rows, .n = dim, .k = dim }, .{ .rows = few_rows, .n = 3 * dim, .k = dim }, .{ .rows = most_rows, .n = ffn, .k = dim } } },
        .swiglu => .{ .name = "h3 vae SwiGLU h3_vae_swiglu", .accum = .none, .depth = none, .ops = 16, .mids = 4, .out = .f32, .served = &.{.{ .rows = most_rows, .n = ffn, .k = 1 }} },
        .rmsnorm => .{ .name = "h3 vae rms norm h3_vae_rmsnorm", .accum = .f32, .depth = tgChain, .ops = 3, .out = .f32, .served = &.{ .{ .rows = most_rows, .n = dim, .k = dim }, .{ .rows = few_rows, .n = dim, .k = dim } } },
        .layernorm => .{ .name = "h3 vae layer norm h3_vae_layernorm", .accum = .f32, .depth = lnChain, .ops = 3, .out = .f32, .sums = .{ .f16 = 0, .bf16 = 0, .exact = false, .why = "the mean and the variance pass through fp32 subtracts: fp32 sums stay near the value, fp16 does not" }, .served = &.{ .{ .rows = most_rows, .n = dim, .k = dim }, .{ .rows = few_rows, .n = dim, .k = dim } } },
        .postquant => .{ .name = "h3 vae post quant h3_vae_postquant", .accum = .f32, .depth = qChain, .out = .f32, .served = &.{ .{ .rows = most_rows, .n = chans, .k = chans }, .{ .rows = few_rows, .n = chans, .k = chans } } },
    };
}

/// The kernel `w` launches: its function name in h3_kernels' sources (vae.metal).
fn function(w: Which) []const u8 {
    return switch (w) {
        .matmul => "h3_vae_matmul",
        .swiglu => "h3_vae_swiglu",
        .rmsnorm => "h3_vae_rmsnorm",
        .layernorm => "h3_vae_layernorm",
        .postquant => "h3_vae_postquant",
    };
}

/// A planted control: the source edit that breaks one gate, proven red.
fn control(w: Which) ?bolt.Mut {
    return switch (w) {
        .matmul => .{ .name = "control: h3 vae matmul h3_vae_matmul, its last K step dropped", .find = "for (uint k = 0; k < K; k++) sum += x[k] * w[k];", .with = "for (uint k = 0; k + 1 < K; k++) sum += x[k] * w[k];", .gate = .shapes },
        .swiglu => .{ .name = "control: h3 vae SwiGLU h3_vae_swiglu, its gate and value swapped", .find = "const float gate = FUSED[r * 2 * F + f];", .with = "const float gate = FUSED[r * 2 * F + F + f];", .gate = .shapes },
        .rmsnorm => .{ .name = "control: h3 vae rms norm h3_vae_rmsnorm, its tree sums in half", .find = "if (tid < stride) red[tid] += red[tid + stride];", .with = "if (tid < stride) red[tid] += half(red[tid + stride]);", .gate = .sums },
        .layernorm => .{ .name = "control: h3 vae layer norm h3_vae_layernorm, its mean dropped", .find = "const float mean = red[0] / float(D);", .with = "const float mean = 0.0f * red[0] / float(D);", .gate = .shapes },
        .postquant => .{ .name = "control: h3 vae post quant h3_vae_postquant, its bias dropped", .find = "OUT[v * C + o] = sum + B[o];", .with = "OUT[v * C + o] = sum;", .gate = .shapes },
    };
}
/// The inputs a VAE kernel reads: its shape and its tables.
pub const In = struct {
    rows: usize,
    n: usize,
    k: usize,
    a: []const f64 = &.{},
    b: []const f64 = &.{},
    d: []const f64 = &.{},
    e: []const f64 = &.{},
};

pub const Vae = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    which: Which,
    pipe: mtl.Pipeline,
    mut: ?bolt.Mut = null,
    shape: c.Shape = undefined,
    arena: std.heap.ArenaAllocator,
    in: In = .{ .rows = 0, .n = 0, .k = 0 },
    bufs: [8]rig_.Ref = undefined,
    nbufs: usize = 0,
    outs: usize = 0,

    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, which: Which, mut: ?bolt.Mut) !Vae {
        const rig: rig_.Rig = .{ .a = a, .device = device, .queue = queue };
        const src = if (mut) |m| try bolt.mutated(a, h3k.source, m) else h3k.source;
        defer if (mut != null) a.free(src);
        return .{ .a = a, .rig = rig, .which = which, .pipe = try rig_.pipeline(device, src, function(which)), .mut = mut, .arena = std.heap.ArenaAllocator.init(a) };
    }

    pub fn deinit(self: *Vae) void {
        self.rig.deinit();
        self.pipe.deinit();
        self.arena.deinit();
    }

    pub fn kernel(self: *Vae) g.Kernel {
        return .{ .spec = bolt.under(spec(self.which), self.mut), .ctx = self, .vt = &vt, .control = if (self.mut) |m| m.gate else null };
    }

    const vt = g.hooks(Vae);

    /// `n` fp32 values, uniform in [lo, hi), stress makes them small integers, plants put non-finites at 3 spots.
    pub fn values(self: *Vae, n: usize, lo: f64, hi: f64, fill: g.Fill, rng: std.Random, plant: bool) ![]f64 {
        const v = try self.arena.allocator().alloc(f64, n);
        for (v, 0..) |*x, i| {
            const r = lo + (hi - lo) * rng.float(f64);
            x.* = switch (fill) {
                .stress => @floatFromInt(@as(i64, @intFromFloat(@round(r * 8)))),
                else => r,
            };
            if (i % 61 == 7 and fill != .stress) x.* *= 6;
            x.* = @as(f32, @floatCast(x.*));
        }
        if (fill == .nonfinite and plant) {
            const at = [_]usize{ n / 3, n / 2, n - 1 };
            const put = [_]f64{ std.math.nan(f64), std.math.inf(f64), -std.math.inf(f64) };
            for (at, put) |p, x| v[p] = x;
        }
        return v;
    }

    /// Upload `v` as fp32, as an input.
    pub fn upload(self: *Vae, v: []const f64) !void {
        const al = self.arena.allocator();
        const f = try al.alloc(f32, v.len);
        for (f, v) |*o, x| o.* = @floatCast(x);
        self.bufs[self.nbufs] = try self.rig.input(std.mem.sliceAsBytes(f));
        self.nbufs += 1;
    }

    pub fn ints(self: *Vae, v: []const i32) !void {
        self.bufs[self.nbufs] = try self.rig.input(std.mem.sliceAsBytes(v));
        self.nbufs += 1;
    }

    pub fn output(self: *Vae, bytes: usize) !void {
        self.bufs[self.nbufs] = try self.rig.output(bytes);
        self.nbufs += 1;
    }

    /// Zero-filled f64 values, uploaded as fp32: an exact additive bias.
    fn zeros(self: *Vae, n: usize) ![]f64 {
        const v = try self.arena.allocator().alloc(f64, n);
        @memset(v, 0);
        return v;
    }

    pub fn prepare(self: *Vae, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        self.rig.reset();
        _ = self.arena.reset(.retain_capacity);
        self.nbufs = 0;
        self.outs = 0;
        self.shape = shape;
        self.in = .{ .rows = shape.rows, .n = shape.n, .k = shape.k };
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        const s = shape;
        switch (self.which) {
            .matmul => {
                if (s.rows == 0 or s.n == 0 or s.k == 0) return false;
                const lo: f64 = if (fill == .stress) -4 else -1;
                const x = try self.values(s.rows * s.k, lo, -lo, fill, rng, true);
                const w = try self.values(s.n * s.k, lo, -lo, fill, rng, false);
                const b = if (fill == .stress) try self.zeros(s.n) else try self.values(s.n, -0.5, 0.5, .random, rng, false);
                self.in.a = x;
                self.in.b = w;
                self.in.d = b;
                try self.upload(x);
                try self.upload(w);
                try self.upload(b);
                try self.output(s.rows * s.n * 4);
                self.outs = s.rows * s.n;
            },
            .swiglu => {
                if (s.rows == 0 or s.n == 0) return false;
                const fused = try self.values(s.rows * 2 * s.n, -3, 3, fill, rng, true);
                self.in.a = fused;
                try self.upload(fused);
                try self.output(s.rows * s.n * 4);
                self.outs = s.rows * s.n;
            },
            .rmsnorm => {
                if (s.rows == 0 or s.k != s.n) return false;
                // gate 4: same-sign small values, so the sum of squares stays exact in fp32
                const x = try self.values(s.rows * s.n, 0, 2, fill, rng, true);
                const w = try self.values(s.n, 0.5, 1.5, .random, rng, false);
                self.in.a = x;
                self.in.d = w;
                try self.upload(x);
                try self.upload(w);
                try self.output(s.rows * s.n * 4);
                self.outs = s.rows * s.n;
            },
            .layernorm => {
                if (s.rows == 0 or s.k != s.n) return false;
                const x = try self.values(s.rows * s.n, 0, 2, fill, rng, true);
                const w = try self.values(s.n, 0.5, 1.5, fill, rng, false);
                const b = try self.values(s.n, -0.5, 0.5, fill, rng, false);
                self.in.a = x;
                self.in.d = w;
                self.in.e = b;
                try self.upload(x);
                try self.upload(w);
                try self.upload(b);
                try self.output(s.rows * s.n * 4);
                self.outs = s.rows * s.n;
            },
            .postquant => {
                if (s.rows == 0 or s.k != chans or s.n != chans) return false;
                const lo: f64 = if (fill == .stress) -4 else -1;
                const x = try self.values(s.rows * s.k, lo, -lo, fill, rng, true);
                const w = try self.values(s.n * s.k, lo, -lo, fill, rng, false);
                const b = if (fill == .stress) try self.zeros(s.n) else try self.values(s.n, -0.5, 0.5, .random, rng, false);
                self.in.a = x;
                self.in.b = w;
                self.in.d = b;
                try self.upload(x);
                try self.upload(w);
                try self.upload(b);
                try self.output(s.rows * s.n * 4);
                self.outs = s.rows * s.n;
            },
        }
        return true;
    }

    pub fn run(self: *Vae) !void {
        const s = self.shape;
        const p = self.rig.begin();
        const e = p.e;
        e.setPipeline(self.pipe);
        switch (self.which) {
            .matmul => {
                for (0..4) |i| self.bind(e, i, i);
                e.setValue([4]u32{ @intCast(s.rows), @intCast(s.k), @intCast(s.n), 0 }, 4);
                e.dispatchThreads(mtl.Size.of(@intCast(s.n), @intCast(s.rows), 1), mtl.Size.of(64, 1, 1));
            },
            .swiglu => {
                self.bind(e, 0, 0);
                self.bind(e, 1, 1);
                e.setValue([1]u32{@intCast(s.n)}, 2);
                e.dispatchThreads(mtl.Size.of(@intCast(s.n), @intCast(s.rows), 1), mtl.Size.of(64, 1, 1));
            },
            .rmsnorm => {
                self.bind(e, 0, 0);
                self.bind(e, 1, 1);
                self.bind(e, 2, 2);
                e.setValue([1]u32{@intCast(s.n)}, 3);
                e.dispatchGroups(mtl.Size.of(@intCast(s.rows), 1, 1), mtl.Size.of(256, 1, 1));
            },
            .layernorm => {
                for (0..4) |i| self.bind(e, i, i);
                e.setValue([1]u32{@intCast(s.n)}, 4);
                e.dispatchGroups(mtl.Size.of(@intCast(s.rows), 1, 1), mtl.Size.of(256, 1, 1));
            },
            .postquant => {
                for (0..4) |i| self.bind(e, i, i);
                e.setValue([2]u32{ @intCast(s.rows), @intCast(s.n) }, 4);
                e.dispatchThreads(mtl.Size.of(@intCast(s.n * s.rows), 1, 1), mtl.Size.of(256, 1, 1));
            },
        }
        try self.rig.finish(p);
    }

    pub fn bind(self: *Vae, e: mtl.ComputeEncoder, i: usize, at: usize) void {
        e.setBuffer(self.bufs[i].buf, self.bufs[i].off, @intCast(at));
    }

    pub fn count(self: *Vae) usize {
        return self.outs;
    }

    /// The kernel's formula in float64: gate 1, 2 and 3's yardstick.
    pub fn reference(self: *Vae, want: []f64, mag: []f64) !void {
        const s = self.shape;
        switch (self.which) {
            .matmul => {
                for (0..s.rows) |r| for (0..s.n) |n| {
                    var sum: f64 = 0;
                    var m: f64 = 0;
                    for (0..s.k) |k| {
                        const t = self.in.a[r * s.k + k] * self.in.b[n * s.k + k];
                        sum += t;
                        m += @abs(t);
                    }
                    want[r * s.n + n] = sum + self.in.d[n];
                    mag[r * s.n + n] = m + @abs(self.in.d[n]);
                };
            },
            .swiglu => {
                for (0..s.rows) |r| for (0..s.n) |f| {
                    const gate = self.in.a[r * 2 * s.n + f];
                    const val = self.in.a[r * 2 * s.n + s.n + f];
                    want[r * s.n + f] = val * gate / (1 + @exp(-gate));
                    mag[r * s.n + f] = @abs(val * gate) + 0.5 * @abs(val);
                };
            },
            .rmsnorm => {
                var sq: f64 = 0;
                _ = &sq;
                for (0..s.rows) |r| {
                    var m: f64 = 0;
                    for (0..s.n) |i| {
                        const t = self.in.a[r * s.n + i];
                        sq = t * t;
                        m += sq;
                    }
                    const inv = 1 / @sqrt(m / @as(f64, @floatFromInt(s.n)) + 1e-5);
                    for (0..s.n) |i| {
                        const x = self.in.a[r * s.n + i];
                        want[r * s.n + i] = x * inv * self.in.d[i];
                        mag[r * s.n + i] = @abs(x * inv * self.in.d[i]);
                    }
                }
            },
            .layernorm => {
                for (0..s.rows) |r| {
                    var mean: f64 = 0;
                    for (0..s.n) |i| mean += self.in.a[r * s.n + i];
                    mean /= @as(f64, @floatFromInt(s.n));
                    var m2: f64 = 0;
                    for (0..s.n) |i| {
                        const t = self.in.a[r * s.n + i] - mean;
                        m2 += t * t;
                    }
                    const rstd = 1 / @sqrt(m2 / @as(f64, @floatFromInt(s.n)) + 1e-5);
                    for (0..s.n) |i| {
                        const x = self.in.a[r * s.n + i];
                        want[r * s.n + i] = (x - mean) * rstd * self.in.d[i] + self.in.e[i];
                        mag[r * s.n + i] = (@abs(x) + @abs(mean)) * rstd * @abs(self.in.d[i]) + @abs(self.in.e[i]);
                    }
                }
            },
            .postquant => {
                for (0..s.rows) |v| for (0..s.n) |o| {
                    var sum: f64 = 0;
                    var m: f64 = 0;
                    for (0..s.k) |i| {
                        const t = self.in.a[v * s.k + i] * self.in.b[o * s.k + i];
                        sum += t;
                        m += @abs(t);
                    }
                    want[v * s.n + o] = sum + self.in.d[o];
                    mag[v * s.n + o] = m + @abs(self.in.d[o]);
                };
            },
        }
    }

    /// Gate 4's simulation: the same sums in the accumulator and order the kernel uses.
    pub fn simulate(self: *Vae, acc: c.Accum, order: g.Order, out: []f64) !void {
        const s = self.shape;
        switch (self.which) {
            .matmul, .postquant => {
                for (0..s.rows * s.n) |i| out[i] = 0;
                for (0..s.rows) |r| for (0..s.n) |n| {
                    const t = self.arena.allocator().alloc(f64, s.k) catch return error.OutOfMemory;
                    defer self.arena.allocator().free(t);
                    for (t, 0..) |*v, k| v.* = self.in.a[r * s.k + k] * self.in.b[n * s.k + k];
                    out[r * s.n + n] = @import("contract").rows.sumIn(acc, order, t) + self.in.d[n];
                };
            },
            .rmsnorm => {
                for (0..s.rows * s.n) |i| out[i] = 0;
                for (0..s.rows) |r| {
                    const full = self.arena.allocator().alloc(f64, s.n) catch return error.OutOfMemory;
                    defer self.arena.allocator().free(full);
                    for (full, 0..) |*v, i| v.* = self.in.a[r * s.n + i] * self.in.a[r * s.n + i];
                    const ss = @import("contract").rows.sumIn(acc, order, full);
                    const inv = 1 / @sqrt(ss / @as(f64, @floatFromInt(s.n)) + 1e-5);
                    for (0..s.n) |i| {
                        const x = self.in.a[r * s.n + i];
                        const w = self.in.d[i];
                        out[r * s.n + i] = x * inv * w;
                    }
                }
            },
            .layernorm => {
                for (0..s.rows * s.n) |i| out[i] = 0;
                for (0..s.rows) |r| {
                    const row = self.in.a[r * s.n .. r * s.n + s.n];
                    const mean = @import("contract").rows.sumIn(acc, order, row) / @as(f64, @floatFromInt(s.n));
                    const cent = self.arena.allocator().alloc(f64, s.n) catch return error.OutOfMemory;
                    defer self.arena.allocator().free(cent);
                    for (cent, 0..) |*v, i| v.* = (row[i] - mean) * (row[i] - mean);
                    const m2sum = @import("contract").rows.sumIn(acc, order, cent);
                    const inv = 1 / @sqrt(m2sum / @as(f64, @floatFromInt(s.n)) + 1e-5);
                    for (0..s.n) |i| {
                        const x = self.in.a[r * s.n + i];
                        out[r * s.n + i] = (x - mean) * inv * self.in.d[i] + self.in.e[i];
                    }
                }
            },
            .swiglu => {
                for (0..s.rows * s.n) |i| out[i] = self.in.a[i];
            },
        }
    }
};

/// Every H3 video VAE kernel: metal_main.zig's registration.
pub const Suite = struct {
    a: std.mem.Allocator,
    device: mtl.Device,
    queue: mtl.Queue,
    kernels: *std.ArrayList(g.Kernel),
    list: std.ArrayList(*Vae) = .empty,

    pub fn collect(s: *Suite, only: ?[]const u8) !void {
        for (std.enums.values(Which)) |w| {
            if (only) |o| if (std.mem.indexOf(u8, spec(w).name, o) == null) continue;
            try s.add(w, null);
            if (control(w)) |m| try s.add(w, m);
        }
    }

    fn add(s: *Suite, w: Which, mut: ?bolt.Mut) !void {
        const x = try s.a.create(Vae);
        errdefer s.a.destroy(x);
        x.* = try Vae.init(s.a, s.device, s.queue, w, mut);
        errdefer x.deinit();
        try @import("contract").register(s.a, &s.list, &s.kernels, x, x.kernel());
    }

    pub fn deinit(s: *Suite) void {
        for (s.list.items) |x| {
            x.deinit();
            s.a.destroy(x);
        }
        s.list.deinit(s.a);
    }
};
