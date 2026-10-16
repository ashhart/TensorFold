//! Nemotron's Mamba-2 conv, dt and group norm under the contract on HIP, in the chip's activation kind.
const std = @import("std");
const hip = @import("hip");
const core = @import("core");
const rig_ = @import("contract_rig.zig");
const kit = @import("contract_kit.zig");

const contract = core.contract;
const c = contract.spec;
const g = contract.gates;
const rows = contract.rows;

pub const Which = enum { conv, dt, gnorm };

const dh = 64; // a head's value rows
const ds = 128; // a value row's states

/// A rank's Mamba-2 columns for `heads` (Nano's 64 at tp 1, 32 at tp 2, 16 at tp 4): eight heads a group.
pub const Dims = struct {
    heads: usize,

    pub fn groups(d: Dims) usize {
        return @max(d.heads / 8, 1);
    }
    pub fn xd(d: Dims) usize {
        return d.heads * dh;
    }
    pub fn cd(d: Dims) usize {
        return d.xd() + 2 * d.groups() * ds;
    }
    pub fn proj(d: Dims) usize {
        return d.xd() + d.cd() + d.heads;
    }
};

/// Four fmas from the bias.
fn convDepth(_: c.Shape) usize {
    return 4;
}

fn none(_: c.Shape) usize {
    return 0;
}

/// Eight squares a lane by fma, the wave's xor tree, then the four waves' partials in order.
fn gnormDepth(_: c.Shape) usize {
    return 8 + 5 + 3;
}

const conv_why = "five terms through silu, which squashes a sum's error below zero: the fp16 margin proves fp32 sums";
const gnorm_why = "a 16-bit output follows the sum of squares as ss^-1/2: a bf16 sum's error is a few output ulps";

pub fn spec(which: Which, k: kit.Kind) c.Spec {
    const out = kit.elem(k);
    return switch (which) {
        // rows x heads; the conv slides 16 rows a thread, so rows have edges at 16
        .conv => .{
            .name = if (k == .bf16) "hip mamba tf_mamba_conv (bf16)" else "hip mamba tf_mamba_conv (fp16)",
            .accum = .f32,
            .depth = convDepth,
            // silu's exp, add and divide
            .ops = 6,
            // the sum rounded to the kind before silu, whose relative slope |1 + r (1 - sigmoid r)| <= 2.4 here
            .mids = 3,
            .out = out,
            .tiles = .{ .{ .size = 16 }, .{}, .{} },
            .served = &.{
                .{ .rows = 1, .n = 64, .k = 5 },
                .{ .rows = 3, .n = 32, .k = 5 },
                .{ .rows = 2048, .n = 16, .k = 5 },
            },
            .sums = .{ .bf16 = 0, .why = conv_why },
        },
        // softplus's exp and log1p, and the bias add's rounding, which softplus amplifies by at most |v| <= 8 here
        .dt => .{
            .name = if (k == .bf16) "hip mamba tf_mamba_dt (bf16 in)" else "hip mamba tf_mamba_dt (fp16 in)",
            .accum = .none,
            .depth = none,
            .ops = 16,
            .out = .f32,
            .tiles = .{ .{ .size = 4 }, .{}, .{} },
            .served = &.{
                .{ .rows = 1, .n = 64, .k = 0 },
                .{ .rows = 3, .n = 32, .k = 0 },
                .{ .rows = 2048, .n = 16, .k = 0 },
            },
        },
        // rows x groups of k; a block a (row, group), element i on lane i % 128
        .gnorm => .{
            .name = if (k == .bf16) "hip mamba tf_mamba_gnorm (bf16)" else "hip mamba tf_mamba_gnorm (fp16)",
            .accum = .f32,
            .depth = gnormDepth,
            // the mean, eps, sqrt, divide, the product and the weight
            .ops = 8,
            // v / rms rounded to the kind before the weight
            .mids = 1,
            .out = out,
            .tiles = .{ .{}, .{}, .{ .size = 128 } },
            .served = &.{
                .{ .rows = 1, .n = 8, .k = 512 },
                .{ .rows = 64, .n = 4, .k = 512 },
                .{ .rows = 3, .n = 2, .k = 1024 },
            },
            .sums = .{ .bf16 = 0, .why = gnorm_why },
        },
    };
}

pub const Mamba = struct {
    rig: rig_.Rig,
    which: Which,
    l: *const hip.Launcher,
    k: kit.Kind,
    arena: std.heap.ArenaAllocator,
    shape: c.Shape = undefined,
    d: Dims = undefined,
    /// The conv's and dt's float64 inputs: P [rows][proj], the conv tail [3][cd], conv w [cd][4], b [cd], dt bias.
    p: []f64 = &.{},
    tail: []f64 = &.{},
    cw: []f64 = &.{},
    cb: []f64 = &.{},
    bias: []f64 = &.{},
    /// The group norm's rows over (row, group) pairs, the weight repeated for each group.
    r: ?rows.Rows = null,
    in: [6]u64 = @splat(0),
    out: [2]u64 = @splat(0),

    pub fn init(a: std.mem.Allocator, rt: *const hip.Runtime, s: hip.Stream, l: *const hip.Launcher, w: Which) Mamba {
        return .{ .rig = .{ .a = a, .r = rt, .s = s }, .which = w, .l = l, .k = kit.act(), .arena = .init(a) };
    }

    pub fn deinit(self: *Mamba) void {
        self.arena.deinit();
        self.rig.deinit();
    }

    pub fn kernel(self: *Mamba) g.Kernel {
        return .{ .spec = spec(self.which, self.k), .ctx = self, .vt = &vt };
    }

    const vt = g.hooks(Mamba);

    pub fn prepare(self: *Mamba, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.n == 0) return false;
        if (self.which == .gnorm and (shape.k == 0 or shape.k > 1024)) return false;
        self.rig.reset();
        _ = self.arena.reset(.retain_capacity);
        self.r = null;
        self.shape = shape;
        self.d = .{ .heads = shape.n };
        const a = self.arena.allocator();
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        switch (self.which) {
            .gnorm => return self.prepareNorm(fill, seed),
            .conv => try self.prepareConv(rng, fill),
            .dt => {
                self.p = try a.alloc(f64, shape.rows * self.d.proj());
                @memset(self.p, 0);
                const at = self.d.xd() + self.d.cd();
                for (0..shape.rows) |row| for (0..shape.n) |h| {
                    self.p[row * self.d.proj() + at + h] = kit.round(self.k, rng.float(f64) * 8 - 7);
                };
                self.bias = try kit.floats(a, rng, shape.n, -0.5, 0.5);
                const odd = [_]f64{ std.math.nan(f64), std.math.inf(f64), -std.math.inf(f64) };
                if (fill == .nonfinite) for (odd, 0..) |v, i| {
                    if (i < shape.rows) self.p[i * self.d.proj() + at + i % shape.n] = v;
                };
            },
        }
        const pb = try a.alloc(u8, self.p.len * kit.size(self.k));
        for (self.p, 0..) |v, i| kit.put(self.k, pb, i, v);
        self.in[0] = try self.rig.input(pb);
        if (self.which == .dt) {
            self.in[1] = try self.rig.input(try kit.f32Bytes(a, self.bias));
            self.out[0] = try self.rig.output(shape.rows * shape.n * 4);
            return true;
        }
        const tb = try a.alloc(u8, self.tail.len * kit.size(self.k));
        for (self.tail, 0..) |v, i| kit.put(self.k, tb, i, v);
        self.in[1] = try self.rig.input(tb);
        self.in[2] = try self.rig.input(try kit.f32Bytes(a, self.cw));
        self.in[3] = try self.rig.input(try kit.f32Bytes(a, self.cb));
        self.out[0] = try self.rig.output(shape.rows * self.d.cd() * kit.size(self.k));
        self.out[1] = try self.rig.output(3 * self.d.cd() * kit.size(self.k));
        return true;
    }

    /// Taps in [-1, 1) and weights in [-0.25, 0.25) around a bias of 1.5 to 2.5 either side: |sum| in [0.5, 3.5].
    fn prepareConv(self: *Mamba, rng: std.Random, fill: g.Fill) !void {
        const a = self.arena.allocator();
        const d = self.d;
        const n = self.shape.rows;
        self.p = try a.alloc(f64, n * d.proj());
        @memset(self.p, 0);
        self.tail = try a.alloc(f64, 3 * d.cd());
        self.cw = try a.alloc(f64, 4 * d.cd());
        self.cb = try a.alloc(f64, d.cd());
        const stress = fill == .stress;
        for (0..n + 3) |t| for (0..d.cd()) |ch| {
            // gate 4: taps repeat every two rows, so weights a, b, -a, -b cancel to the bias exactly
            const even: f64 = @floatFromInt(40 + (ch + t % 2) % 24);
            const v = if (stress) even else kit.round(self.k, rng.float(f64) * 2 - 1);
            if (t < 3) self.tail[t * d.cd() + ch] = v else self.p[(t - 3) * d.proj() + d.xd() + ch] = v;
        };
        for (self.cb, 0..) |*b, ch| {
            const sign: f64 = if (rng.boolean()) 1 else -1;
            const small: f64 = @floatFromInt(1 + ch % 3);
            b.* = if (stress) sign * small else @as(f32, @floatCast(sign * (1.5 + rng.float(f64))));
            const w = self.cw[4 * ch ..][0..4];
            if (stress) {
                w[0] = @floatFromInt(2000 + ch % 40);
                w[1] = @floatFromInt(1990 + ch % 50);
                w[2] = -w[0];
                w[3] = -w[1];
            } else for (w) |*x| {
                x.* = @as(f32, @floatCast(rng.float(f64) * 0.5 - 0.25));
            }
        }
        if (fill != .nonfinite) return;
        const at = [_][2]usize{ .{ 0, 0 }, .{ 0, 5 }, .{ 1, 9 }, .{ 2, 3 } };
        const put = [_]f64{ std.math.nan(f64), std.math.inf(f64), -std.math.inf(f64), std.math.inf(f64) };
        for (at, put) |p, v| self.tail[p[0] * d.cd() + p[1]] = v;
        if (n > 1) self.p[d.proj() + d.xd() + 7] = std.math.nan(f64);
    }

    /// (row, group) pairs as rows of k; one weight row repeated for every group.
    fn prepareNorm(self: *Mamba, fill: g.Fill, seed: u64) !bool {
        const a = self.arena.allocator();
        const s = self.shape;
        var r = try rows.Rows.init(a, .rms, .{ .rows = s.rows * s.n, .n = s.k, .k = s.k });
        r.fill(fill, seed);
        r.eps = 1e-5;
        if (self.k == .f16) for (r.x) |*b| {
            if (@abs(c.fromBf16(b.*)) < 0x1p-14) b.* = 0;
        };
        self.r = r;
        const xb = try a.alloc(u8, r.x.len * 2);
        for (r.x, 0..) |b, i| kit.put(self.k, xb, i, c.fromBf16(b));
        const wb = try a.alloc(u8, s.n * s.k * 4);
        for (0..s.n * s.k) |i| kit.put(.f32, wb, i, c.fromBf16(r.w[i % s.k]));
        self.in[0] = try self.rig.input(xb);
        self.in[1] = try self.rig.input(wb);
        self.out[0] = try self.rig.output(r.x.len * 2);
        return true;
    }

    pub fn run(self: *Mamba) !void {
        const l = self.l;
        const d = self.d;
        const n: c_int = @intCast(self.shape.rows);
        const kind = @backingInt(self.k);
        const s = self.rig.s.handle;
        const C = ?*const anyopaque;
        const P = ?*anyopaque;
        const CF = ?[*]const f32;
        switch (self.which) {
            .conv => try l.tf_mamba_conv(
                kit.ptr(C, self.in[0]),
                kind,
                kit.ptr(C, self.in[1]),
                kit.ptr(P, self.out[0]),
                kit.ptr(CF, self.in[2]),
                kit.ptr(CF, self.in[3]),
                kit.ptr(P, self.out[1]),
                n,
                @intCast(d.proj()),
                @intCast(d.xd()),
                @intCast(d.cd()),
                s,
            ),
            .dt => try l.tf_mamba_dt(
                kit.ptr(C, self.in[0]),
                kind,
                kit.ptr(CF, self.in[1]),
                kit.ptr(?[*]f32, self.out[0]),
                n,
                @intCast(d.proj()),
                @intCast(d.xd() + d.cd()),
                @intCast(d.heads),
                0,
                std.math.inf(f32),
                s,
            ),
            .gnorm => {
                const k: c_int = @intCast(self.shape.k);
                try l.tf_mamba_gnorm(
                    kit.ptr(C, self.in[0]),
                    kind,
                    kit.ptr(CF, self.in[1]),
                    kit.ptr(P, self.out[0]),
                    n,
                    k * @as(c_int, @intCast(self.shape.n)),
                    k,
                    self.r.?.eps,
                    s,
                );
            },
        }
        try self.rig.s.synchronize();
    }

    pub fn count(self: *Mamba) usize {
        return switch (self.which) {
            .conv => (self.shape.rows + 3) * self.d.cd(),
            .dt => self.shape.rows * self.shape.n,
            .gnorm => self.r.?.outputs(),
        };
    }

    /// The input tap `t` of channel `ch`: the conv tail's rows, then P's x columns.
    fn tap(self: *Mamba, t: usize, ch: usize) f64 {
        const d = self.d;
        return if (t < 3) self.tail[t * d.cd() + ch] else self.p[(t - 3) * d.proj() + d.xd() + ch];
    }

    pub fn reference(self: *Mamba, want: []f64, mag: []f64) !void {
        const d = self.d;
        const n = self.shape.rows;
        switch (self.which) {
            .gnorm => return self.r.?.reference(want, mag),
            .dt => for (0..n) |row| for (0..d.heads) |h| {
                const v = self.p[row * d.proj() + d.xd() + d.cd() + h] + self.bias[h];
                const sp = if (std.math.isNan(v)) v else @max(v, 0) + std.math.log1p(@exp(-@abs(v)));
                want[row * d.heads + h] = if (std.math.isNan(sp)) sp else @min(@max(sp, 0), std.math.inf(f64));
                mag[row * d.heads + h] = 0;
            },
            .conv => for (0..d.cd()) |ch| {
                for (0..n) |row| {
                    var acc = self.cb[ch];
                    var abs = @abs(acc);
                    for (0..4) |j| {
                        const term = self.cw[4 * ch + j] * self.tap(row + j, ch);
                        acc += term;
                        abs += @abs(term);
                    }
                    want[row * d.cd() + ch] = kit.silu(acc);
                    mag[row * d.cd() + ch] = abs * kit.siluSlope(acc);
                }
                for (0..3) |j| {
                    want[(n + j) * d.cd() + ch] = self.tap(n + j, ch);
                    mag[(n + j) * d.cd() + ch] = 0;
                }
            },
        }
    }

    pub fn simulate(self: *Mamba, acc: c.Accum, order: g.Order, out: []f64) !void {
        if (self.which == .gnorm) return self.r.?.simulate(acc, order, out);
        const d = self.d;
        const n = self.shape.rows;
        for (0..d.cd()) |ch| {
            for (0..n) |row| {
                var t: [5]f64 = undefined;
                t[0] = self.cb[ch];
                for (0..4) |j| t[j + 1] = self.cw[4 * ch + j] * self.tap(row + j, ch);
                out[row * d.cd() + ch] = kit.silu(rows.sumIn(acc, order, &t));
            }
            for (0..3) |j| out[(n + j) * d.cd() + ch] = self.tap(n + j, ch);
        }
    }
};
