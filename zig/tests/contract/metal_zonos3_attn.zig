//! ZONOS3's attention kernels on Metal under the contract (contract_zonos3_attn.zig's twin); their chains are CUDA's.
const std = @import("std");
const mtl = @import("metal");
const contract = @import("contract");
const ref = @import("metal_zonos3_ref.zig");
const pick = @import("metal_zonos3_pick.zig");

const c = contract.spec;
const g = contract.gates;
const Iv = ref.Iv;
const Sums = ref.Sums;

pub const Which = enum { backbone, codec, teacher };

/// A thread's chain over every 256th (backbone) or 128th key, then the threads' partials in order.
fn backboneDepth(s: c.Shape) usize {
    return (s.k + 255) / 256 + 256;
}

fn codecDepth(s: c.Shape) usize {
    return (s.k + 127) / 128 + 128;
}

fn teacherDepth(s: c.Shape) usize {
    return (s.rows + 127) / 128 + 128;
}

/// Backbone: rows x 512 x the last row's keys. Codec: rows x a stage's width x its context. Teacher: k is 64.
pub fn attentionSpec(which: Which, fault: bool) c.Spec {
    return switch (which) {
        .backbone => .{ .name = if (fault) "control: metal zonos3 z3_attention seeing one key fewer" else "metal zonos3 z3_attention", .accum = .f32, .depth = backboneDepth, .out = .f32, .served = &.{ .{ .rows = 64, .n = 512, .k = 256 }, .{ .rows = 1, .n = 512, .k = 4096 }, .{ .rows = 1, .n = 512, .k = 1 }, .{ .rows = 7, .n = 512, .k = 300 } } },
        .codec => .{ .name = if (fault) "control: metal zonos3 zc_attention seeing one key fewer" else "metal zonos3 zc_attention", .accum = .f32, .depth = codecDepth, .out = .f32, .served = &.{ .{ .rows = 16, .n = 768, .k = 300 }, .{ .rows = 8, .n = 1280, .k = 150 }, .{ .rows = 1, .n = 768, .k = 480 } } },
        .teacher => .{ .name = if (fault) "control: metal zonos3 zt_attention seeing one key fewer" else "metal zonos3 zt_attention", .accum = .f32, .depth = teacherDepth, .out = .f32, .served = &.{ .{ .rows = 64, .n = 1024, .k = 64 }, .{ .rows = 1, .n = 1024, .k = 64 }, .{ .rows = 200, .n = 1024, .k = 64 } } },
    };
}

const distances = 73;

/// Softmax attention of a (row, head): fma-chain scores, then values and weights summed by thread, then across.
pub const Attention = struct {
    pipe: mtl.Pipeline,
    which: Which,
    /// control: the device sees one key fewer than the reference
    fault: bool = false,
    shape: c.Shape = undefined,
    pos0: usize = 0,
    /// codec: the caches' rows, a ring over positions (position j is row j % cap), as the served stages hold them
    cap: usize = 0,
    /// backbone: q [T][512]; codec and teacher: qkv [T][3 n]
    q: []f32 = &.{},
    /// the key and value caches (backbone [L][128], codec [L][n]); the teacher's keys and values are qkv's
    kc: []f32 = &.{},
    vc: []f32 = &.{},
    /// backbone: a gate a (row, head); teacher: the distance table [73][64]
    extra: []f32 = &.{},
    dev: [4]u64 = @splat(0),
    out: u64 = 0,

    pub const sums = {};

    pub fn spec(self: *const Attention) c.Spec {
        return attentionSpec(self.which, self.fault);
    }

    pub fn control(self: *const Attention) ?g.Gate {
        return if (self.fault) .shapes else null;
    }

    pub fn free(self: *Attention, a: std.mem.Allocator) void {
        for ([_]*[]f32{ &self.q, &self.kc, &self.vc, &self.extra }) |x| {
            a.free(x.*);
            x.* = &.{};
        }
    }

    fn headDim(self: *const Attention) usize {
        return if (self.which == .backbone) 32 else 64;
    }

    /// The width of a cache row, and of the values a key head holds.
    fn cacheWidth(self: *const Attention) usize {
        return if (self.which == .backbone) 128 else self.shape.n;
    }

    /// Rows of keys the launch can reach.
    fn keys(self: *const Attention) usize {
        return if (self.which == .teacher) self.shape.rows else self.pos0 + self.shape.rows;
    }

    pub fn prepare(self: *Attention, a: std.mem.Allocator, rig: *ref.Rig, shape: c.Shape, fill: g.Fill, rng: std.Random) !bool {
        const T = shape.rows;
        if (T == 0 or shape.k == 0) return false;
        switch (self.which) {
            .backbone => if (shape.n != 512 or shape.k < T) return false,
            .codec => if (shape.n % 64 != 0) return false,
            .teacher => if (shape.n != 1024 or shape.k != 64) return false,
        }
        self.free(a);
        self.shape = shape;
        const n = shape.n;
        self.pos0 = switch (self.which) {
            .backbone => shape.k - T,
            // gate 4: the first row sees the first `ctx` keys; else the last row's window is full or sliding
            .codec => if (fill == .stress) shape.k - 1 else (shape.k -| T) + rng.uintLessThan(usize, shape.k),
            .teacher => 0,
        };
        const L = self.keys();
        const cw = self.cacheWidth();
        const qw: usize = if (self.which == .backbone) n else 3 * n;
        self.q = try ref.values(a, rng, T * qw, .random, 1);
        if (self.which != .teacher) {
            self.kc = try ref.values(a, rng, L * cw, .random, 1);
            self.vc = try ref.values(a, rng, L * cw, .random, 2);
        }
        if (self.which == .backbone) self.extra = try ref.values(a, rng, T * 16, .random, 3);
        if (self.which == .teacher) self.extra = try ref.values(a, rng, distances * 64, .random, 1);
        const vals = if (self.which == .teacher) self.q else self.vc;
        const vstride: usize = if (self.which == .teacher) 3 * n else cw;
        const vfirst: usize = if (self.which == .teacher) 2 * n else 0;
        if (fill == .stress) {
            // zero queries: every score is 0 and every weight 1; the values carry gate 4's terms along the keys
            if (self.which == .backbone) @memset(self.q, 0) else for (0..T) |t| @memset(self.q[t * 3 * n ..][0..n], 0);
            for (0..cw) |i| ref.stressAxis(vals, vfirst + i, vstride, L, rng);
            if (self.which == .backbone) @memset(self.extra, 100);
        }
        if (fill == .nonfinite) {
            const hd = self.headDim();
            self.q[5] = ref.nan; // row 0, head 0
            const keys_ = if (self.which == .teacher) self.q else self.kc;
            const kfirst: usize = if (self.which == .teacher) n else 0;
            if (T > 2) keys_[(self.pos0 + 2) * vstride + kfirst + hd + 3] = ref.inf; // key head 1, from row 2 on
            if (T > 4) vals[(self.pos0 + 4) * vstride + vfirst + 2 * hd + 1] = -ref.inf; // key head 2, from row 4 on
        }
        self.dev[0] = try ref.input(rig, self.q);
        if (self.which == .codec) {
            // the window and a pass's rows, the served ring's size: every position a row reads has a row of its own
            self.cap = shape.k + T;
            for ([_][]const f32{ self.kc, self.vc }, 1..) |rows, i| {
                const ring = try a.alloc(f32, self.cap * cw);
                defer a.free(ring);
                @memset(ring, 0);
                for (0..L) |j| @memcpy(ring[(j % self.cap) * cw ..][0..cw], rows[j * cw ..][0..cw]);
                self.dev[i] = try ref.input(rig, ring);
            }
        } else {
            if (self.kc.len > 0) self.dev[1] = try ref.input(rig, self.kc);
            if (self.vc.len > 0) self.dev[2] = try ref.input(rig, self.vc);
        }
        if (self.extra.len > 0) self.dev[3] = try ref.input(rig, self.extra);
        self.out = try rig.output(T * n * 4);
        return true;
    }

    pub fn run(self: *Attention, rig: *ref.Rig) !void {
        const T = self.shape.rows;
        const n = self.shape.n;
        const less = ref.one(self.fault);
        switch (self.which) {
            .backbone => try rig.go(self.pipe, .{ T, 16, 1 }, 256, &.{ self.dev[0], self.dev[1], self.dev[2], self.dev[3], self.out }, extern struct { pos0: i32 }{ .pos0 = ref.int(self.pos0 -| less) }, 0),
            .codec => {
                const A = extern struct { D: i32, pos0: i32, ctx: i32, cap: i32 };
                try rig.go(self.pipe, .{ T, n / 64, 1 }, 128, &.{ self.dev[0], self.dev[1], self.dev[2], self.out }, A{ .D = ref.int(n), .pos0 = ref.int(self.pos0), .ctx = ref.int(self.shape.k - less), .cap = ref.int(self.cap) }, 0);
            },
            .teacher => {
                const A = extern struct { D: i32, T: i32 };
                try rig.go(self.pipe, .{ T, n / 64, 1 }, 128, &.{ self.dev[0], self.dev[3], self.out }, A{ .D = ref.int(n), .T = ref.int(T - less) }, 0);
            },
        }
    }

    pub fn count(self: *const Attention) usize {
        return self.shape.rows * self.shape.n;
    }

    pub fn solve(self: *const Attention, a: std.mem.Allocator, s: Sums, out: []Iv) !void {
        const T = self.shape.rows;
        const n = self.shape.n;
        const hd = self.headDim();
        const L = self.keys();
        const depth = self.spec().depth(self.shape);
        const e = try a.alloc(Iv, 2 * L);
        defer a.free(e);
        const scratch = try a.alloc(f64, @max(L, 64));
        defer a.free(scratch);
        const terms = e[L..];
        var dot: [64]Iv = undefined;
        var pe: [distances]Iv = undefined;
        for (0..T * n / hd) |at| {
            const t = at / (n / hd);
            const h = at % (n / hd);
            const kh = if (self.which == .backbone) h >> 2 else h;
            const p = self.pos0 + t;
            const lo = if (self.which == .codec) (p + 1) -| self.shape.k else 0;
            const hi = if (self.which == .teacher) T else p + 1;
            const q = switch (self.which) {
                .backbone => self.q[t * n + h * hd ..][0..hd],
                else => self.q[t * 3 * n + h * hd ..][0..hd],
            };
            if (self.which == .teacher and !s.exact) for (&pe, 0..) |*v, d| {
                for (dot[0..hd], q, self.extra[d * 64 ..][0..64]) |*term, x, y| term.* = Iv.of(x).mul(Iv.of(y));
                v.* = s.sum(dot[0..hd], hd, scratch);
            };
            const w = e[0 .. hi - lo];
            if (!s.exact) for (w, lo..) |*score, j| {
                const k = switch (self.which) {
                    .teacher => self.q[j * 3 * n + n + kh * hd ..][0..hd],
                    else => self.kc[j * self.cacheWidth() + kh * hd ..][0..hd],
                };
                for (dot[0..hd], q, k) |*term, x, y| term.* = Iv.of(x).mul(Iv.of(y));
                const d = s.sum(dot[0..hd], hd, scratch);
                score.* = switch (self.which) {
                    .backbone => d.mul(Iv.of(@as(f32, 0.17677669529663688))),
                    .codec => d.mul(Iv.of(0.125)),
                    .teacher => blk: {
                        const rel = @as(i64, @intCast(j)) - @as(i64, @intCast(t));
                        break :blk d.add(pe[@intCast(std.math.clamp(rel, -64, 8) + 64)]).mul(Iv.of(0.125));
                    },
                };
            };
            const den = ref.softmaxParts(s, w, depth, scratch);
            const gate = if (self.which == .backbone) Iv.of(self.extra[t * 16 + h]).sigmoid() else Iv.of(1);
            for (0..hd) |i| {
                for (terms[0 .. hi - lo], w, lo..) |*term, weight, j| {
                    const v = switch (self.which) {
                        .teacher => self.q[j * 3 * n + 2 * n + kh * hd + i],
                        else => self.vc[j * self.cacheWidth() + kh * hd + i],
                    };
                    term.* = weight.mul(Iv.of(v));
                }
                const r = s.sum(terms[0 .. hi - lo], depth, scratch).div(den);
                out[t * n + h * hd + i] = if (self.which == .backbone) r.mul(gate) else r;
            }
        }
    }
};

const theta: f32 = 10000;

/// Rows x the queries' 512 x a head's 32.
pub fn prepSpec(fault: bool) c.Spec {
    return .{ .name = if (fault) "control: metal zonos3 z3_attn_prep rotating its rows one position late" else "metal zonos3 z3_attn_prep (q in place)", .accum = .f32, .depth = warpDeep, .out = .f32, .in_place = true, .served = &.{ .{ .rows = 64, .n = 512, .k = 32 }, .{ .rows = 1, .n = 512, .k = 32 }, .{ .rows = 7, .n = 512, .k = 32 } } };
}

fn warpDeep(_: c.Shape) usize {
    return 5;
}

/// The backbone's q and k rms-normed per head of 32 (q times |temp|) and rotated; k and v land in their cache rows.
pub const Prep = struct {
    pipe: mtl.Pipeline,
    /// control: the device takes the rows one position late
    fault: bool = false,
    T: usize = 0,
    pos0: usize = 0,
    q: []f32 = &.{},
    kv: []f32 = &.{},
    temp: [16]u16 = undefined,
    dev: [2]u64 = @splat(0),
    out: [3]u64 = @splat(0),
    /// the key and value caches, which the kernel writes at their absolute rows (pos0 + row)
    cache: [2]u64 = @splat(0),

    pub const sums = {};

    pub fn spec(self: *const Prep) c.Spec {
        return prepSpec(self.fault);
    }

    pub fn control(self: *const Prep) ?g.Gate {
        return if (self.fault) .shapes else null;
    }

    pub fn free(self: *Prep, a: std.mem.Allocator) void {
        a.free(self.q);
        a.free(self.kv);
        self.q = &.{};
        self.kv = &.{};
    }

    pub fn prepare(self: *Prep, a: std.mem.Allocator, rig: *ref.Rig, shape: c.Shape, fill: g.Fill, rng: std.Random) !bool {
        if (shape.rows == 0 or shape.n != 512 or shape.k != 32) return false;
        self.free(a);
        const T = shape.rows;
        self.T = T;
        // gate 4: row 0 at position 0, where the rotation is by 1 and 0
        self.pos0 = if (fill == .stress) 0 else rng.uintLessThan(usize, 4096);
        self.q = try ref.values(a, rng, T * 512, fill, 2);
        self.kv = try ref.values(a, rng, T * 256, fill, 2);
        for (&self.temp) |*v| v.* = c.toBf16(if (fill == .stress) 1 else rng.float(f32) * 4 - 2);
        if (fill == .nonfinite) {
            ref.plant(self.q, T, 512);
            self.kv[40] = ref.nan; // row 0, key head 1
        }
        self.dev = .{ try ref.input(rig, self.kv), try ref.input(rig, &self.temp) };
        self.out = .{ try rig.output(T * 512 * 4), try rig.output(T * 128 * 4), try rig.output(T * 128 * 4) };
        // the caches from row 0 (the control's rows land a row late: a row more keeps them inside)
        const cache = (self.pos0 + T + 1) * 128 * 4;
        self.cache = .{ try rig.scratch(cache), try rig.scratch(cache) };
        return true;
    }

    pub fn run(self: *Prep, rig: *ref.Rig) !void {
        try ref.restore(rig, self.out[0], std.mem.sliceAsBytes(self.q));
        const A = extern struct { pos0: i32, theta: f32 };
        try rig.go(self.pipe, .{ self.T, 1, 1 }, 512, &.{ self.out[0], self.dev[0], self.dev[1], self.cache[0], self.cache[1] }, A{ .pos0 = ref.int(self.pos0 + ref.one(self.fault)), .theta = theta }, 0);
        // the cache rows pos0 .. pos0 + T are the launch's outputs
        const len = self.T * 128 * 4;
        for (0..2) |i| @memcpy(rig.bytes(self.out[1 + i], len), rig.bytes(self.cache[i] + self.pos0 * 128 * 4, len));
    }

    pub fn count(self: *const Prep) usize {
        return self.T * 512 + 2 * self.T * 128;
    }

    /// A head's 32 values `x` scaled by `scale` over their rms and rotated in adjacent pairs at `pos`.
    fn rotate(s: Sums, x: []const f32, scale: ?Iv, pos: usize, out: []Iv) void {
        var scratch: [32]f64 = undefined;
        var sq: [32]Iv = undefined;
        for (&sq, x) |*v, y| v.* = ref.square(Iv.of(y));
        const inv = ref.rmsScale(s.sum(&sq, 5, &scratch), 32, 1e-6);
        var y: [32]Iv = undefined;
        for (&y, x) |*v, raw| {
            v.* = Iv.of(raw).mul(inv);
            if (scale) |m| v.* = v.mul(m);
        }
        for (0..16) |pair| {
            const exponent = -@as(f64, @floatFromInt(2 * pair)) / 32;
            const freq = std.math.pow(f64, theta, exponent);
            // precise::pow within 16 ulps
            const ang = Iv.of(@floatFromInt(pos)).mul(ref.wide(freq, freq, freq, 16 * 0x1p-23));
            const cs = ang.trig(false);
            const sn = ang.trig(true);
            out[2 * pair] = y[2 * pair].mul(cs).sub(y[2 * pair + 1].mul(sn));
            out[2 * pair + 1] = y[2 * pair].mul(sn).add(y[2 * pair + 1].mul(cs));
        }
    }

    pub fn solve(self: *const Prep, _: std.mem.Allocator, s: Sums, out: []Iv) !void {
        const T = self.T;
        const kc = out[T * 512 ..];
        const vc = out[T * 512 + T * 128 ..];
        for (0..T) |t| {
            for (0..16) |h| {
                const scale = Iv.of(@abs(ref.bf(self.temp[h])));
                rotate(s, self.q[t * 512 + h * 32 ..][0..32], scale, self.pos0 + t, out[t * 512 + h * 32 ..][0..32]);
            }
            for (0..4) |h| rotate(s, self.kv[t * 256 + h * 32 ..][0..32], null, self.pos0 + t, kc[t * 128 + h * 32 ..][0..32]);
            for (self.kv[t * 256 + 128 ..][0..128], 0..) |v, i| vc[t * 128 + i] = Iv.of(v);
        }
    }
};

/// Rows x [q | k | v] of a stage's width.
pub fn codecPrepSpec(fault: bool) c.Spec {
    return .{ .name = if (fault) "control: metal zonos3 zc_attn_prep rotating its rows one position late" else "metal zonos3 zc_attn_prep (q in place)", .accum = .none, .depth = ref.noSums, .out = .f32, .in_place = true, .served = &.{ .{ .rows = 16, .n = 2304, .k = 1 }, .{ .rows = 8, .n = 3840, .k = 1 }, .{ .rows = 1, .n = 2304, .k = 1 } } };
}

/// The codec's attention inputs: q and k rotated in adjacent pairs by position; k and v land in their cache rows.
pub const CodecPrep = struct {
    pipe: mtl.Pipeline,
    /// control: the device takes the rows one position late
    fault: bool = false,
    T: usize = 0,
    d: usize = 0,
    pos0: usize = 0,
    qkv: []f32 = &.{},
    out: [3]u64 = @splat(0),
    /// the key and value caches, a ring of `cap` rows the kernel writes at (pos0 + row) % cap
    cache: [2]u64 = @splat(0),
    cap: usize = 0,

    const rate: f32 = -9.210340371976184 * 2.0 / 64.0;

    pub fn spec(self: *const CodecPrep) c.Spec {
        return codecPrepSpec(self.fault);
    }

    pub fn control(self: *const CodecPrep) ?g.Gate {
        return if (self.fault) .shapes else null;
    }

    pub fn free(self: *CodecPrep, a: std.mem.Allocator) void {
        a.free(self.qkv);
        self.qkv = &.{};
    }

    pub fn prepare(self: *CodecPrep, a: std.mem.Allocator, rig: *ref.Rig, shape: c.Shape, fill: g.Fill, rng: std.Random) !bool {
        if (shape.rows == 0 or shape.n == 0 or shape.n % 192 != 0) return false;
        self.free(a);
        self.T = shape.rows;
        self.d = shape.n / 3;
        self.pos0 = rng.uintLessThan(usize, 2048);
        self.qkv = try ref.values(a, rng, self.T * shape.n, fill, 2);
        if (fill == .nonfinite) ref.plant(self.qkv, self.T, shape.n);
        self.out = .{ try rig.output(self.qkv.len * 4), try rig.output(self.T * self.d * 4), try rig.output(self.T * self.d * 4) };
        // one row more than a pass: the rows wrap past the ring's end wherever pos0 % cap leaves fewer than T after it
        self.cap = self.T + 1;
        const cache = self.cap * self.d * 4;
        self.cache = .{ try rig.scratch(cache), try rig.scratch(cache) };
        return true;
    }

    pub fn run(self: *CodecPrep, rig: *ref.Rig) !void {
        try ref.restore(rig, self.out[0], std.mem.sliceAsBytes(self.qkv));
        const A = extern struct { D: i32, pos0: i32, cap: i32 };
        try rig.go(self.pipe, .{ self.T, self.d / 64, 1 }, 32, &.{ self.out[0], self.cache[0], self.cache[1] }, A{ .D = ref.int(self.d), .pos0 = ref.int(self.pos0 + ref.one(self.fault)), .cap = ref.int(self.cap) }, 0);
        const row = self.d * 4;
        // the launch's outputs: positions pos0 .. pos0 + T, gathered from their ring rows
        for (0..2) |i| for (0..self.T) |t| @memcpy(rig.bytes(self.out[1 + i] + t * row, row), rig.bytes(self.cache[i] + ((self.pos0 + t) % self.cap) * row, row));
    }

    pub fn count(self: *const CodecPrep) usize {
        return self.T * 5 * self.d;
    }

    pub fn solve(self: *const CodecPrep, _: std.mem.Allocator, _: Sums, out: []Iv) !void {
        const T = self.T;
        const d = self.d;
        const kc = out[T * 3 * d ..];
        const vc = out[T * 4 * d ..];
        for (0..T) |t| {
            const row = self.qkv[t * 3 * d ..][0 .. 3 * d];
            for (0..d / 2) |pair| {
                const m: f64 = @floatFromInt(pair % 32);
                const phase = Iv.of(@floatFromInt(self.pos0 + t)).mul(Iv.of(m).mul(Iv.of(rate)).exp());
                const cs = phase.trig(false);
                const sn = phase.trig(true);
                for ([_][]Iv{ out[t * 3 * d ..], kc[t * d ..] }, [_]usize{ 0, d }) |dst, from| {
                    const re = Iv.of(row[from + 2 * pair]);
                    const im = Iv.of(row[from + 2 * pair + 1]);
                    dst[2 * pair] = re.mul(cs).sub(im.mul(sn));
                    dst[2 * pair + 1] = re.mul(sn).add(im.mul(cs));
                }
            }
            for (row[d..], 0..) |v, i| out[t * 3 * d + d + i] = Iv.of(v);
            for (row[2 * d ..], 0..) |v, i| vc[t * d + i] = Iv.of(v);
        }
    }
};

fn Pair(comptime K: type) type {
    return [2]ref.Gated(K);
}

/// The group's registrations, each followed by its control; the picking kernels' Set with them.
pub const Set = struct {
    prep: Pair(Prep),
    attention: [3]Pair(Attention),
    codec_prep: Pair(CodecPrep),
    pick: pick.Set,

    pub fn init(set: *Set, libs: *ref.Libs, rig: ref.Rig, list: ref.List) !void {
        const whiches = [3]Which{ .backbone, .codec, .teacher };
        const files = [3]ref.File{ .backbone, .codec, .teacher };
        const symbols = [3][]const u8{ "z3_attention", "zc_attention", "zt_attention" };
        for (0..2) |j| {
            const fault = j == 1;
            set.prep[j] = .{ .rig = rig, .k = .{ .pipe = try libs.pipe(.backbone, "z3_attn_prep"), .fault = fault } };
            for (0..3) |w| set.attention[w][j] = .{ .rig = rig, .k = .{ .pipe = try libs.pipe(files[w], symbols[w]), .which = whiches[w], .fault = fault } };
            set.codec_prep[j] = .{ .rig = rig, .k = .{ .pipe = try libs.pipe(.codec, "zc_attn_prep"), .fault = fault } };
        }
        for (&set.prep) |*x| try list.add(x.kernel());
        for (&set.attention) |*pair| for (pair) |*x| try list.add(x.kernel());
        for (&set.codec_prep) |*x| try list.add(x.kernel());
        try set.pick.init(libs, rig, list);
    }

    pub fn deinit(set: *Set) void {
        set.pick.deinit();
        for (&set.codec_prep) |*x| x.deinit();
        for (&set.attention) |*pair| for (pair) |*x| x.deinit();
        for (&set.prep) |*x| x.deinit();
    }
};
