//! The batched DSpark kernels over several streams: each stream's bytes must equal its own one-stream launch's.
const std = @import("std");
const core = @import("core");
const dk = @import("inkling2").dspark_ops;
const rig_ = @import("contract_rig.zig");
const ref = @import("contract_minimax_glue_ref.zig");

const c = core.contract.spec;
const g = core.contract.gates;

const D = 128;
const T = 8; // a draft block's rows
const hq = 24; // a TP=2 rank's query heads
const hk = 4;
const nqt = T * hq;
const hidden = 4608;
const R = 256; // the Markov rank
const ld = 2 * R + 2 * hidden; // rnn_proj's row
const ring = 4096; // a sliding layer's cache rows
const layer = 3; // a later layer of the table: a launch reading layer 0 finds no caches
const prev_rows = 4096; // w1's rows the walks' ids index
const eps: f32 = 1e-6;
const theta: f32 = 1e6;
const scale: f32 = 1.0 / @sqrt(@as(f32, D));
const rule = "every stream's bytes equal its own one-stream launch's on that stream's inputs";

/// The batched launch a registration checks.
pub const Kind = enum { qk, attn, ctx, rnn };

fn noDepth(_: c.Shape) usize {
    return 0;
}

const dw = (hq + 2 * hk) * D;
const aw = 2 * hk * D;
/// rows: streams; n: the call's width; k: the deepest end position, or the rows (ctx) and steps (rnn) a stream.
const qk_served = [_]c.Shape{ .{ .rows = 3, .n = dw, .k = 3007 }, .{ .rows = 4, .n = dw, .k = 70_008 }, .{ .rows = 3, .n = dw, .k = 1_048_576 }, .{ .rows = 3, .n = aw, .k = 2016 }, .{ .rows = 6, .n = aw, .k = 70_001 }, .{ .rows = 4, .n = aw, .k = 1_048_576 } };
const attn_served = [_]c.Shape{ .{ .rows = 3, .n = hq * D, .k = 3007 }, .{ .rows = 4, .n = hq * D, .k = 70_008 }, .{ .rows = 3, .n = hq * D, .k = 1_048_576 } };
const ctx_served = [_]c.Shape{ .{ .rows = 3, .n = 3 * R, .k = 7 }, .{ .rows = 4, .n = 3 * R, .k = 1 }, .{ .rows = 6, .n = 3 * R, .k = 8 } };
const rnn_served = [_]c.Shape{ .{ .rows = 3, .n = R, .k = 7 }, .{ .rows = 4, .n = R, .k = 1 }, .{ .rows = 5, .n = R, .k = 3 } };

pub fn spec(kind: Kind, fault: bool) c.Spec {
    const name = switch (kind) {
        .qk => if (fault) "inkling2 dspark control: tf_ink2_dsp_qk_many with one row tagged as the next stream's" else "inkling2 dspark tf_ink2_dsp_qk_many over 3+ streams, each its own tf_ink2_dsp_qk",
        .attn => if (fault) "inkling2 dspark control: tf_ink2_dsp_attn_many with two streams' key caches swapped in the table" else "inkling2 dspark tf_ink2_dsp_attn_many + merge over 3+ streams, each its own attend",
        .ctx => if (fault) "inkling2 dspark control: tf_ink2_dsp_ctx_many told the packed hidden stride" else "inkling2 dspark tf_ink2_dsp_ctx_many over 3+ streams, each its own tf_ink2_dsp_ctx",
        .rnn => if (fault) "inkling2 dspark control: tf_ink2_dsp_rnn_many told a walk stride one id short" else "inkling2 dspark tf_ink2_dsp_rnn_many over 3+ streams, each its own tf_ink2_dsp_rnn",
    };
    const one = switch (kind) {
        .qk => "each stream's own tf_ink2_dsp_qk",
        .attn => "each stream's own tf_ink2_dsp_attn + tf_ink2_dsp_merge",
        .ctx => "each stream's own tf_ink2_dsp_ctx",
        .rnn => "each stream's own tf_ink2_dsp_rnn",
    };
    const served: []const c.Shape = switch (kind) {
        .qk => &qk_served,
        .attn => &attn_served,
        .ctx => &ctx_served,
        .rnn => &rnn_served,
    };
    return .{ .name = name, .accum = .none, .depth = noDepth, .out = .raw, .in_place = kind == .rnn, .replaces = one, .nan = .{ .stated = rule }, .served = served };
}

/// An output and the scratch its one-stream launches write instead; `init` is what both hold before a run.
const Pair = struct { out: u64, base: u64, len: usize, init: ?[]const u8 = null };

/// One batched launch over several streams, and the same streams launched one at a time.
pub const Many = struct {
    rig: rig_.Rig,
    k: *const dk.Kernels,
    kind: Kind,
    /// control: a wrong table entry (qk, attn) or a wrong stride (ctx, rnn) on the batched launch only
    fault: bool = false,
    arena: std.heap.ArenaAllocator,
    shape: c.Shape = undefined,
    n: usize = 0,
    in: [6]u64 = @splat(0),
    tab: dk.Table = .{ .dev = 0, .host = &.{} },
    rows: dk.Rows = .{ .dev = 0, .host = &.{} },
    /// per stream: its first row, position and rows (qk), or its window [lo, hi] and caches (attn)
    first: []usize = &.{},
    lo: []usize = &.{},
    hi: []usize = &.{},
    caches: []u64 = &.{},
    pairs: std.ArrayList(Pair) = .empty,
    maxc: usize = 0,
    xld: usize = 0,
    base: std.ArrayList(u8) = .empty,

    pub fn deinit(self: *Many) void {
        self.base.deinit(self.rig.gpu.gpa);
        self.arena.deinit();
        self.rig.deinit();
    }

    pub fn kernel(self: *Many) g.Kernel {
        return .{ .spec = spec(self.kind, self.fault), .ctx = self, .vt = &vt, .control = if (self.fault) .shapes else null };
    }

    const vt = g.hooks(Many);

    pub fn prepare(self: *Many, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        self.rig.reset();
        _ = self.arena.reset(.free_all);
        self.pairs = .empty;
        if (shape.rows < 3 or shape.rows > 16) return false;
        self.shape = shape;
        self.n = shape.rows;
        var prng = std.Random.DefaultPrng.init(seed);
        const a = self.arena.allocator();
        return switch (self.kind) {
            .qk => self.prepQk(a, fill, prng.random()),
            .attn => self.prepAttn(a, fill, prng.random()),
            .ctx => self.prepCtx(a, fill, prng.random()),
            .rnn => self.prepRnn(a, fill, prng.random()),
        };
    }

    /// An output poisoned as the rig poisons it, and its baseline twin.
    fn twin(self: *Many, len: usize, init: ?[]const u8) !u64 {
        const p: Pair = .{ .out = try self.rig.output(len), .base = try self.rig.scratch(len), .len = len, .init = init };
        try self.pairs.append(self.arena.allocator(), p);
        if (init) |b| try ref.restore(&self.rig, p.out, b);
        return p.out;
    }

    fn hqOf(self: *const Many) usize {
        return if (self.shape.n == dw) hq else 0;
    }

    /// Streams laid out last first in the rows, a draft block each or 1 to 7 absorbed rows, ends far apart.
    fn prepQk(self: *Many, a: std.mem.Allocator, fill: g.Fill, rng: std.Random) !bool {
        if (self.shape.n != dw and self.shape.n != aw) return false;
        const n = self.n;
        const h = self.hqOf();
        self.first = try a.alloc(usize, n);
        self.lo = try a.alloc(usize, n);
        self.hi = try a.alloc(usize, n);
        var total: usize = 0;
        for (0..n) |i| {
            self.hi[i] = if (h > 0) T else 1 + (3 * i + 2) % 7; // the stream's rows
            const end = if (self.shape.k > i * 4999 + self.hi[i]) self.shape.k - i * 4999 else self.hi[i] + 37 * i;
            self.lo[i] = end - self.hi[i];
            total += self.hi[i];
        }
        const at = try a.alloc([2]i32, total);
        var r: usize = 0;
        var i = n;
        while (i > 0) {
            i -= 1;
            self.first[i] = r;
            for (0..self.hi[i]) |j| at[r + j] = .{ @intCast(i), @intCast(self.lo[i] + j) };
            r += self.hi[i];
        }
        if (self.fault) at[self.first[1]][0] = 2;
        self.xld = self.shape.n + 64; // padded rows: the launches must honour the stride
        const x = try ref.bfs(a, rng, total * self.xld, -2, 2);
        if (fill == .nonfinite) {
            x[3] = c.toBf16(ref.nan);
            x[self.xld + D + 9] = c.toBf16(ref.inf);
            x[2 * self.xld + h * D + 70] = c.toBf16(-ref.inf);
        }
        self.in = .{ try ref.input(&self.rig, x), try ref.input(&self.rig, try ref.bfs(a, rng, D, 0.5, 1.5)), try ref.input(&self.rig, try ref.bfs(a, rng, D, 0.5, 1.5)), 0, 0, 0 };
        if (h > 0) _ = try self.twin(total * h * D * 2, null);
        const st = try a.alloc(dk.Stream, n);
        for (st) |*s| {
            s.* = .{};
            s.kc[layer] = try self.twin(ring * hk * D * 2, null);
            s.vc[layer] = try self.twin(ring * hk * D * 2, null);
        }
        self.tab = .{ .dev = try ref.input(&self.rig, st), .host = st };
        self.rows = .{ .dev = try ref.input(&self.rig, at), .host = at };
        return true;
    }

    /// Windows of a whole ring, a block and a few keys, and 1,030 keys across a chunk edge, each in its own caches.
    fn prepAttn(self: *Many, a: std.mem.Allocator, fill: g.Fill, rng: std.Random) !bool {
        if (self.shape.n != hq * D or self.shape.k < 2 * T) return false;
        const n = self.n;
        self.lo = try a.alloc(usize, n);
        self.hi = try a.alloc(usize, n);
        self.caches = try a.alloc(u64, 2 * n);
        const st = try a.alloc(dk.Stream, n);
        const q = try a.alloc(u16, n * nqt * D);
        for (q) |*v| v.* = c.toBf16((rng.float(f32) * 2 - 1) * 4);
        if (fill == .nonfinite) q[3] = c.toBf16(ref.nan);
        self.maxc = 0;
        for (0..n) |i| {
            self.hi[i] = if (self.shape.k > i * 1500 + 2 * T) self.shape.k - 1 - i * 1500 else T - 1 + 5 * i;
            const keys: usize = switch (i % 3) {
                0 => ring,
                1 => T + 3 * i,
                else => 1030 + i,
            };
            self.lo[i] = self.hi[i] + 1 - @min(keys, self.hi[i] + 1);
            self.maxc = @max(self.maxc, dk.chunks(self.lo[i], self.hi[i]));
            const kv = try ref.bfs(a, rng, 2 * ring * hk * D, -1, 1);
            if (fill == .nonfinite and i == 1) kv[((self.lo[i] + 2) % ring * hk + 1) * D + 5] = c.toBf16(ref.inf);
            self.caches[2 * i] = try ref.input(&self.rig, kv[0 .. ring * hk * D]);
            self.caches[2 * i + 1] = try ref.input(&self.rig, kv[ring * hk * D ..]);
            st[i] = .{ .p = @intCast(self.hi[i] + 1 - T) };
            st[i].lo[layer] = @intCast(self.lo[i]);
            st[i].kc[layer] = self.caches[2 * i];
            st[i].vc[layer] = self.caches[2 * i + 1];
        }
        if (self.fault) std.mem.swap(u64, &st[1].kc[layer], &st[2].kc[layer]);
        self.in = .{ try ref.input(&self.rig, q), try self.rig.scratch(n * self.maxc * nqt * D * 4), try self.rig.scratch(n * self.maxc * nqt * 2 * 4), try self.rig.scratch(self.maxc * nqt * D * 4), try self.rig.scratch(self.maxc * nqt * 2 * 4), 0 };
        _ = try self.twin(n * nqt * D * 2, null);
        self.tab = .{ .dev = try ref.input(&self.rig, st), .host = st };
        return true;
    }

    /// Each stream's hidden rows a padded stride apart, its own f_rnn from the table.
    fn prepCtx(self: *Many, a: std.mem.Allocator, fill: g.Fill, rng: std.Random) !bool {
        const rows = self.shape.k;
        if (self.shape.n != 3 * R or rows == 0 or rows > dk.max_rows) return false;
        const n = self.n;
        self.xld = rows * hidden + 64;
        const h = try ref.bfs(a, rng, n * self.xld, -2, 2);
        const bias = try ref.bfs(a, rng, 3 * R, -1, 1);
        if (fill == .nonfinite) {
            h[11] = c.toBf16(ref.nan);
            h[self.xld + 7] = c.toBf16(ref.inf);
            bias[5] = c.toBf16(ref.nan);
        }
        self.in = .{ try ref.input(&self.rig, h), try ref.input(&self.rig, try ref.bfs(a, rng, n * hidden, -2, 2)), try ref.input(&self.rig, try ref.bfs(a, rng, 3 * R * ld, -0.5, 0.5)), try ref.input(&self.rig, bias), 0, 0 };
        const st = try a.alloc(dk.Stream, n);
        for (st, 0..) |*s, i| s.* = .{ .frnn = self.in[1] + i * hidden * 2 };
        _ = try self.twin(n * rows * 3 * R * 4, null);
        self.tab = .{ .dev = try ref.input(&self.rig, st), .host = st };
        return true;
    }

    /// Each stream's context rows and walk a padded stride apart; its state updated in place, a latent a step.
    fn prepRnn(self: *Many, a: std.mem.Allocator, fill: g.Fill, rng: std.Random) !bool {
        const steps = self.shape.k;
        if (self.shape.n != R or steps == 0 or steps > dk.max_rows - 1) return false;
        const n = self.n;
        self.xld = (steps + 1) * 3 * R; // ctx floats a stream: one spare row
        const ctx = try ref.f32s(a, rng, n * self.xld, -2, 2);
        if (fill == .nonfinite) {
            ctx[5] = ref.nan;
            ctx[self.xld + R + 9] = -ref.inf;
        }
        const walk = try a.alloc(i32, n * (steps + 2));
        for (walk) |*p| p.* = @intCast(rng.uintLessThan(u32, prev_rows));
        self.in = .{ try ref.input(&self.rig, ctx), try ref.input(&self.rig, try ref.bfs(a, rng, 3 * R * ld, -0.1, 0.1)), try ref.input(&self.rig, try ref.bfs(a, rng, prev_rows * R, -1, 1)), try ref.input(&self.rig, walk), 0, 0 };
        _ = try self.twin(n * R * 2, std.mem.sliceAsBytes(try ref.bfs(a, rng, n * R, -1, 1)));
        _ = try self.twin(steps * n * R * 2, null);
        return true;
    }

    pub fn run(self: *Many) !void {
        const s = self.rig.s;
        const p = self.pairs.items;
        const n = self.n;
        switch (self.kind) {
            .qk => {
                const h = self.hqOf();
                try dk.qkMany(self.k, s, self.qkArgs(if (h > 0) p[0].out else 0, 0, 0, 0), self.tab, self.rows, layer);
            },
            .attn => try dk.attendMany(self.k, s, self.attnArgs(self.in[0], 0, 0, self.in[1], self.in[2], 0, 0), self.tab, layer, self.maxc, n * self.maxc, p[0].out),
            .ctx => {
                const rows = self.shape.k;
                const stride = if (self.fault) rows * hidden else self.xld;
                try dk.ctxMany(self.k, s, self.in[0], stride, self.tab.dev, self.in[2], self.in[3], p[0].out, n, rows, hidden, R);
            },
            .rnn => {
                try ref.restore(&self.rig, p[0].out, p[0].init.?);
                const wt = self.shape.k + 2 - @intFromBool(self.fault);
                for (0..self.shape.k) |j| try dk.rnnMany(self.k, s, self.in[0] + j * 3 * R * 4, self.xld, self.in[1], self.in[2], self.in[3], wt, j, p[0].out, p[1].out + j * n * R * 2, n, R, hidden);
            },
        }
        try s.synchronize();
    }

    fn qkArgs(self: *const Many, q: u64, kc: u64, vc: u64, pos0: usize) dk.QkArgs {
        return .{ .x = self.in[0], .qw = if (self.hqOf() > 0) self.in[1] else 0, .kw = self.in[2], .q = q, .kc = kc, .vc = vc, .ld = @intCast(self.xld), .hq = @intCast(self.hqOf()), .hk = hk, .ring = ring, .pos0 = @intCast(pos0), .eps = eps, .theta = theta };
    }

    fn attnArgs(_: *const Many, q: u64, kc: u64, vc: u64, po: u64, pml: u64, lo: usize, hi: usize) dk.AttnArgs {
        return .{ .q = q, .kc = kc, .vc = vc, .po = po, .pml = pml, .T = T, .hq = hq, .hk = hk, .ring = ring, .lo = @intCast(lo), .hi = @intCast(hi), .c0 = 0, .scale = scale };
    }

    /// The streams one at a time through the one-stream kernels, into the outputs' twins, in the outputs' order.
    pub fn replaced(self: *Many) ![]const u8 {
        const gpa = self.rig.gpu.gpa;
        const s = self.rig.s;
        const p = self.pairs.items;
        const n = self.n;
        for (p) |x| {
            if (x.init) |b| {
                try ref.restore(&self.rig, x.base, b);
                continue;
            }
            const poison = try gpa.alloc(u8, x.len);
            defer gpa.free(poison);
            @memset(poison, 0xff);
            try ref.restore(&self.rig, x.base, poison);
        }
        switch (self.kind) {
            .qk => {
                const h = self.hqOf();
                const kv = @intFromBool(h > 0);
                for (0..n) |i| {
                    var q = self.qkArgs(if (h > 0) p[0].base + self.first[i] * h * D * 2 else 0, p[kv + 2 * i].base, p[kv + 2 * i + 1].base, self.lo[i]);
                    q.x += self.first[i] * self.xld * 2;
                    try dk.qk(self.k, s, q, self.hi[i]);
                }
            },
            .attn => for (0..n) |i| {
                const at = self.attnArgs(self.in[0] + i * nqt * D * 2, self.caches[2 * i], self.caches[2 * i + 1], self.in[3], self.in[4], self.lo[i], self.hi[i]);
                try dk.attend(self.k, s, at, p[0].base + i * nqt * D * 2);
            },
            .ctx => {
                const rows = self.shape.k;
                for (0..n) |i| try dk.ctx(self.k, s, self.in[0] + i * self.xld * 2, self.in[1] + i * hidden * 2, self.in[2], self.in[3], p[0].base + i * rows * 3 * R * 4, rows, hidden, R);
            },
            .rnn => for (0..self.shape.k) |j| for (0..n) |i| {
                const walk = self.in[3] + i * (self.shape.k + 2) * 4;
                try dk.rnn(self.k, s, self.in[0] + (i * self.xld + j * 3 * R) * 4, self.in[1], self.in[2], walk, j, p[0].base + i * R * 2, p[1].base + (j * n + i) * R * 2, R, hidden);
            },
        }
        self.base.clearRetainingCapacity();
        for (p) |x| try self.base.appendSlice(gpa, try self.rig.bytes(x.base, x.len));
        return self.base.items;
    }

    /// Gate 1's own check: the batched bytes, NaN and infinities included, equal the one-stream launches'.
    pub fn check(self: *Many, _: g.Gate) !?[]const u8 {
        const gpa = self.rig.gpu.gpa;
        const got = try gpa.dupe(u8, try self.rig.outputBytes());
        defer gpa.free(got);
        return if (std.mem.eql(u8, got, try self.replaced())) null else rule;
    }

    pub fn count(self: *const Many) usize {
        var bytes: usize = 0;
        for (self.pairs.items) |x| bytes += x.len;
        return bytes;
    }

    pub fn reference(_: *Many, _: []f64, _: []f64) !void {
        return error.ReplacementHasNoReference;
    }
};
