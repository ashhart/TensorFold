//! MiniMax-M3.1-Flash's CUDA data moves and tower glue under the contract: copies, conversions, RoPE, patch embedding.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const mm3 = @import("minimax_m3_cuda");
const rig_ = @import("contract_rig.zig");
const ref = @import("contract_minimax_glue_ref.zig");

const c = core.contract.spec;
const g = core.contract.gates;
const kern = mm3.kernels;
const Iv = ref.Iv;
const Sums = ref.Sums;
const Kernels = kern.Kernels;

const hidden = 6144;
const taps = 5;
const vocab = 1000; // the test's table rows (the model's: 200,064)

pub const Move = enum { embed, splice, tap };

const move_nan = "each value as it is (to bf16: nearest even, past bf16's largest to +-Inf); a NaN stays NaN";

/// Rows x the row's elements x 1: every output exact (float64's value of the source, or its bf16 rounding).
pub fn moveSpec(m: Move) c.Spec {
    const rows = &[_]c.Shape{ .{ .rows = 16, .n = hidden, .k = 1 }, .{ .rows = 1, .n = hidden, .k = 1 }, .{ .rows = 7, .n = hidden, .k = 1 }, .{ .rows = 512, .n = hidden, .k = 1 } };
    const images = &[_]c.Shape{ .{ .rows = 64, .n = hidden, .k = 1 }, .{ .rows = 1, .n = hidden, .k = 1 }, .{ .rows = 576, .n = hidden, .k = 1 } };
    const elementwise: c.Tile = .{ .size = 256, .step = 1, .min = 1 };
    return switch (m) {
        .embed => .{ .name = "cuda minimax tf_mm3_embed", .accum = .none, .depth = ref.noSums, .out = .f32, .nan = .{ .stated = move_nan }, .tiles = .{ .{}, elementwise, .{} }, .served = rows },
        .splice => .{ .name = "cuda minimax tf_mm3_splice (image rows over the residual)", .accum = .none, .depth = ref.noSums, .out = .f32, .nan = .{ .stated = move_nan }, .served = images },
        .tap => .{ .name = "cuda minimax tf_mm3_tap (a drafter's feature rows)", .accum = .none, .depth = ref.noSums, .out = .bf16, .nan = .{ .stated = move_nan }, .served = rows },
    };
}

/// A copy or conversion: the source (bf16 bits or fp32 values), and for tap the feature rows' other slots as they were.
pub const Copy = struct {
    kern: *const Kernels,
    move: Move,
    shape: c.Shape = undefined,
    src: []u32 = &.{}, // fp32 bits, or bf16 bits widened
    ids: []i32 = &.{},
    before: []u16 = &.{}, // tap: the feature rows before the launch
    dev: [2]u64 = @splat(0),
    out: u64 = 0,

    pub fn spec(self: *const Copy) c.Spec {
        return moveSpec(self.move);
    }

    pub fn control(_: *const Copy) ?g.Gate {
        return null;
    }

    pub fn free(self: *Copy, a: std.mem.Allocator) void {
        a.free(self.src);
        a.free(self.ids);
        a.free(self.before);
        self.src = &.{};
        self.ids = &.{};
        self.before = &.{};
    }

    fn bf16In(self: *const Copy) bool {
        return self.move != .tap;
    }

    fn which(self: *const Copy) usize {
        return self.shape.rows % taps;
    }

    /// Values over fp32's exponents (subnormals too, none past bf16's largest); non-finite: NaN, +-Inf, fp32's max.
    pub fn prepare(self: *Copy, a: std.mem.Allocator, rig: *rig_.Rig, shape: c.Shape, fill: g.Fill, rng: std.Random) !bool {
        if (shape.rows == 0 or shape.n == 0) return false;
        self.free(a);
        self.shape = shape;
        const n = if (self.move == .embed) vocab * shape.n else shape.rows * shape.n;
        self.src = try a.alloc(u32, n);
        for (self.src) |*v| {
            const sign = rng.int(u32) & 0x8000_0000;
            const e: u32 = rng.uintLessThan(u32, 254); // 0: subnormals; below 254: no rounding past bf16's largest
            v.* = sign | (e << 23) | (rng.int(u32) & 0x7f_ffff);
            if (self.bf16In()) v.* = c.toBf16(@bitCast(v.*));
        }
        if (self.move == .embed) {
            self.ids = try a.alloc(i32, shape.rows);
            for (self.ids, 0..) |*id, r| id.* = @intCast(if (r == 0) vocab - 1 else rng.uintLessThan(u32, vocab));
            self.dev[1] = try ref.input(rig, self.ids);
        }
        if (fill == .nonfinite) {
            const put = [_]f32{ ref.nan, ref.inf, -ref.inf, std.math.floatMax(f32), -std.math.floatMax(f32) };
            for (put, 0..) |x, i| {
                const row = if (self.move == .embed) @as(usize, @intCast(self.ids[i % shape.rows])) else i % shape.rows;
                const at = (row * shape.n + i + 1) % n; // a row the call reads
                self.src[at] = if (self.bf16In()) c.toBf16(x) else @bitCast(x);
            }
        }
        if (self.bf16In()) {
            const bits = try a.alloc(u16, n);
            defer a.free(bits);
            for (bits, self.src) |*b, v| b.* = @truncate(v);
            self.dev[0] = try ref.input(rig, bits);
        } else self.dev[0] = try ref.input(rig, self.src);
        const outs = shape.rows * shape.n * (if (self.move == .tap) @as(usize, taps) else 1);
        if (self.move == .tap) self.before = try ref.bfs(a, rng, outs, -4, 4);
        self.out = try rig.output(outs * (if (self.spec().out == .f32) @as(usize, 4) else 2));
        return true;
    }

    pub fn run(self: *Copy, rig: *rig_.Rig) !void {
        const rows = self.shape.rows;
        const n = self.shape.n;
        const k = self.kern;
        switch (self.move) {
            .embed => try kern.embed(k, rig.s, self.dev[1], self.dev[0], self.out, rows, n),
            .splice => try kern.splice(k, rig.s, self.dev[0], self.out, rows, n),
            .tap => {
                try ref.restore(rig, self.out, std.mem.sliceAsBytes(self.before)); // the feature rows' other slots
                try kern.tap(k, rig.s, self.dev[0], self.out, rows, n, taps, self.which());
            },
        }
    }

    pub fn count(self: *const Copy) usize {
        return self.shape.rows * self.shape.n * (if (self.move == .tap) @as(usize, taps) else 1);
    }

    fn value(self: *const Copy, i: usize) f64 {
        return if (self.bf16In()) ref.bf(@truncate(self.src[i])) else @as(f32, @bitCast(self.src[i]));
    }

    pub fn solve(self: *const Copy, _: std.mem.Allocator, _: Sums, out: []Iv) !void {
        const n = self.shape.n;
        switch (self.move) {
            .embed => for (0..self.shape.rows) |r| for (0..n) |i| {
                out[r * n + i] = Iv.of(self.value(@as(usize, @intCast(self.ids[r])) * n + i));
            },
            .splice => for (out, 0..) |*o, i| {
                o.* = Iv.of(self.value(i));
            },
            .tap => for (out, 0..) |*o, i| {
                const r = i / (taps * n);
                o.* = if (i / n % taps == self.which()) Iv.of(self.value(r * n + i % n)).bf() else Iv.of(ref.bf(self.before[i]));
            },
        }
    }
};

/// A step's header and ids as `put_words` sends them: one launch of 128 threads, `n` of 96 words.
pub fn wordsSpec() c.Spec {
    return .{ .name = "cuda minimax tf_mm3_put_words", .accum = .none, .depth = ref.noSums, .out = .raw, .nan = .{ .stated = "every word's bits as given, NaN patterns included; words past n untouched" }, .served = &.{ .{ .rows = 1, .n = 16, .k = 1 }, .{ .rows = 6, .n = 96, .k = 1 }, .{ .rows = 4, .n = 41, .k = 1 } } };
}

const Words = extern struct { w: [96]u32 };

pub const PutWords = struct {
    kern: *const Kernels,
    shape: c.Shape = undefined,
    words: []Words = &.{},
    out: u64 = 0,

    pub fn spec(_: *const PutWords) c.Spec {
        return wordsSpec();
    }

    pub fn control(_: *const PutWords) ?g.Gate {
        return null;
    }

    pub fn free(self: *PutWords, a: std.mem.Allocator) void {
        a.free(self.words);
        self.words = &.{};
    }

    pub fn prepare(self: *PutWords, a: std.mem.Allocator, rig: *rig_.Rig, shape: c.Shape, fill: g.Fill, rng: std.Random) !bool {
        if (shape.rows == 0 or shape.n == 0 or shape.n > 96) return false;
        self.free(a);
        self.shape = shape;
        self.words = try a.alloc(Words, shape.rows);
        for (self.words) |*w| for (&w.w) |*v| {
            v.* = rng.int(u32);
        };
        if (fill == .nonfinite) self.words[0].w[1] = @bitCast(ref.nan);
        self.out = try rig.output(shape.rows * 96 * 4);
        return true;
    }

    pub fn run(self: *PutWords, rig: *rig_.Rig) !void {
        for (self.words, 0..) |w, j| {
            var a: cuda.Args = .{};
            a.add(self.out + j * 96 * 4);
            a.add(@as(u32, @intCast(self.shape.n)));
            a.add(w);
            try cuda.launch.launch(self.kern.put_words, .{ .grid = .{ .x = 1 }, .block = .{ .x = 128 } }, rig.s, &a);
        }
    }

    pub fn count(self: *const PutWords) usize {
        return self.shape.rows * 96 * 4;
    }

    pub fn check(self: *const PutWords, _: std.mem.Allocator, got: []const u8, msg: []u8) !?[]const u8 {
        for (self.words, 0..) |w, j| for (w.w, 0..) |v, i| {
            const want: u32 = if (i < self.shape.n) v else 0xffff_ffff;
            const have = std.mem.readInt(u32, got[(j * 96 + i) * 4 ..][0..4], .little);
            if (have != want) return try std.fmt.bufPrint(msg, "row {d} word {d}: {x} where {x} is due", .{ j, i, have, want });
        };
        return null;
    }
};

const tower_heads = 16;
const head_dim = 80;
const pairs = 39;

/// Patches x qkv's 3 x 1,280 x 1: 16 patches (a 56 x 56 image) to the processor's 2,304.
pub fn ropeSpec() c.Spec {
    const served = &[_]c.Shape{ .{ .rows = 64, .n = 3 * 1280, .k = 1 }, .{ .rows = 16, .n = 3 * 1280, .k = 1 }, .{ .rows = 2304, .n = 3 * 1280, .k = 1 } };
    return .{ .name = "cuda minimax tf_mm3_vit_rope (in place)", .accum = .none, .depth = ref.noSums, .out = .bf16, .in_place = true, .served = served };
}

/// q and k's pairs (i, i + 39) rotated by fp32 cos and sin: a c - b s and b c + a s in fp32, rounded to bf16 once.
pub const VitRope = struct {
    kern: *const Kernels,
    shape: c.Shape = undefined,
    qkv: []u16 = &.{},
    cs: []f32 = &.{},
    sn: []f32 = &.{},
    dev: [2]u64 = @splat(0),
    out: u64 = 0,

    pub fn spec(_: *const VitRope) c.Spec {
        return ropeSpec();
    }

    pub fn control(_: *const VitRope) ?g.Gate {
        return null;
    }

    pub fn free(self: *VitRope, a: std.mem.Allocator) void {
        a.free(self.qkv);
        a.free(self.cs);
        a.free(self.sn);
        self.qkv = &.{};
        self.cs = &.{};
        self.sn = &.{};
    }

    pub fn prepare(self: *VitRope, a: std.mem.Allocator, rig: *rig_.Rig, shape: c.Shape, fill: g.Fill, rng: std.Random) !bool {
        if (shape.rows == 0 or shape.n != 3 * tower_heads * head_dim) return false;
        self.free(a);
        self.shape = shape;
        const p = shape.rows;
        self.qkv = try ref.bfs(a, rng, p * shape.n, -3, 3);
        self.cs = try a.alloc(f32, p * pairs);
        self.sn = try a.alloc(f32, p * pairs);
        for (self.cs, self.sn) |*x, *y| {
            const angle = rng.float(f32) * 100;
            x.* = @cos(angle);
            y.* = @sin(angle);
        }
        if (fill == .nonfinite) {
            self.qkv[3] = c.toBf16(ref.nan); // patch 0, q head 0: pair 3 NaN
            self.qkv[shape.n + 1280 + 80 + 10] = c.toBf16(ref.inf); // patch 1, k head 1
            self.qkv[2 * shape.n + 2560 + 5] = c.toBf16(-ref.inf); // patch 2, v: as it is
            self.qkv[3 * shape.n + 79] = c.toBf16(ref.nan); // patch 3, q head 0's last dim: past the pairs
        }
        self.dev = .{ try ref.input(rig, self.cs), try ref.input(rig, self.sn) };
        self.out = try rig.output(self.qkv.len * 2);
        return true;
    }

    pub fn run(self: *VitRope, rig: *rig_.Rig) !void {
        try ref.restore(rig, self.out, std.mem.sliceAsBytes(self.qkv)); // rotated in place: every run from the same rows
        try kern.vitRope(self.kern, rig.s, self.out, self.dev[0], self.dev[1], self.shape.rows);
    }

    pub fn count(self: *const VitRope) usize {
        return self.qkv.len;
    }

    pub fn solve(self: *const VitRope, _: std.mem.Allocator, _: Sums, out: []Iv) !void {
        for (out, self.qkv, 0..) |*o, x, at| {
            const d = at % head_dim;
            const part = at / 1280 % 3;
            if (part == 2 or d >= 2 * pairs) {
                o.* = Iv.of(ref.bf(x));
                continue;
            }
            const head = at - d;
            const i = d % pairs;
            const p = at / self.shape.n;
            const cv = Iv.of(self.cs[p * pairs + i]);
            const sv = Iv.of(self.sn[p * pairs + i]);
            const av = Iv.of(ref.bf(self.qkv[head + i]));
            const bv = Iv.of(ref.bf(self.qkv[head + i + pairs]));
            o.* = (if (d < pairs) av.mul(cv).sub(bv.mul(sv)) else bv.mul(cv).add(av.mul(sv))).bf();
        }
    }
};

fn patchDepth(s: c.Shape) usize {
    return s.k; // one fma chain in k order
}

/// Patches x the tower's width (1,280) x the patch's 1,176 pixels: 64 x 64 tiles, 16 pixels a step.
pub fn patchSpec() c.Spec {
    const served = &[_]c.Shape{ .{ .rows = 576, .n = 1280, .k = 1176 }, .{ .rows = 16, .n = 1280, .k = 1176 }, .{ .rows = 2304, .n = 1280, .k = 1176 } };
    return .{ .name = "cuda minimax tf_mm3_patch_embed (fp32 pixels and weights, bf16 out)", .accum = .f32, .depth = patchDepth, .out = .bf16, .tiles = .{ .{ .size = 64 }, .{ .size = 64 }, .{ .size = 16 } }, .served = served };
}

/// Y = bf16(X W^T), each output one fp32 fma chain over k: gamma(k) of its terms either side, then one rounding.
pub const PatchEmbed = struct {
    kern: *const Kernels,
    shape: c.Shape = undefined,
    x: []f32 = &.{},
    w: []f32 = &.{},
    dev: [2]u64 = @splat(0),
    out: u64 = 0,

    pub const sums = {};

    pub fn spec(_: *const PatchEmbed) c.Spec {
        return patchSpec();
    }

    pub fn control(_: *const PatchEmbed) ?g.Gate {
        return null;
    }

    pub fn free(self: *PatchEmbed, a: std.mem.Allocator) void {
        a.free(self.x);
        a.free(self.w);
        self.x = &.{};
        self.w = &.{};
    }

    /// Pixels in [-1.8, 2.2), weights in [-0.05, 0.05); stress: integers, the second half cancelling the first.
    pub fn prepare(self: *PatchEmbed, a: std.mem.Allocator, rig: *rig_.Rig, shape: c.Shape, fill: g.Fill, rng: std.Random) !bool {
        if (shape.rows == 0 or shape.n == 0 or shape.k == 0) return false;
        self.free(a);
        self.shape = shape;
        const k = shape.k;
        self.x = try ref.f32s(a, rng, shape.rows * k, -1.8, 2.2);
        self.w = try ref.f32s(a, rng, shape.n * k, -0.05, 0.05);
        if (fill == .stress) {
            const h = k / 2;
            for (0..shape.rows) |r| for (0..h) |i| {
                const v = ref.int(rng, 1, 64);
                self.x[r * k + i] = v;
                self.x[r * k + h + i] = v + @as(f32, if (i < 8) 1 else 0);
            };
            for (0..shape.n) |e| for (0..h) |i| {
                const v = ref.int(rng, 1, 7);
                self.w[e * k + i] = v;
                self.w[e * k + h + i] = -v;
            };
        }
        if (fill == .nonfinite) {
            self.x[k / 3] = ref.nan;
            self.w[5 * k + k / 2] = ref.inf; // column 5: every patch's +-Inf
            if (shape.rows > 3) {
                self.x[3 * k] = ref.inf;
                self.x[3 * k + k - 1] = -ref.inf;
            }
        }
        self.dev = .{ try ref.input(rig, self.x), try ref.input(rig, self.w) };
        self.out = try rig.output(shape.rows * shape.n * 2);
        return true;
    }

    pub fn run(self: *PatchEmbed, rig: *rig_.Rig) !void {
        try kern.patchEmbed(self.kern, rig.s, self.dev[0], self.dev[1], self.out, self.shape.rows, self.shape.n, self.shape.k);
    }

    pub fn count(self: *const PatchEmbed) usize {
        return self.shape.rows * self.shape.n;
    }

    pub fn solve(self: *const PatchEmbed, a: std.mem.Allocator, s: Sums, out: []Iv) !void {
        const k = self.shape.k;
        const terms = try a.alloc(Iv, k);
        defer a.free(terms);
        const scratch = try a.alloc(f64, k);
        defer a.free(scratch);
        for (0..self.shape.rows) |r| for (0..self.shape.n) |e| {
            for (terms, self.x[r * k ..][0..k], self.w[e * k ..][0..k]) |*t, x, w| t.* = Iv.of(@as(f64, x) * w);
            out[r * self.shape.n + e] = s.sum(terms, k, scratch).bf();
        };
    }
};
