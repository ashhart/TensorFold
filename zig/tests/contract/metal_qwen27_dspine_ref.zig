//! DSpine's eight Metal kernels on the host: inputs as each gate fills them, float64 outputs and simulated sums.
const std = @import("std");
const contract = @import("contract");
const c = contract.spec;
const Order = contract.rows.Order;
const Fill = contract.gates.Fill;

/// The stock 27B drafter (qwen27/dspine/config.zig pin): hidden, code, rank, block and the head's top-k.
pub const hidden = 5120;
pub const code = 1024;
pub const rank = 512;
pub const block = 16;
pub const topk = 16;
/// A stream's head rows: the block's rows after its anchor.
pub const heads = block - 1;
/// Rows the codebook, head and TG hold here: ids are drawn below it (a kernel reads only the rows it is given).
pub const vocab = 1024;
/// core.bf16_topk.Row in words: nonfinite, count, then 16 (token, score) pairs.
pub const row_words = 2 + 2 * 16;
pub const eps: f32 = 1e-6;
const scale: f32 = 32; // sqrt(code), as kernels.zig scales codes and senders

pub const Kind = enum { add, read_xn, read_rms, code, message, write, senders, base, table };

fn bf(x: f32) f32 {
    return c.fromBf16(c.toBf16(x));
}

fn sigmoid(x: f64) f64 {
    return 1 / (1 + @exp(-x));
}

/// A sum in float64, or as `acc` adds in `order` (gate 4's simulation).
fn sum(acc: ?c.Accum, order: Order, t: []const f64) f64 {
    const a = acc orelse {
        var s: f64 = 0;
        for (t) |v| s += v;
        return s;
    };
    return contract.rows.sumIn(a, order, t);
}

/// One prepared call's host inputs (bf16 bits unless named otherwise) and the outputs' extra allowance.
pub const Fixture = struct {
    kind: Kind,
    rows: usize,
    x: []u16 = &.{}, // add A, read/write RAW, code P, base RAW (rows + 1)
    y: []u16 = &.{}, // add B, write DELTA
    w: []u16 = &.{}, // write W, base NORM
    cb: []u16 = &.{}, // codebook [vocab, code]
    ids: []u32 = &.{}, // code's anchor ids
    packed_rows: []u16 = &.{}, // message/table [Q|S|V] rows
    gb: []u16 = &.{}, // gate bias [rank]
    rms: []f32 = &.{}, // write and table RMS
    top: []u32 = &.{}, // bf16_topk rows
    hw: []u32 = &.{}, // head words [vocab, hidden / 8]
    hs: []u16 = &.{}, // head scales [vocab, hidden / 64]
    hb: []u16 = &.{}, // head biases
    tg: []u16 = &.{}, // TG [vocab, rank]
    tsv: []u16 = &.{}, // senders' [G|V] rows [rows * k, 2 rank]
    base: []f32 = &.{}, // table's base [rows, k]
    anchor: u32 = 0,
    extra: []f64 = &.{}, // base: what a fused dequant may move each output by

    fn token(f: *const Fixture, row: usize, j: usize) u32 {
        return f.top[row * row_words + 2 + 2 * j];
    }

    pub fn count(f: *const Fixture) usize {
        const r = f.rows;
        return switch (f.kind) {
            .add, .read_xn => r * hidden,
            .read_rms => r,
            .code => r * code,
            .message => r * rank,
            .write => 2 * r * hidden,
            .senders => r * topk * code,
            .base => r * topk,
            .table => r * topk * topk,
        };
    }

    /// Outputs in float64 before their last rounding, sums in `acc` (null: exact), and their terms' magnitudes.
    pub fn compute(f: *Fixture, gpa: std.mem.Allocator, acc: ?c.Accum, order: Order, want: []f64, mag: []f64) !void {
        const r = f.rows;
        const terms = try gpa.alloc(f64, hidden);
        defer gpa.free(terms);
        @memset(mag, 0);
        switch (f.kind) {
            .add => for (want, f.x, f.y) |*o, a, b| {
                o.* = c.fromBf16(a) + c.fromBf16(b); // the kernel's fp32 add, exact here
            },
            .read_xn, .read_rms => for (0..r) |row| {
                const raw = f.x[row * hidden ..][0..hidden];
                for (terms, raw) |*t, v| t.* = @as(f64, c.fromBf16(v)) * c.fromBf16(v);
                const rms = @sqrt(@max(sum(acc, order, terms) / hidden, 1.17549435e-38));
                if (f.kind == .read_rms) {
                    want[row] = rms;
                    mag[row] = rms;
                } else for (want[row * hidden ..][0..hidden], mag[row * hidden ..][0..hidden], raw) |*o, *m, v| {
                    o.* = c.fromBf16(v) / (rms + eps);
                    m.* = @abs(o.*);
                }
            },
            .code => for (0..r) |row| {
                const out = want[row * code ..][0..code];
                if (row % block == 0) {
                    for (out, f.cb[f.ids[row] * code ..][0..code]) |*o, v| o.* = scale * c.fromBf16(v);
                    continue;
                }
                const p = f.x[row * code ..][0..code];
                for (terms[0..code], p) |*t, v| t.* = @as(f64, c.fromBf16(v)) * c.fromBf16(v);
                const n = @sqrt(sum(acc, order, terms[0..code])) + eps;
                for (out, mag[row * code ..][0..code], p) |*o, *m, v| {
                    o.* = scale * c.fromBf16(v) / n;
                    m.* = @abs(o.*);
                }
            },
            .message => for (0..r) |t| for (0..rank) |i| {
                if (t % block == 0) {
                    want[t * rank + i] = 0;
                    continue;
                }
                const q = f.packed_rows[t * 3 * rank ..];
                const prev = f.packed_rows[(t - 1) * 3 * rank ..];
                const x = c.fromBf16(q[i]) + c.fromBf16(prev[rank + i]) + c.fromBf16(f.gb[i]);
                want[t * rank + i] = sigmoid(x) * c.fromBf16(prev[2 * rank + i]);
            },
            .write => {
                const kept = try gpa.alloc(f32, hidden);
                defer gpa.free(kept);
                for (0..r) |row| {
                    const anchor = row % block == 0;
                    const res = want[row * hidden ..][0..hidden];
                    for (res, terms, kept, f.x[row * hidden ..][0..hidden], f.y[row * hidden ..][0..hidden]) |*o, *t, *k, raw, d| {
                        const a = c.fromBf16(raw);
                        o.* = if (anchor) a else a + @as(f64, f.rms[row]) * c.fromBf16(d);
                        k.* = if (anchor) a else bf(a + f.rms[row] * c.fromBf16(d)); // the stored RES the norm reads
                        t.* = @as(f64, k.*) * k.*;
                    }
                    const inv = 1 / @sqrt(sum(acc, order, terms) / hidden + eps);
                    for (want[(r + row) * hidden ..][0..hidden], mag[(r + row) * hidden ..][0..hidden], kept, f.w) |*o, *m, k, w| {
                        o.* = @as(f64, bf(@floatCast(k * inv))) * c.fromBf16(w);
                        m.* = @abs(o.*);
                    }
                }
            },
            .senders => for (0..r) |row| for (0..topk) |j| {
                const sid = if (row == 0) f.anchor else f.token(row - 1, j);
                for (want[(row * topk + j) * code ..][0..code], f.cb[sid * code ..][0..code]) |*o, v| o.* = scale * c.fromBf16(v);
            },
            .base => for (0..r) |i| for (0..topk) |j| {
                const id = f.token(i, j);
                var diff: f64 = 0;
                for (terms, 0..) |*t, h| {
                    const p = c.fromBf16(f.x[(i + 1) * hidden + h]) * c.fromBf16(f.w[h]);
                    const q: f32 = @floatFromInt((f.hw[id * (hidden / 8) + h / 8] >> @intCast(4 * (h % 8))) & 15);
                    const s = c.fromBf16(f.hs[id * (hidden / 64) + h / 64]);
                    const b = c.fromBf16(f.hb[id * (hidden / 64) + h / 64]);
                    const head = bf(s * q + b);
                    t.* = @as(f64, p) * head;
                    diff += @abs(@as(f64, p) * (bf(@mulAdd(f32, s, q, b)) - head));
                }
                want[i * topk + j] = sum(acc, order, terms);
                var m: f64 = 0;
                for (terms) |t| m += @abs(t);
                mag[i * topk + j] = m;
                f.extra[i * topk + j] = diff;
            },
            .table => for (0..r) |i| for (0..topk) |s| {
                const q = f.packed_rows[(i + 1) * 3 * rank ..];
                const sv = f.tsv[(i * topk + s) * 2 * rank ..];
                var m: [rank]f64 = undefined;
                for (&m, 0..) |*v, x| v.* = sigmoid(c.fromBf16(q[x]) + c.fromBf16(sv[x]) + c.fromBf16(f.gb[x])) * c.fromBf16(sv[rank + x]);
                const rms: f64 = f.rms[i + 1];
                for (0..topk) |j| {
                    const row = f.tg[f.token(i, j) * rank ..][0..rank];
                    var a: f64 = 0;
                    for (terms[0..rank], m, row) |*t, v, g| {
                        t.* = v * c.fromBf16(g);
                        a += @abs(t.*);
                    }
                    const at = (i * topk + s) * topk + j;
                    const b: f64 = f.base[i * topk + j];
                    want[at] = b + rms * sum(acc, order, terms[0..rank]);
                    mag[at] = @abs(b) + @abs(rms) * a;
                }
            },
        }
    }
};

/// Values a gate's fill asks for: random ones, or the stress rows' 64 to 95 (fp16 sums overflow).
const Gen = struct {
    rng: std.Random,
    fill: Fill,
    fn val(g: Gen, lo: f32, hi: f32, i: usize) u16 {
        if (g.fill == .stress) return c.toBf16(@floatFromInt(64 + i % 32));
        return c.toBf16(lo + (hi - lo) * g.rng.float(f32));
    }
    fn bf16s(g: Gen, a: std.mem.Allocator, n: usize, lo: f32, hi: f32) ![]u16 {
        const out = try a.alloc(u16, n);
        for (out, 0..) |*v, i| v.* = g.val(lo, hi, i);
        return out;
    }
    fn flat(a: std.mem.Allocator, n: usize, v: f32) ![]u16 {
        const out = try a.alloc(u16, n);
        @memset(out, c.toBf16(v));
        return out;
    }
};

const nan = std.math.nan(f32);
const inf = std.math.inf(f32);

/// A call's inputs for `rows` rows of `kind`, from `seed`, as `fill` asks; `a` owns every slice.
pub fn make(a: std.mem.Allocator, kind: Kind, rows: usize, fill: Fill, seed: u64) !Fixture {
    var prng = std.Random.DefaultPrng.init(seed);
    const g = Gen{ .rng = prng.random(), .fill = fill };
    var f = Fixture{ .kind = kind, .rows = rows };
    const stress = fill == .stress;
    const planted = fill == .nonfinite;
    // the head kinds read rows 1.. of their block (raw_first, rms_first) and a top-k row each
    const head_rows = rows + 1;
    f.top = try a.alloc(u32, rows * row_words);
    for (0..rows) |row| {
        const words = f.top[row * row_words ..][0..row_words];
        @memset(words, 0);
        words[1] = topk;
        for (0..topk) |j| {
            words[2 + 2 * j] = g.rng.uintLessThan(u32, vocab);
            words[3 + 2 * j] = @bitCast(@as(f32, @floatFromInt(topk - j)));
        }
    }
    f.extra = try a.alloc(f64, f.count());
    @memset(f.extra, 0);
    switch (kind) {
        .add => {
            f.x = try g.bf16s(a, rows * hidden, -4, 4);
            f.y = try g.bf16s(a, rows * hidden, -4, 4);
            if (planted) {
                f.x[3] = c.toBf16(nan);
                f.x[7] = c.toBf16(inf);
                f.y[11] = c.toBf16(-inf);
            }
        },
        .read_xn, .read_rms => {
            f.x = try g.bf16s(a, rows * hidden, -3, 3);
            if (planted) {
                f.x[1 * hidden + 5] = c.toBf16(inf);
                f.x[3 * hidden] = c.toBf16(nan);
            }
        },
        .code => {
            f.x = try g.bf16s(a, rows * code, -2, 2);
            f.cb = try g.bf16s(a, vocab * code, -1, 1);
            f.ids = try a.alloc(u32, rows);
            for (f.ids) |*id| id.* = g.rng.uintLessThan(u32, vocab);
            if (planted) {
                f.x[2 * code + 7] = c.toBf16(nan);
                f.x[3 * code + 1] = c.toBf16(inf);
                f.cb[f.ids[0] * code + 4] = c.toBf16(-inf);
            }
        },
        .message => {
            f.packed_rows = try g.bf16s(a, rows * 3 * rank, -3, 3);
            f.gb = try g.bf16s(a, rank, -1, 1);
            if (planted) {
                f.packed_rows[2 * 3 * rank + 3] = c.toBf16(nan);
                f.packed_rows[4 * 3 * rank + 2 * rank + 6] = c.toBf16(inf);
                f.packed_rows[5 * 3 * rank + rank + 10] = c.toBf16(inf);
                f.gb[9] = c.toBf16(-inf);
            }
        },
        .write => {
            f.x = try g.bf16s(a, rows * hidden, -3, 3);
            f.y = if (stress) try Gen.flat(a, rows * hidden, 0) else try g.bf16s(a, rows * hidden, -1, 1);
            f.w = if (stress) try Gen.flat(a, hidden, 1) else try g.bf16s(a, hidden, 0.5, 1.5);
            f.rms = try a.alloc(f32, rows);
            for (f.rms) |*v| v.* = if (stress) 1 else 0.25 + g.rng.float(f32);
            if (planted) {
                f.x[2 * hidden + 3] = c.toBf16(inf);
                f.rms[4] = nan;
                f.y[5 * hidden + 1] = c.toBf16(-inf);
            }
        },
        .senders => {
            f.cb = try g.bf16s(a, vocab * code, -1, 1);
            f.anchor = g.rng.uintLessThan(u32, vocab);
            if (planted) {
                f.cb[f.anchor * code + 2] = c.toBf16(nan);
                f.cb[f.token(0, 1) * code + 5] = c.toBf16(inf);
            }
        },
        .base => {
            f.x = try g.bf16s(a, head_rows * hidden, -3, 3);
            f.w = if (stress) try Gen.flat(a, hidden, 1) else try g.bf16s(a, hidden, 0.5, 1.5);
            f.hw = try a.alloc(u32, vocab * (hidden / 8));
            for (f.hw) |*v| v.* = g.rng.int(u32);
            f.hs = if (stress) try Gen.flat(a, vocab * (hidden / 64), 4) else try g.bf16s(a, vocab * (hidden / 64), 0.005, 0.05);
            f.hb = if (stress) try Gen.flat(a, vocab * (hidden / 64), 64) else try g.bf16s(a, vocab * (hidden / 64), -0.3, 0.3);
            if (planted) {
                f.x[2 * hidden + 7] = c.toBf16(nan);
                f.hs[f.token(3, 0) * (hidden / 64)] = c.toBf16(inf);
            }
        },
        .table => {
            f.packed_rows = if (stress) try Gen.flat(a, head_rows * 3 * rank, 16) else try g.bf16s(a, head_rows * 3 * rank, -3, 3);
            f.tsv = try a.alloc(u16, rows * topk * 2 * rank);
            for (0..rows * topk) |e| for (0..2 * rank) |x| {
                f.tsv[e * 2 * rank + x] = if (stress and x < rank) c.toBf16(16) else g.val(-3, 3, x);
            };
            f.gb = if (stress) try Gen.flat(a, rank, 16) else try g.bf16s(a, rank, -1, 1);
            f.tg = try g.bf16s(a, vocab * rank, -0.5, 0.5);
            f.base = try a.alloc(f32, rows * topk);
            for (f.base) |*v| v.* = if (stress) 0 else 8 * g.rng.float(f32) - 4;
            f.rms = try a.alloc(f32, head_rows);
            for (f.rms) |*v| v.* = if (stress) 1 else 0.25 + g.rng.float(f32);
            if (planted) {
                f.packed_rows[3 * 3 * rank + 4] = c.toBf16(nan);
                f.tg[f.token(4, 2) * rank + 7] = c.toBf16(inf);
                f.base[5 * topk + 3] = nan;
            }
        },
    }
    return f;
}

test "DSpine contract fixtures: each kind's float64 outputs are finite on random rows and planted ones mark some" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    for (std.enums.values(Kind)) |kind| {
        const rows: usize = if (kind == .add or kind == .message) 16 else 8;
        for ([_]Fill{ .random, .nonfinite }) |fill| {
            var f = try make(a, kind, rows, fill, 7);
            const want = try a.alloc(f64, f.count());
            const mag = try a.alloc(f64, f.count());
            try f.compute(a, null, .sequential, want, mag);
            var bad: usize = 0;
            for (want) |v| bad += @intFromBool(!std.math.isFinite(v));
            if (fill == .random) try std.testing.expectEqual(@as(usize, 0), bad) else try std.testing.expect(bad > 0 and bad < want.len);
        }
    }
}
