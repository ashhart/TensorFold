//! The LM schedule's passes on a computing host binding: a decode step's logits and a prompt split in two passes.
const std = @import("std");
const cfg = @import("config.zig");
const ops = @import("ref_ops.zig");
const lm = @import("lm.zig");
const st = @import("schedule_test.zig");

const t: cfg.Text = .{ .hidden = 16, .layers = 2, .heads = 4, .kv_heads = 2, .head_dim = 8, .intermediate = 24, .vocab = 40, .eps = 1e-6, .theta = 1e6, .context = 64 };
const qd = t.heads * t.head_dim;
const kvd = t.kv_heads * t.head_dim;
const cap = 32;

/// Each op row by row in float64 at bf16 points, so a row's values never depend on the rows beside it in a pass.
const LmHost = struct {
    m: *st.Mem,
    gpa: std.mem.Allocator,
    ids: [cap]u32 = undefined,
    pos: usize = 0,
    rows: usize = 0,
    q: [cap * qd]f64 = undefined,
    kc: [t.layers * cap * kvd]f64 = @splat(std.math.nan(f64)),
    vc: [t.layers * cap * kvd]f64 = @splat(std.math.nan(f64)),
    cs: [cap * t.head_dim]f32 = undefined,
    sn: [cap * t.head_dim]f32 = undefined,
    logits: std.ArrayList(f64) = .empty,
    plant_slot: bool = false, // the planted fault: k and v written at the pass's row, not its position

    fn init(m: *st.Mem, gpa: std.mem.Allocator) LmHost {
        var h: LmHost = .{ .m = m, .gpa = gpa };
        lm.ropeTables(t, &h.cs, &h.sn, cap);
        return h;
    }

    fn deinit(h: *LmHost) void {
        h.logits.deinit(h.gpa);
    }

    pub fn positions(h: *LmHost, ids: []const u32, pos: usize) !void {
        if (pos + ids.len > cap) return error.ContextFull;
        @memcpy(h.ids[0..ids.len], ids);
        h.pos = pos;
        h.rows = ids.len;
    }
    pub fn embed(h: *LmHost, table: u64, out: u64, rows: usize, hidden: usize) !void {
        const tab = try h.m.at(table, t.vocab * hidden);
        const o = try h.m.at(out, rows * hidden);
        for (0..rows) |r| @memcpy(o[r * hidden ..][0..hidden], tab[h.ids[r] * hidden ..][0..hidden]);
    }
    pub fn copy(h: *LmHost, dst: u64, src: u64, n: usize) !void {
        @memcpy(try h.m.at(dst, n / 2), try h.m.at(src, n / 2));
    }
    fn rms(x: []const f64, w: []const f64, y: []f64) void {
        var ss: f64 = 0;
        for (x) |v| ss += v * v;
        const inv = 1 / @sqrt(ss / @as(f64, @floatFromInt(x.len)) + t.eps);
        for (y, x, w) |*o, v, g| o.* = ops.bf16(g * ops.bf16(v * inv));
    }
    pub fn addNorm(h: *LmHost, x: u64, res: ?u64, w: u64, hb: u64, y: u64, rows: usize) !void {
        const xs = try h.m.at(x, rows * t.hidden);
        const ws = try h.m.at(w, t.hidden);
        const ys = try h.m.at(y, rows * t.hidden);
        const hs = if (res != null) try h.m.at(hb, rows * t.hidden) else xs;
        if (res) |p| for (hs, xs, try h.m.at(p, rows * t.hidden)) |*o, a, b| {
            o.* = ops.bf16(a + b);
        };
        for (0..rows) |r| rms(hs[r * t.hidden ..][0..t.hidden], ws, ys[r * t.hidden ..][0..t.hidden]);
    }
    pub fn gemm(h: *LmHost, x: u64, w: u64, y: u64, rows: usize, n: usize, k: usize) !void {
        const tmp = try h.gpa.alloc(f64, rows * n);
        defer h.gpa.free(tmp);
        ops.linear(tmp, try h.m.at(x, rows * k), try h.m.at(w, n * k), null, rows, n, k);
        for (try h.m.at(y, rows * n), tmp) |*o, v| o.* = ops.bf16(v);
    }
    fn rope(h: *const LmHost, v: []f64, p: usize) void {
        const half = t.head_dim / 2;
        var tmp: [t.head_dim]f64 = undefined;
        for (0..t.head_dim) |i| {
            const other = if (i < half) -v[i + half] else v[i - half];
            tmp[i] = ops.bf16(v[i] * h.cs[p * t.head_dim + i] + other * h.sn[p * t.head_dim + i]);
        }
        @memcpy(v, &tmp);
    }
    pub fn qkRope(h: *LmHost, qkv: u64, qn: u64, kn: u64, layer: usize, rows: usize) !void {
        const xs = try h.m.at(qkv, rows * (qd + 2 * kvd));
        const qw = try h.m.at(qn, t.head_dim);
        const kw = try h.m.at(kn, t.head_dim);
        for (0..rows) |r| {
            const p = h.pos + r;
            const row = xs[r * (qd + 2 * kvd) ..];
            for (0..t.heads) |i| {
                const q = h.q[(r * t.heads + i) * t.head_dim ..][0..t.head_dim];
                rms(row[i * t.head_dim ..][0..t.head_dim], qw, q);
                h.rope(q, p);
            }
            const slot = layer * cap + (if (h.plant_slot) r else p);
            for (0..t.kv_heads) |i| {
                const k = h.kc[slot * kvd + i * t.head_dim ..][0..t.head_dim];
                rms(row[qd + i * t.head_dim ..][0..t.head_dim], kw, k);
                h.rope(k, p);
            }
            @memcpy(h.vc[slot * kvd ..][0..kvd], row[qd + kvd ..][0..kvd]);
        }
    }
    pub fn attention(h: *LmHost, layer: usize, out: u64, rows: usize) !void {
        const o = try h.m.at(out, rows * qd);
        const scale = 1 / @sqrt(@as(f64, @floatFromInt(t.head_dim)));
        var s: [cap]f64 = undefined;
        for (0..rows) |r| for (0..t.heads) |i| {
            const p = h.pos + r;
            const g = i / (t.heads / t.kv_heads);
            const q = h.q[(r * t.heads + i) * t.head_dim ..][0..t.head_dim];
            var top: f64 = -std.math.inf(f64);
            for (0..p + 1) |j| {
                const k = h.kc[(layer * cap + j) * kvd + g * t.head_dim ..][0..t.head_dim];
                var d: f64 = 0;
                for (q, k) |a, b| d += a * b;
                s[j] = d * scale;
                top = @max(top, s[j]);
            }
            var sum: f64 = 0;
            for (s[0 .. p + 1]) |*v| {
                v.* = @exp(v.* - top);
                sum += v.*;
            }
            for (0..t.head_dim) |c| {
                var acc: f64 = 0;
                for (0..p + 1) |j| acc += s[j] * h.vc[(layer * cap + j) * kvd + g * t.head_dim + c];
                o[r * qd + i * t.head_dim + c] = ops.bf16(acc / sum);
            }
        };
    }
    pub fn swiglu(h: *LmHost, gu: u64, out: u64, rows: usize, i: usize) !void {
        const x = try h.m.at(gu, rows * 2 * i);
        const o = try h.m.at(out, rows * i);
        for (0..rows) |r| for (0..i) |c| {
            const g = x[r * 2 * i + c];
            o[r * i + c] = ops.bf16(ops.bf16(g / (1 + @exp(-g))) * x[r * 2 * i + i + c]);
        };
    }
    pub fn dump(_: *LmHost, _: []const u8, _: u64, _: usize, _: usize) !void {}
    pub fn greedy(h: *LmHost, row: u64, table: u64) !u32 {
        const x = try h.m.at(row, t.hidden);
        const tab = try h.m.at(table, t.vocab * t.hidden);
        var out: [t.vocab]f64 = undefined;
        var best: u32 = 0;
        for (&out, 0..) |*o, j| {
            var d: f64 = 0;
            for (x, tab[j * t.hidden ..][0..t.hidden]) |a, b| d += a * b;
            o.* = ops.bf16(d);
            if (o.* > out[best]) best = @intCast(j);
        }
        try h.logits.appendSlice(h.gpa, &out);
        return best;
    }
};

const Fixture = struct {
    mem: st.Mem,
    fake: st.Fake,
    w: lm.Weights = undefined,
    b: lm.Buffers = undefined,
    audio: lm.Splice = undefined,

    fn init(f: *Fixture) !void {
        const gpa = std.testing.allocator;
        f.* = .{ .mem = .{ .gpa = gpa }, .fake = .{ .gpa = gpa } };
        const head = try f.fake.expect("thinker.model.embed_tokens.weight", &.{ t.vocab, t.hidden });
        try f.fake.map.put(gpa, try gpa.dupe(u8, "thinker.lm_head.weight"), try gpa.dupe(u8, head));
        f.w = try lm.Weights.load(gpa, t, &f.fake, &f.mem);
        f.b = try lm.Buffers.alloc(t, 16, &f.mem);
        const rows = try f.mem.alloc(2 * 4 * t.hidden); // four audio rows at positions [3, 7)
        var rng: std.Random.DefaultPrng = .init(11);
        for (try f.mem.at(rows, 4 * t.hidden)) |*v| v.* = ops.bf16(rng.random().floatNorm(f64));
        f.audio = .{ .ptr = rows, .at = 3, .n = 4 };
    }

    fn deinit(f: *Fixture) void {
        f.w.deinit(std.testing.allocator);
        f.fake.deinit();
        f.mem.deinit();
    }
};

const prompt = [_]u32{ 1, 2, 3, 9, 9, 9, 9, 4, 5, 6, 7, 8 };

/// Rows 7..11's logits by decode steps after a prompt pass over prompt[0..8], as tf-qwen3-asr runs a reply.
fn decoded(f: *Fixture, plant: bool) ![]f64 {
    var h: LmHost = .init(&f.mem, std.testing.allocator);
    defer h.deinit();
    h.plant_slot = plant;
    _ = try lm.forward(&h, t, &f.w, f.b, prompt[0..8], 0, f.audio);
    for (8..prompt.len) |p| _ = try lm.forward(&h, t, &f.w, f.b, prompt[p..][0..1], p, null);
    return h.logits.toOwnedSlice(std.testing.allocator);
}

test "lm schedule: each decode step's logits equal the prompt pass's for the same rows" {
    var f: Fixture = undefined;
    try f.init();
    defer f.deinit();
    var h: LmHost = .init(&f.mem, std.testing.allocator);
    defer h.deinit();
    _ = try lm.forwardHeads(&h, t, &f.w, f.b, &prompt, 0, f.audio, 5);
    const steps = try decoded(&f, false);
    defer std.testing.allocator.free(steps);
    try std.testing.expectEqualSlices(f64, h.logits.items, steps);
    // the planted fault (k and v at the pass's row) moves a decode step: the comparison is not vacuous
    const planted = try decoded(&f, true);
    defer std.testing.allocator.free(planted);
    try std.testing.expect(!std.mem.eql(f64, h.logits.items, planted));
}

test "lm schedule: a prompt in two passes, cut inside the audio rows, leaves the one pass's caches and logits" {
    var f: Fixture = undefined;
    try f.init();
    defer f.deinit();
    var one: LmHost = .init(&f.mem, std.testing.allocator);
    defer one.deinit();
    _ = try lm.forward(&one, t, &f.w, f.b, &prompt, 0, f.audio);
    for ([_]usize{ 5, 3, 7, 11 }) |cut| {
        var two: LmHost = .init(&f.mem, std.testing.allocator);
        defer two.deinit();
        _ = try lm.forward(&two, t, &f.w, f.b, prompt[0..cut], 0, f.audio);
        two.logits.clearRetainingCapacity();
        _ = try lm.forward(&two, t, &f.w, f.b, prompt[cut..], cut, f.audio);
        for (0..t.layers) |l| {
            const at = l * cap * kvd;
            try std.testing.expectEqualSlices(f64, one.kc[at..][0 .. prompt.len * kvd], two.kc[at..][0 .. prompt.len * kvd]);
            try std.testing.expectEqualSlices(f64, one.vc[at..][0 .. prompt.len * kvd], two.vc[at..][0 .. prompt.len * kvd]);
        }
        try std.testing.expectEqualSlices(f64, one.logits.items, two.logits.items);
    }
}
