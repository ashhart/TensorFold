//! MiniMax-M3.1-Flash's prompt kernels on the tensor units (index_nax.metal, attention_nax.metal), as model.zig builds.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const contract = @import("contract");
const ar = contract.attention_range;
const sources = @import("kernel_sources");
const rig_ = @import("rig.zig");
const nax = @import("metal_minimax_nax.zig");

const c = contract.spec;
const g = contract.gates;
const cfg = nax.config;
const st = tf.minimax_m3.state;
const dh: usize = cfg.head_dim;
const block: usize = cfg.block;
const words = st.selWords(&cfg);
const scale: f64 = 1.0 / @sqrt(@as(f64, @floatFromInt(cfg.head_dim)));
const nan16 = 0x7fc0;

/// model.zig's header for attention_nax.metal.
pub fn promptHead(a: std.mem.Allocator) ![]u8 {
    return std.fmt.allocPrint(a, "#define DH {d}\n#define NQ {d}\n#define G {d}\n#define KVH {d}\n#define WORDS {d}\n#define SCALE {e}f\n", .{ cfg.head_dim, cfg.heads, cfg.heads / cfg.kv_heads, cfg.kv_heads, words, scale });
}

/// model.zig's header for index_nax.metal.
pub fn indexHead(a: std.mem.Allocator) ![]u8 {
    return std.fmt.allocPrint(a, "#define DH {d}\n#define IH {d}\n", .{ cfg.index_dim, cfg.index_heads });
}

/// A source as the engine builds it on this GPU: defines, then frags.source's text.
fn pipeline(a: std.mem.Allocator, device: mtl.Device, head: []const u8, text: []const u8, name: []const u8) !mtl.Pipeline {
    const body = try tf.frags.source(device, a, text);
    defer a.free(body);
    const src = try std.mem.concat(a, u8, &.{ head, body });
    defer a.free(src);
    return rig_.pipeline(device, src, name);
}

fn bf(v: f64) f64 {
    return c.fromBf16(c.toBf16(@floatCast(v)));
}

/// Random bf16 values in [-r, r).
fn fillRandom(rng: std.Random, out: []u16, r: f32) void {
    for (out) |*v| v.* = c.toBf16((rng.float(f32) * 2 - 1) * r);
}

/// An mma k16 step's error in 2^-24 of |acc| + sum |a b|.
const mma_step = ar.mma_step;

// --- mm_index_scores_nax: S[row, h, b] = max over b's keys up to the row of IQ[row, h] . IK[key] ----------------

/// The score's dot: DH / 16 mma steps of 16 products, each step's error against the running sum.
fn indexDepth(_: c.Shape) usize {
    return 2 * mma_step * (cfg.index_dim / 16);
}

pub const Index = struct {
    const ih: usize = cfg.index_heads;
    const di: usize = cfg.index_dim;
    const rt = 32; // a tile's query rows
    a: std.mem.Allocator,
    rig: rig_.Rig,
    pipe: mtl.Pipeline,
    rows: usize = 0,
    p0: usize = 0,
    q: []u16 = &.{},
    k: []u16 = &.{},
    /// Each kept output's index in S: blocks up to the tile's last row's (later blocks stay unwritten).
    kept: std.ArrayList(u32) = .empty,
    shown: std.ArrayList(u8) = .empty,
    refs: [3]rig_.Ref = undefined,
    out: rig_.Ref = undefined,

    /// An accumulator and order gate 4 replays the dots in.
    const Sim = struct { a: c.Accum, o: g.Order };

    /// Rows: a prompt chunk's rows; n: the index dim; k: the chunk's first position.
    pub fn spec() c.Spec {
        return .{
            .name = "metal minimax_m3 mm_index_scores_nax index heads 4 dim 128",
            .accum = .f32,
            .depth = indexDepth,
            .out = .f32,
            .tiles = .{ .{ .size = rt }, .{}, .{ .size = block, .step = 1, .min = 1 } },
            // a chunk past the 16 blocks a row keeps, a whole 4,096-row chunk from the start, one deep in the window
            .served = &.{ .{ .rows = 33, .n = di, .k = 2048 }, .{ .rows = st.prompt_rows, .n = di, .k = 0 }, .{ .rows = 512, .n = di, .k = 30720 } },
        };
    }

    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue) !Index {
        const head = try indexHead(a);
        defer a.free(head);
        return .{ .a = a, .rig = .{ .a = a, .device = device, .queue = queue }, .pipe = try pipeline(a, device, head, sources.minimax_m3_index_nax, "mm_index_scores_nax") };
    }
    pub fn deinit(t: *Index) void {
        t.a.free(t.q);
        t.a.free(t.k);
        t.kept.deinit(t.a);
        t.shown.deinit(t.a);
        t.rig.deinit();
        t.pipe.deinit();
    }
    pub fn kernel(t: *Index) g.Kernel {
        return .{ .spec = spec(), .ctx = t, .vt = &vt };
    }
    const vt = blk: {
        var h = g.hooks(Index);
        h.output = output;
        break :blk h;
    };

    fn blocks(t: *const Index) usize {
        return (t.p0 + t.rows + block - 1) / block;
    }

    pub fn prepare(t: *Index, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.n != di) return false;
        t.rig.reset();
        t.a.free(t.q);
        t.a.free(t.k);
        t.rows = shape.rows;
        t.p0 = shape.k;
        const pad = std.mem.alignForward(usize, t.rows, rt); // a tile reads all its 32 rows
        const slots = std.mem.alignForward(usize, t.p0 + t.rows, block); // whole index blocks, as Cache.slots
        t.q = try t.a.alloc(u16, pad * ih * di);
        t.k = try t.a.alloc(u16, slots * di);
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        fillRandom(rng, t.q, 2);
        fillRandom(rng, t.k, 2);
        // rows past the chunk and keys past its last position hold anything: NaN here, which no kept output may see
        @memset(t.q[t.rows * ih * di ..], nan16);
        @memset(t.k[(t.p0 + t.rows) * di ..], nan16);
        switch (fill) {
            .random => {},
            .nonfinite => {
                const at = [_]usize{ (0 * ih + 0) * di + 3, (1 * ih + 1) * di, (2 * ih + 2) * di + 5 };
                for (at, [_]f32{ std.math.nan(f32), std.math.inf(f32), -std.math.inf(f32) }) |i, v| if (i < t.rows * ih * di) {
                    t.q[i] = c.toBf16(v);
                };
                t.k[(t.p0 + 2) * di + 1] = c.toBf16(std.math.nan(f32));
            },
            .stress => {
                // integer dots exact in fp32 in any order; a key's halves cancel but for 8 nudged queries: small scores
                const h = di / 2;
                for (0..t.rows * ih) |r| for (0..h) |d| {
                    const x: f32 = @floatFromInt(1 + rng.uintLessThan(u32, 255));
                    t.q[r * di + d] = c.toBf16(x);
                    t.q[r * di + h + d] = c.toBf16(x + @as(f32, @floatFromInt(@intFromBool(d < 8))));
                };
                for (0..t.p0 + t.rows) |key| for (0..h) |d| {
                    const y: f32 = @floatFromInt(1 + rng.uintLessThan(u32, 255));
                    t.k[key * di + d] = c.toBf16(y);
                    t.k[key * di + h + d] = c.toBf16(-y);
                };
            },
        }
        t.kept.clearRetainingCapacity();
        for (0..t.rows) |r| {
            const tile_last = @min((r / rt) * rt + rt - 1, t.rows - 1);
            const last_block = (t.p0 + tile_last) / block;
            for (0..ih) |hh| for (0..last_block + 1) |b| try t.kept.append(t.a, @intCast((r * ih + hh) * t.blocks() + b));
        }
        t.refs[0] = try t.rig.input(std.mem.sliceAsBytes(t.q));
        t.refs[1] = try t.rig.input(std.mem.sliceAsBytes(t.k));
        t.refs[2] = try t.rig.input(std.mem.asBytes(&[3]i32{ @intCast(t.rows), @intCast(t.p0), @intCast(t.blocks()) }));
        t.out = try t.rig.output(t.rows * ih * t.blocks() * 4);
        return true;
    }

    pub fn run(t: *Index) !void {
        const p = t.rig.begin();
        p.e.setPipeline(t.pipe);
        for (t.refs[0..2], 0..) |r, i| p.e.setBuffer(r.buf, r.off, i);
        p.e.setBuffer(t.out.buf, t.out.off, 2);
        p.e.setBuffer(t.refs[2].buf, t.refs[2].off, 3);
        // as forward.zig: (rows / 32, index heads) groups, 32 threads a 64-column simdgroup
        p.e.dispatchGroups(mtl.Size.of((t.rows + rt - 1) / rt, ih, 1), mtl.Size.of(di / 64 * 32, 1, 1));
        try t.rig.finish(p);
    }

    /// The written scores only: each row's blocks up to its tile's last.
    fn output(ctx: *anyopaque) anyerror![]const u8 {
        const t: *Index = @ptrCast(@alignCast(ctx));
        const all = try t.rig.outputBytes();
        t.shown.clearRetainingCapacity();
        for (t.kept.items) |i| try t.shown.appendSlice(t.a, all[i * 4 ..][0..4]);
        return t.shown.items;
    }

    pub fn count(t: *Index) usize {
        return t.kept.items.len;
    }

    fn dot(t: *const Index, r: usize, hh: usize, key: usize, acc: ?Sim) [2]f64 {
        var terms: [di]f64 = undefined;
        var mag: f64 = 0;
        var sum: f64 = 0;
        for (&terms, 0..) |*x, d| {
            x.* = @as(f64, c.fromBf16(t.q[(r * ih + hh) * di + d])) * c.fromBf16(t.k[key * di + d]);
            sum += x.*;
            mag += @abs(x.*);
        }
        if (acc) |s| sum = contract.rows.sumIn(s.a, s.o, &terms);
        return .{ sum, mag };
    }

    /// Kept output i's max (NaN if any dot is NaN, -Inf when the block holds no key up to the row) and its terms' size.
    fn one(t: *const Index, i: u32, acc: ?Sim) [2]f64 {
        const per = ih * t.blocks();
        const r = i / per;
        const hh = (i % per) / t.blocks();
        const b = i % t.blocks();
        const pos = t.p0 + r;
        var best = -std.math.inf(f64);
        var mag: f64 = 0;
        var key = b * block;
        while (key < (b + 1) * block and key <= pos) : (key += 1) {
            const s = t.dot(r, hh, key, acc);
            if (std.math.isNan(s[0]) or std.math.isNan(best)) best = std.math.nan(f64) else best = @max(best, s[0]);
            if (std.math.isFinite(s[1])) mag = @max(mag, s[1]);
        }
        return .{ best, mag };
    }

    pub fn reference(t: *Index, want: []f64, mag: []f64) !void {
        for (t.kept.items, want, mag) |i, *w, *m| {
            const o = t.one(i, null);
            w.* = o[0];
            m.* = o[1];
        }
    }

    pub fn simulate(t: *Index, acc: c.Accum, order: g.Order, out: []f64) !void {
        for (t.kept.items, out) |i, *o| o.* = t.one(i, Sim{ .a = acc, .o = order })[0];
    }
};

// --- mm_prompt_attention: softmax over the kept blocks' keys up to each row, bf16 out ----------------------------

/// The range's fp32 arithmetic: mma scores, fp32 SCALE * log2e, fast exp2, 64-key blocks; fp32 P cut as it meets V.
fn form(n: usize) ar.Form {
    const blocks: f64 = @floatFromInt(n / 64 + 2);
    return .{ .score = @as(f64, @floatFromInt(dh / 16)) * mma_step + 2, .fast_exp = true, .chunk_exp = false, .p_rel = 0, .p_cut = ar.operand_cut, .num = mma_step * 4 + blocks + 2, .den = blocks + 6, .score_round = 4, .nan_max = true };
}

/// Gate 4's keys: the first 1,024 score high (their probabilities exactly 1), every other key scores far below.
const stress_on = 1024;

pub const Prompt = struct {
    const nq: usize = cfg.heads;
    const kvh: usize = cfg.kv_heads;
    const grp: usize = cfg.heads / cfg.kv_heads;
    const per = 2; // positions a tile: 32 rows of 16 query heads
    a: std.mem.Allocator,
    rig: rig_.Rig,
    pipe: mtl.Pipeline,
    rows: usize = 0,
    p0: usize = 0,
    fill: g.Fill = .random,
    q: []u16 = &.{},
    k: []u16 = &.{},
    v: []u16 = &.{},
    sel: []u32 = &.{},
    point: []f64 = &.{},
    half: []f64 = &.{},
    down: []f64 = &.{},
    up: []f64 = &.{},
    refs: [5]rig_.Ref = undefined,
    out: rig_.Ref = undefined,

    /// Rows: a prompt chunk's rows; n: the head dim; k: the chunk's first position.
    pub fn spec() c.Spec {
        return .{
            .name = "metal minimax_m3 mm_prompt_attention 64 heads over 4, kept 128-key blocks",
            .accum = .f32,
            .depth = indexDepth,
            .out = .bf16,
            .tiles = .{ .{ .size = per }, .{}, .{ .size = block, .step = 1, .min = 1 } },
            // the shortest prompt chunk with selection, a chunk from the start, one deep in the window
            .served = &.{ .{ .rows = st.window_rows + 1, .n = dh, .k = 2100 }, .{ .rows = 256, .n = dh, .k = 0 }, .{ .rows = 64, .n = dh, .k = 32704 } },
            .sums = .{ .exact = true },
            // gate 1's Inf at key p0 + 1, which row 0 does not see but its tile reads, makes row 0's outputs NaN
            .known = .{ .gate = .nonfinite, .finding = "a non-finite value at a key masked for a row of its tile reaches that row as NaN (P = 0 times Inf inside the P x V mma)" },
        };
    }

    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue) !Prompt {
        const h = try promptHead(a);
        defer a.free(h);
        return .{ .a = a, .rig = .{ .a = a, .device = device, .queue = queue }, .pipe = try pipeline(a, device, h, sources.minimax_m3_attention_nax, "mm_prompt_attention") };
    }
    fn free(t: *Prompt) void {
        for ([_][]u16{ t.q, t.k, t.v }) |x| t.a.free(x);
        t.a.free(t.sel);
        for ([_][]f64{ t.point, t.half, t.down, t.up }) |x| t.a.free(x);
        t.q = &.{};
        t.k = &.{};
        t.v = &.{};
        t.sel = &.{};
        t.point = &.{};
        t.half = &.{};
        t.down = &.{};
        t.up = &.{};
    }
    pub fn deinit(t: *Prompt) void {
        t.free();
        t.rig.deinit();
        t.pipe.deinit();
    }
    pub fn kernel(t: *Prompt) g.Kernel {
        return .{ .spec = spec(), .ctx = t, .vt = &vt };
    }
    const vt = g.hooks(Prompt);

    fn keptBlock(t: *const Prompt, r: usize, kv: usize, b: usize) bool {
        return (t.sel[(r * kvh + kv) * words + b / 32] >> @intCast(b % 32)) & 1 != 0;
    }

    /// index_select's bitmap: a row keeps its own block and up to 15 earlier ones (all of them while it sees 16).
    fn select(t: *Prompt, rng: std.Random) void {
        @memset(t.sel, 0);
        for (0..t.rows) |r| for (0..kvh) |kv| {
            const nb = (t.p0 + r) / block + 1;
            const at = t.sel[(r * kvh + kv) * words ..][0..words];
            const set = struct {
                fn f(s: []u32, b: usize) void {
                    s[b / 32] |= @as(u32, 1) << @intCast(b % 32);
                }
            }.f;
            set(at, nb - 1);
            if (t.fill == .stress) {
                for (0..@min(nb - 1, stress_on / block)) |b| set(at, b);
                continue;
            }
            if (nb <= cfg.topk_blocks) {
                for (0..nb) |b| set(at, b);
                continue;
            }
            var kept: usize = 1;
            while (kept < cfg.topk_blocks) {
                const b = rng.uintLessThan(usize, nb - 1);
                if ((at[b / 32] >> @intCast(b % 32)) & 1 != 0) continue;
                set(at, b);
                kept += 1;
            }
        };
    }

    pub fn prepare(t: *Prompt, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.n != dh) return false;
        if (fill == .stress and shape.k < stress_on) return false;
        t.rig.reset();
        t.free();
        t.rows = shape.rows;
        t.p0 = shape.k;
        t.fill = fill;
        const pad = std.mem.alignForward(usize, t.rows, per); // a tile's second row is read past an odd chunk
        const slots = std.mem.alignForward(usize, t.p0 + t.rows, block);
        const kw = kvh * dh;
        t.q = try t.a.alloc(u16, pad * nq * dh);
        t.k = try t.a.alloc(u16, slots * kw);
        t.v = try t.a.alloc(u16, slots * kw);
        t.sel = try t.a.alloc(u32, pad * kvh * words);
        for ([_]*[]f64{ &t.point, &t.half, &t.down, &t.up }) |x| x.* = try t.a.alloc(f64, t.count());
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        fillRandom(rng, t.q, 3);
        fillRandom(rng, t.k, 3);
        fillRandom(rng, t.v, 2);
        t.select(rng);
        switch (fill) {
            .random => {},
            .nonfinite => {
                // in the rows' own block, which every row keeps; keys at and after p0 + 2 are seen by rows 2 on
                for ([_]usize{ 3, 1 * dh, 2 * dh + 5 }, [_]f32{ std.math.nan(f32), std.math.inf(f32), -std.math.inf(f32) }, 0..) |at, x, r| if (r < t.rows) {
                    t.q[r * nq * dh + (r * grp + 1) * dh + at % dh] = c.toBf16(x);
                };
                t.v[t.p0 * kw + 0 * dh + 5] = c.toBf16(std.math.inf(f32));
                t.v[t.p0 * kw + 1 * dh + 6] = c.toBf16(std.math.nan(f32));
                t.k[(t.p0 + 2) * kw + 2 * dh] = c.toBf16(std.math.nan(f32));
                // row 0 does not see key p0 + 1, which its 2-row tile reads
                if (t.rows > 1) t.v[(t.p0 + 1) * kw + 3 * dh + 7] = c.toBf16(std.math.inf(f32));
            },
            .stress => {
                // keys of 3s or -3s: the first 1,024 score high, the rest probability 0; values cancel but for 8 nudged
                @memset(t.q, c.toBf16(3));
                for (0..t.p0 + t.rows) |key| @memset(t.k[key * kw ..][0..kw], c.toBf16(if (key < stress_on) 3 else -3));
                for (0..stress_on / 2) |key| for (0..kw) |e| {
                    const x: f32 = @floatFromInt(1 + rng.uintLessThan(u32, 255));
                    t.v[key * kw + e] = c.toBf16(x);
                    t.v[(key + stress_on / 2) * kw + e] = c.toBf16(-x + @as(f32, @floatFromInt(@intFromBool(key < 8))));
                };
            },
        }
        // cache rows past the chunk's last position hold anything: NaN here, which the kernel must never read in
        @memset(t.k[(t.p0 + t.rows) * kw ..], nan16);
        @memset(t.v[(t.p0 + t.rows) * kw ..], nan16);
        t.refs[0] = try t.rig.input(std.mem.sliceAsBytes(t.q));
        t.refs[1] = try t.rig.input(std.mem.sliceAsBytes(t.k));
        t.refs[2] = try t.rig.input(std.mem.sliceAsBytes(t.v));
        t.refs[3] = try t.rig.input(std.mem.asBytes(&[2]i32{ @intCast(t.rows), @intCast(t.p0) }));
        t.refs[4] = try t.rig.input(std.mem.sliceAsBytes(t.sel));
        t.out = try t.rig.output(t.count() * 2);
        return true;
    }

    pub fn run(t: *Prompt) !void {
        const p = t.rig.begin();
        p.e.setPipeline(t.pipe);
        for (t.refs[0..3], 0..) |r, i| p.e.setBuffer(r.buf, r.off, i);
        p.e.setBuffer(t.out.buf, t.out.off, 3);
        p.e.setBuffer(t.refs[3].buf, t.refs[3].off, 4);
        p.e.setBuffer(t.refs[4].buf, t.refs[4].off, 5);
        // as forward.zig: (rows / 2, key heads) groups, 32 threads a 64-column simdgroup
        p.e.dispatchGroups(mtl.Size.of((t.rows + per - 1) / per, kvh, 1), mtl.Size.of(dh / 64 * 32, 1, 1));
        try t.rig.finish(p);
    }

    pub fn count(t: *Prompt) usize {
        return t.rows * nq * dh;
    }

    /// Each row's key heads over their kept keys up to the row: float64's outputs and their ranges.
    pub fn reference(t: *Prompt, want: []f64, mag: []f64) !void {
        @memset(mag, 0);
        const most = (t.p0 + t.rows) * dh;
        const buf = try t.a.alloc(f64, 2 * most + 2 * (t.p0 + t.rows) + 2 * dh);
        defer t.a.free(buf);
        const ks = buf[0..most];
        const vs = buf[most .. 2 * most];
        const w = buf[2 * most ..][0 .. t.p0 + t.rows];
        const tau = buf[2 * most + t.p0 + t.rows ..][0 .. t.p0 + t.rows];
        const qv = buf[buf.len - 2 * dh ..][0..dh];
        const hf = buf[buf.len - dh ..];
        for (0..t.rows) |r| for (0..kvh) |kv| {
            var n: usize = 0;
            for (0..t.p0 + r + 1) |key| {
                if (!t.keptBlock(r, kv, key / block)) continue;
                for (0..dh) |e| {
                    ks[n * dh + e] = c.fromBf16(t.k[(key * kvh + kv) * dh + e]);
                    vs[n * dh + e] = c.fromBf16(t.v[(key * kvh + kv) * dh + e]);
                }
                n += 1;
            }
            for (0..grp) |gh| {
                const hq = kv * grp + gh;
                for (qv, 0..) |*x, e| x.* = c.fromBf16(t.q[(r * nq + hq) * dh + e]);
                const at = (r * nq + hq) * dh;
                const f = form(n);
                ar.head(qv, ks, vs, n, dh, scale, f, w, tau, want[at..][0..dh], hf);
                @memcpy(t.half[at..][0..dh], hf);
                ar.cutSides(f, w, tau, vs, n, dh, t.down[at..][0..dh], t.up[at..][0..dh]);
            }
        };
        @memcpy(t.point, want);
    }

    /// The range's half width and the cut's wider side plus bf16's rounding; gate 4 allows the rounding alone.
    pub fn bounds(t: *Prompt, want: []const f64, _: []const f64, allow: []f64) !void {
        for (allow, want, t.half, t.down, t.up) |*l, w, h, d, u| {
            if (t.fill == .stress) {
                l.* = c.boundExact(spec(), w);
            } else if (!std.math.isFinite(w)) {
                l.* = 0;
            } else if (!std.math.isFinite(h)) {
                l.* = std.math.floatMax(f64); // scores too far apart for the range's first order: any finite value
            } else l.* = h + @max(d, u) + 0x1p-8 * (@abs(w) + h + @max(d, u)) + 0x1p-133;
        }
    }

    /// Each bf16 output within its range's ends as rounded, the cut on its own side; gate 4 and wide ranges are open.
    pub fn intervals(t: *Prompt, low: []f64, high: []f64) !void {
        for (low, high, t.point, t.half, t.down, t.up) |*l, *hi, p, h, d, u| {
            const open = t.fill == .stress or !std.math.isFinite(p) or !std.math.isFinite(h);
            const e = if (open) [2]f64{ -std.math.floatMax(f64), std.math.floatMax(f64) } else ar.ends(p, h, d, u);
            l.* = e[0];
            hi.* = e[1];
        }
    }

    /// Gate 4: the outputs if P x V's sum of the 1,024 values ran in `acc`.
    pub fn simulate(t: *Prompt, acc: c.Accum, order: g.Order, out: []f64) !void {
        var terms: [stress_on]f64 = undefined;
        const kw = kvh * dh;
        for (0..t.rows) |r| for (0..nq) |hq| for (0..dh) |e| {
            const col = (hq / grp) * dh + e;
            for (&terms, 0..) |*x, key| x.* = c.fromBf16(t.v[key * kw + col]);
            out[(r * nq + hq) * dh + e] = bf(contract.rows.sumIn(acc, order, &terms) / stress_on);
        };
    }
};
