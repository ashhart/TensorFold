//! Ripple's attention under the contract: rc_decode_vec and rc_prompt_attention, sliding (a ring) and global layers.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const contract = @import("contract");
const rig_ = @import("rig.zig");
const controls = @import("metal_ricursive_controls.zig");
const rc = @import("metal_ricursive.zig");
const c = contract.spec;
const g = contract.gates;
const rows_in = contract.rows;
const st = tf.ricursive.state;
const R = rc.R;
const DH = rc.DH;
const xv = rc.xv;
const kvw = R.kv_heads * DH;
const grp = R.heads / R.kv_heads;
const scale: f64 = tf.ricursive.config.headScale();
const nan16 = 0x7fc0;
/// A global layer's window: every key before (forward.zig's every_key).
const every_key: usize = std.math.maxInt(i32);

/// Sliding layers (RoPE'd keys in a ring, a 1,024-key window) and global layers (every key, a slot a position).
pub const Layer = enum { sliding, global };

fn windowOf(l: Layer) usize {
    return if (l == .sliding) R.window else every_key;
}

/// The cache's slots for `k` positions: the sliding ring, or whole 64-key blocks (state.slots).
fn ringOf(l: Layer, k: usize) usize {
    return if (l == .sliding) rc.family.sliding_ring else st.slots(k);
}

/// The first key position row `pos` sees.
fn firstKey(l: Layer, pos: usize) usize {
    return (pos + 1) -| windowOf(l);
}

const softmax_sums: c.Sums = .{ .f16 = 0, .bf16 = 0, .exact = false, .why = "weights of exp: fp32 sums held to the float64 bound, not integers" };

/// rc_vec_args, as forward.zig fills it.
const VArgs = extern struct { rows: u32, pos: u32, heads: u32, groups: u32, blocks: u32, chunk: u32, first: u32, scale: f32, window: u32, ring: u32 };

/// Random K and V in slots holding positions to `last`; NaN in slots none reached or holding one before `from`.
fn fillCache(rng: std.Random, k: []u16, v: []u16, ring: usize, last: usize, from: ?usize) void {
    rc.random16(rng, k, -2, 2);
    rc.random16(rng, v, -2, 2);
    for (0..ring) |s| {
        const newest: ?usize = if (s > last) null else last - (last - s) % ring;
        const poison = newest == null or (from != null and newest.? < from.?);
        if (poison) {
            @memset(k[s * kvw ..][0..kvw], nan16);
            @memset(v[s * kvw ..][0..kvw], nan16);
        }
    }
}

/// The scaled dot of query `qrow` (a [heads, DH] row) head `h` with the key at position `key`, and its terms' size.
fn dot(q: []const u16, qrow: usize, h: usize, k: []const u16, slot: usize) [2]f64 {
    const kv = h / grp;
    var sum: f64 = 0;
    var abs: f64 = 0;
    for (0..DH) |d| {
        const t = scale * xv(q, (qrow * R.heads + h) * DH + d) * xv(k, slot * kvw + kv * DH + d);
        sum += t;
        abs += @abs(t);
    }
    return .{ sum, abs };
}

/// Every key of a block in sequence, four at a time, then the merge of the four lanes.
fn slidingDepth(_: c.Shape) usize {
    return st.ring_chunk + 16;
}

fn globalDepth(s: c.Shape) usize {
    return @min(s.k, 2048) + 16;
}

pub fn decodeSpec(l: Layer) c.Spec {
    const served: []const c.Shape = switch (l) {
        // a verify window deep in a stream (its window across the ring's wrap), a short one early
        .sliding => &.{ .{ .rows = 16, .n = R.heads, .k = 100016 }, .{ .rows = 4, .n = R.heads, .k = 1500 } },
        .global => &.{ .{ .rows = 16, .n = R.heads, .k = 6000 }, .{ .rows = 3, .n = R.heads, .k = 700 } },
    };
    const chunk: usize = if (l == .sliding) st.ring_chunk else 32;
    return .{ .name = if (l == .sliding) "ricursive sliding rc_decode_vec" else "ricursive global rc_decode_vec", .accum = .f32, .depth = if (l == .sliding) slidingDepth else globalDepth, .ops = 24, .out = .f32, .tiles = .{ .{ .size = 16, .step = 1, .min = 1 }, .{}, .{ .size = chunk, .step = 1, .min = 1 } }, .served = served, .sums = softmax_sums };
}

/// rc_decode_vec: a (key head, row, key block) a threadgroup over the blocks forward.zig dispatches for the rows.
pub const DecodeVec = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    layer: Layer,
    mutation: controls.Mutation,
    lib: mtl.Library,
    pipe: mtl.Pipeline,
    shape: c.Shape = .{ .rows = 0, .n = 0, .k = 0 },
    ring: usize = 0,
    chunk: usize = 0,
    first: usize = 0,
    blocks: usize = 0,
    q: []u16 = &.{},
    k: []u16 = &.{},
    v: []u16 = &.{},
    in: [3]rig_.Ref = undefined,
    out: [3]rig_.Ref = undefined,

    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, layer: Layer, mutation: controls.Mutation) !DecodeVec {
        const lib = try controls.glueLibrary(device, a, mutation);
        errdefer lib.deinit();
        return .{ .a = a, .rig = .{ .a = a, .device = device, .queue = queue }, .layer = layer, .mutation = mutation, .lib = lib, .pipe = try rc.pipelineOf(lib, device, "rc_decode_vec") };
    }

    fn drop(p: *DecodeVec) void {
        for ([_]*[]u16{ &p.q, &p.k, &p.v }) |s| {
            p.a.free(s.*);
            s.* = &.{};
        }
        p.rig.reset();
    }

    pub fn deinit(p: *DecodeVec) void {
        p.drop();
        p.rig.deinit();
        p.pipe.deinit();
        p.lib.deinit();
    }

    const vt = g.hooks(DecodeVec);

    pub fn kernel(p: *DecodeVec) g.Kernel {
        return controls.mark(.{ .spec = decodeSpec(p.layer), .ctx = p, .vt = &vt }, p.mutation);
    }

    fn p0(p: *const DecodeVec) usize {
        return p.shape.k - p.shape.rows;
    }

    /// rows query rows at the end of k positions; the blocks forward.zig dispatches for a stream of capacity k.
    pub fn prepare(p: *DecodeVec, s: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (s.n != R.heads or s.rows == 0 or s.rows > st.window_rows or s.k < s.rows) return false;
        p.drop();
        p.shape = s;
        p.ring = ringOf(p.layer, s.k);
        p.chunk = if (p.layer == .sliding) st.ring_chunk else st.chunk(s.k);
        const start = p.p0();
        p.first = firstKey(p.layer, start) / p.chunk;
        p.blocks = (start + s.rows - 1) / p.chunk + 1 - p.first;
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        p.q = try p.a.alloc(u16, s.rows * R.heads * DH);
        p.k = try p.a.alloc(u16, p.ring * kvw);
        p.v = try p.a.alloc(u16, p.ring * kvw);
        rc.random16(rng, p.q, -2, 2);
        fillCache(rng, p.k, p.v, p.ring, s.k - 1, firstKey(p.layer, start));
        if (fill == .nonfinite and start > 1) { // keys every row sees: the first key head's at p0, the second's values
            p.k[(start % p.ring) * kvw + 3] = c.toBf16(std.math.nan(f32));
            p.v[((start - 1) % p.ring) * kvw + DH + 1] = c.toBf16(std.math.nan(f32));
            p.v[((start - 2) % p.ring) * kvw + 2 * DH + 5] = c.toBf16(std.math.inf(f32));
        }
        p.in[0] = try p.rig.input(std.mem.sliceAsBytes(p.q));
        p.in[1] = try p.rig.input(std.mem.sliceAsBytes(p.k));
        p.in[2] = try p.rig.input(std.mem.sliceAsBytes(p.v));
        const slots = R.heads * s.rows * p.blocks;
        p.out[0] = try p.rig.output(slots * 4);
        p.out[1] = try p.rig.output(slots * 4);
        p.out[2] = try p.rig.output(slots * DH * 4);
        return true;
    }

    pub fn count(p: *DecodeVec) usize {
        return R.heads * p.shape.rows * p.blocks * (2 + DH);
    }

    pub fn run(p: *DecodeVec) !void {
        const s = p.shape;
        const pass = p.rig.begin();
        const e = pass.e;
        e.setPipeline(p.pipe);
        for (p.in, 0..) |r, slot| e.setBuffer(r.buf, r.off, slot);
        const a = VArgs{ .rows = @intCast(s.rows), .pos = @intCast(p.p0()), .heads = R.heads, .groups = R.kv_heads, .blocks = @intCast(p.blocks), .chunk = @intCast(p.chunk), .first = @intCast(p.first), .scale = @floatCast(scale), .window = @intCast(windowOf(p.layer)), .ring = @intCast(p.ring) };
        e.setValue(a, 3);
        for (p.out, 4..) |r, slot| e.setBuffer(r.buf, r.off, slot);
        e.dispatchGroups(mtl.Size.of(R.kv_heads, s.rows, p.blocks), mtl.Size.of(32, grp, 1));
        try p.rig.finish(pass);
    }

    /// A partial's keys: the block's within the row's window, up to the row's own position.
    fn span(p: *const DecodeVec, row: usize, z: usize) [2]usize {
        const pos = p.p0() + row;
        const block = p.first + z;
        const from = @max(block * p.chunk, firstKey(p.layer, pos));
        return .{ from, @max(from, @min(pos + 1, (block + 1) * p.chunk)) };
    }

    /// PM (the block's top score), PL (its mass) and PO (its weighted values); `acc` replays the sums.
    fn partials(p: *DecodeVec, acc: ?struct { a: c.Accum, o: g.Order }, want: []f64, mag: ?[]f64) !void {
        const s = p.shape;
        const slots = R.heads * s.rows * p.blocks;
        const sc = try p.a.alloc(f64, 2 * p.chunk);
        defer p.a.free(sc);
        const t = sc[p.chunk..];
        var abs_of: [2048]f64 = undefined;
        for (0..R.heads) |h| for (0..s.rows) |row| for (0..p.blocks) |z| {
            const slot = (h * s.rows + row) * p.blocks + z;
            const kv = h / grp;
            const from, const to = p.span(row, z);
            var top: f64 = -std.math.inf(f64); // Metal's max drops a NaN score; exp(NaN - top) carries it on
            var top_abs: f64 = 0;
            for (from..to, 0..) |key, i| {
                sc[i], abs_of[i] = dot(p.q, row, h, p.k, key % p.ring);
                if (!std.math.isNan(sc[i]) and sc[i] > top) {
                    top = sc[i];
                    top_abs = abs_of[i];
                }
            }
            const n = to - from;
            var mass: f64 = 0;
            for (0..n) |i| {
                t[i] = @exp(sc[i] - top);
                mass += t[i];
            }
            want[slot] = top;
            want[slots + slot] = if (acc) |x| rows_in.sumIn(x.a, x.o, t[0..n]) else mass;
            if (mag) |m| {
                m[slot] = top_abs;
                m[slots + slot] = mass;
            }
            for (0..DH) |d| {
                var sum: f64 = 0;
                var abs: f64 = 0;
                for (from..to, 0..) |key, i| {
                    t[i] = @exp(sc[i] - top) * xv(p.v, (key % p.ring) * kvw + kv * DH + d);
                    sum += t[i];
                    abs += @abs(t[i]);
                }
                const o = 2 * slots + slot * DH + d;
                want[o] = if (acc) |x| rows_in.sumIn(x.a, x.o, t[0..n]) else sum;
                if (mag) |m| m[o] = abs;
            }
        };
    }

    pub fn reference(p: *DecodeVec, want: []f64, mag: []f64) !void {
        return p.partials(null, want, mag);
    }

    pub fn simulate(p: *DecodeVec, acc: c.Accum, order: g.Order, out: []f64) !void {
        return p.partials(.{ .a = acc, .o = order }, out, null);
    }
};

/// The tensor-op accumulation over a row's keys (a window's at most), the rescales and the final division.
fn promptDepth(s: c.Shape) usize {
    return s.k / 16 + 32;
}

fn slidingPromptDepth(s: c.Shape) usize {
    return @min(s.k, R.window + 64) / 16 + 32;
}

pub fn promptSpec(l: Layer) c.Spec {
    const served: []const c.Shape = switch (l) {
        // a chunk mid-block past a full window, the shortest from the start, one whose window wraps the ring
        .sliding => &.{ .{ .rows = 33, .n = R.heads, .k = 1233 }, .{ .rows = 17, .n = R.heads, .k = 17 }, .{ .rows = 300, .n = R.heads, .k = 3 * 3072 + 500 } },
        // a chunk mid-block, the shortest from the start, a longer one deeper in
        .global => &.{ .{ .rows = 33, .n = R.heads, .k = 333 }, .{ .rows = 17, .n = R.heads, .k = 17 }, .{ .rows = 100, .n = R.heads, .k = 1300 } },
    };
    const name = if (l == .sliding) "ricursive sliding attention rc_prompt_attention" else "ricursive global attention rc_prompt_attention";
    return .{ .name = name, .accum = .f32, .depth = if (l == .sliding) slidingPromptDepth else promptDepth, .ops = 8, .out = .bf16, .tiles = .{ .{ .size = 32, .step = 1, .min = st.window_rows + 1 }, .{}, .{ .size = 64, .step = 1, .min = 1 } }, .served = served, .sums = softmax_sums };
}

/// rc_prompt_attention: rows query rows at positions k - rows.. over the layer's cache (a ring, or a slot a position).
pub const Prompt = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    pipe: mtl.Pipeline,
    layer: Layer,
    mutation: controls.Mutation,
    shape: c.Shape = .{ .rows = 0, .n = 0, .k = 0 },
    ring: usize = 0,
    q: []u16 = &.{},
    k: []u16 = &.{},
    v: []u16 = &.{},
    in: [3]rig_.Ref = undefined,
    out: rig_.Ref = undefined,

    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, layer: Layer, mutation: controls.Mutation) !Prompt {
        const plain = try rc.family.attentionSource(device, a);
        defer a.free(plain);
        const src = try controls.apply(a, plain, mutation);
        defer a.free(src);
        return .{ .a = a, .rig = .{ .a = a, .device = device, .queue = queue }, .pipe = try rig_.pipeline(device, src, rc.family.attention_name), .layer = layer, .mutation = mutation };
    }

    fn drop(p: *Prompt) void {
        for ([_]*[]u16{ &p.q, &p.k, &p.v }) |s| {
            p.a.free(s.*);
            s.* = &.{};
        }
        p.rig.reset();
    }

    pub fn deinit(p: *Prompt) void {
        p.drop();
        p.rig.deinit();
        p.pipe.deinit();
    }

    const vt = g.hooks(Prompt);

    pub fn kernel(p: *Prompt) g.Kernel {
        return controls.mark(.{ .spec = promptSpec(p.layer), .ctx = p, .vt = &vt }, p.mutation);
    }

    fn p0(p: *const Prompt) usize {
        return p.shape.k - p.shape.rows;
    }

    /// Q padded to whole 32-row tiles (the padding NaN), the cache as the engine leaves it before the attention.
    pub fn prepare(p: *Prompt, s: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (s.n != R.heads or s.rows == 0 or s.rows > st.prompt_rows or s.k < s.rows) return false;
        p.drop();
        p.shape = s;
        p.ring = ringOf(p.layer, s.k);
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        const pad = std.mem.alignForward(usize, s.rows, 32);
        p.q = try p.a.alloc(u16, pad * R.heads * DH);
        p.k = try p.a.alloc(u16, p.ring * kvw);
        p.v = try p.a.alloc(u16, p.ring * kvw);
        rc.random16(rng, p.q, -2, 2);
        @memset(p.q[s.rows * R.heads * DH ..], nan16);
        fillCache(rng, p.k, p.v, p.ring, s.k - 1, null);
        const start = p.p0();
        if (fill == .nonfinite and start > 1) { // keys every row sees
            p.k[(start % p.ring) * kvw + 2] = c.toBf16(std.math.nan(f32));
            p.v[((start - 1) % p.ring) * kvw + DH + 4] = c.toBf16(std.math.nan(f32));
            p.v[((start - 2) % p.ring) * kvw + 2 * DH + 9] = c.toBf16(std.math.inf(f32));
        }
        p.in[0] = try p.rig.input(std.mem.sliceAsBytes(p.q));
        p.in[1] = try p.rig.input(std.mem.sliceAsBytes(p.k));
        p.in[2] = try p.rig.input(std.mem.sliceAsBytes(p.v));
        p.out = try p.rig.output(s.rows * R.heads * DH * 2);
        return true;
    }

    pub fn count(p: *Prompt) usize {
        return p.shape.rows * R.heads * DH;
    }

    pub fn run(p: *Prompt) !void {
        const s = p.shape;
        const pass = p.rig.begin();
        const e = pass.e;
        e.setPipeline(p.pipe);
        for (p.in, 0..) |r, slot| e.setBuffer(r.buf, r.off, slot);
        e.setBuffer(p.out.buf, p.out.off, 3);
        e.setValue([4]i32{ @intCast(s.rows), @intCast(p.p0()), @intCast(windowOf(p.layer)), @intCast(p.ring) }, 4);
        e.dispatchGroups(mtl.Size.of((s.rows + 31) / 32, R.heads, 1), mtl.Size.of(DH / 64 * 32, 1, 1));
        try p.rig.finish(pass);
    }

    /// Each row's heads over the keys of its window, in float64 (or with `acc`, the sums replayed).
    fn fold(p: *Prompt, acc: ?struct { a: c.Accum, o: g.Order }, want: []f64, mag: ?[]f64) !void {
        const s = p.shape;
        const most = @min(s.k, windowOf(p.layer));
        const buf = try p.a.alloc(f64, 3 * most);
        defer p.a.free(buf);
        const sc = buf[0..most];
        const w = buf[most .. 2 * most];
        const t = buf[2 * most ..];
        for (0..s.rows) |row| for (0..R.heads) |h| {
            const kv = h / grp;
            const pos = p.p0() + row;
            const from = firstKey(p.layer, pos);
            const n = pos + 1 - from;
            var top: f64 = -std.math.inf(f64); // the tile's max drops a NaN score; exp2(NaN - top) carries it on
            for (0..n) |i| {
                sc[i] = dot(p.q, row, h, p.k, (from + i) % p.ring)[0];
                if (!std.math.isNan(sc[i])) top = @max(top, sc[i]);
            }
            var total: f64 = 0;
            var total_abs: f64 = 0;
            for (0..n) |i| {
                w[i] = @exp(sc[i] - top);
                total += w[i];
                total_abs += @abs(w[i]);
            }
            if (acc) |x| total = rows_in.sumIn(x.a, x.o, w[0..n]);
            for (0..DH) |d| {
                var sum: f64 = 0;
                var abs: f64 = 0;
                for (0..n) |i| {
                    t[i] = w[i] * xv(p.v, ((from + i) % p.ring) * kvw + kv * DH + d);
                    sum += t[i];
                    abs += @abs(t[i]);
                }
                if (acc) |x| sum = rows_in.sumIn(x.a, x.o, t[0..n]);
                const o = (row * R.heads + h) * DH + d;
                want[o] = if (acc == null) sum / total else rc.bf(sum / total);
                if (mag) |m| m[o] = abs / total + @abs(sum) * total_abs / (total * total);
            }
        };
    }

    pub fn reference(p: *Prompt, want: []f64, mag: []f64) !void {
        return p.fold(null, want, mag);
    }

    /// The tensor op takes the fp32 probabilities at relaxed precision: a bf16 ulp of each weighted term more.
    pub fn bounds(_: *Prompt, want: []const f64, mag: []const f64, allow: []f64) !void {
        for (allow, want, mag) |*l, w, m| l.* += 0x1p-7 * m + 0x1p-8 * @abs(w);
    }

    pub fn simulate(p: *Prompt, acc: c.Accum, order: g.Order, out: []f64) !void {
        return p.fold(.{ .a = acc, .o = order }, out, null);
    }
};
