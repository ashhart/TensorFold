//! Llama's attention under the contract: decode partials and their merge (glue), and the tensor-unit prompt kernel.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const contract = @import("contract");
const rig_ = @import("rig.zig");
const llama = @import("metal_llama.zig");
const c = contract.spec;
const g = contract.gates;
const rows_in = contract.rows;
const family = tf.llama.contracts;
const DH = llama.DH;
const bf = llama.bf;

/// glue.metal's ll_vec_args, as forward.zig fills it.
const VArgs = extern struct { rows: u32, pos: u32, heads: u32, groups: u32, blocks: u32, chunk: u32, first: u32, scale: f32 };

const softmax_sums: c.Sums = .{ .f16 = 0, .bf16 = 0, .exact = false, .why = "weights of exp: fp32 sums held to the float64 bound, not integers" };

fn geometryOf(heads: usize) ?family.Geometry {
    for (family.geometries) |geo| if (geo.heads == heads) return geo;
    return null;
}

fn random16(rng: std.Random, out: []u16, lo: f32, hi: f32) void {
    for (out) |*v| v.* = c.toBf16(lo + rng.float(f32) * (hi - lo));
}

fn xv(v: []const u16, i: usize) f64 {
    return c.fromBf16(v[i]);
}

/// A block's keys in sequence, four at a time, then the merge of the four lanes.
fn decodeDepth(s: c.Shape) usize {
    return @min(s.k, 2048) + 16;
}

/// ll_decode_vec: a (key head, row, key block) a threadgroup; its partial maximum, mass and output over the block.
pub const DecodeVec = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    lib: mtl.Library,
    pipe: mtl.Pipeline,
    shape: c.Shape = .{ .rows = 0, .n = 0, .k = 0 },
    geo: family.Geometry = family.nanbeige43,
    chunk: usize = 32,
    blocks: usize = 0,
    q: []u16 = &.{},
    k: []u16 = &.{},
    v: []u16 = &.{},
    in: [3]rig_.Ref = undefined,
    out: [3]rig_.Ref = undefined,

    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue) !DecodeVec {
        const lib = try family.glueLibrary(device, a);
        errdefer lib.deinit();
        return .{ .a = a, .rig = .{ .a = a, .device = device, .queue = queue }, .lib = lib, .pipe = try llama.pipelineOf(lib, device, "ll_decode_vec") };
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
        return .{ .spec = .{ .name = "llama glue ll_decode_vec", .accum = .f32, .depth = decodeDepth, .ops = 24, .out = .f32, .tiles = .{ .{ .size = 16, .step = 1, .min = 1 }, .{}, .{ .size = 32, .step = 1, .min = 1 } }, .served = &.{ .{ .rows = 16, .n = family.nanbeige43.heads, .k = 4096 }, .{ .rows = 4, .n = family.nanbeige41.heads, .k = 1024 } }, .sums = softmax_sums }, .ctx = p, .vt = &vt };
    }

    /// rows query rows at the end of k positions, n query heads; the blocks as the served cache of k would hold.
    pub fn prepare(p: *DecodeVec, s: c.Shape, fill: g.Fill, seed: u64) !bool {
        const geo = geometryOf(s.n) orelse return false;
        if (s.rows == 0 or s.rows > 16 or s.k < s.rows) return false;
        p.drop();
        p.shape = s;
        p.geo = geo;
        p.chunk = tf.llama.state.chunk(s.k);
        p.blocks = (s.k - 1) / p.chunk + 1;
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        const kvw = geo.kv_heads * DH;
        p.q = try p.a.alloc(u16, s.rows * geo.heads * DH);
        p.k = try p.a.alloc(u16, s.k * kvw);
        p.v = try p.a.alloc(u16, s.k * kvw);
        random16(rng, p.q, -2, 2);
        random16(rng, p.k, -2, 2);
        random16(rng, p.v, -2, 2);
        if (fill == .nonfinite) {
            p.k[5 * kvw + 3] = c.toBf16(std.math.nan(f32)); // the first key head's queries all see key 5
            p.v[@min(9, s.k - 1) * kvw + DH + 1] = c.toBf16(std.math.nan(f32)); // the second head's outputs only
        }
        p.in[0] = try p.rig.input(std.mem.sliceAsBytes(p.q));
        p.in[1] = try p.rig.input(std.mem.sliceAsBytes(p.k));
        p.in[2] = try p.rig.input(std.mem.sliceAsBytes(p.v));
        const slots = geo.heads * s.rows * p.blocks;
        p.out[0] = try p.rig.output(slots * 4);
        p.out[1] = try p.rig.output(slots * 4);
        p.out[2] = try p.rig.output(slots * DH * 4);
        return true;
    }

    pub fn count(p: *DecodeVec) usize {
        return p.geo.heads * p.shape.rows * p.blocks * (2 + DH);
    }

    pub fn run(p: *DecodeVec) !void {
        const s = p.shape;
        const geo = p.geo;
        const pass = p.rig.begin();
        const e = pass.e;
        e.setPipeline(p.pipe);
        for (p.in, 0..) |r, slot| e.setBuffer(r.buf, r.off, slot);
        e.setValue(VArgs{ .rows = @intCast(s.rows), .pos = @intCast(s.k - s.rows), .heads = geo.heads, .groups = geo.kv_heads, .blocks = @intCast(p.blocks), .chunk = @intCast(p.chunk), .first = 0, .scale = tf.llama.config.headScale() }, 3);
        for (p.out, 4..) |r, slot| e.setBuffer(r.buf, r.off, slot);
        e.dispatchGroups(mtl.Size.of(geo.kv_heads, s.rows, p.blocks), mtl.Size.of(32, geo.heads / geo.kv_heads, 1));
        try p.rig.finish(pass);
    }

    /// A slot's keys: from the block's first to the row's own position, within the block.
    fn span(p: *const DecodeVec, row: usize, block: usize) [2]usize {
        const pos = p.shape.k - p.shape.rows + row;
        return .{ block * p.chunk, @min(pos + 1, (block + 1) * p.chunk) };
    }

    fn score(p: *const DecodeVec, head: usize, row: usize, key: usize) f64 {
        return p.dot(head, row, key)[0];
    }

    /// The scaled dot and the magnitude of its terms.
    fn dot(p: *const DecodeVec, head: usize, row: usize, key: usize) [2]f64 {
        const geo = p.geo;
        const kv = head / (geo.heads / geo.kv_heads);
        var sum: f64 = 0;
        var abs: f64 = 0;
        for (0..DH) |d| {
            const t = @as(f64, tf.llama.config.headScale()) * xv(p.q, (row * geo.heads + head) * DH + d) * xv(p.k, key * geo.kv_heads * DH + kv * DH + d);
            sum += t;
            abs += @abs(t);
        }
        return .{ sum, abs };
    }

    /// PM (the block's top score), PL (its mass) and PO (its weighted values), each in its own output.
    pub fn reference(p: *DecodeVec, want: []f64, mag: []f64) !void {
        const s = p.shape;
        const geo = p.geo;
        const slots = geo.heads * s.rows * p.blocks;
        const kvw = geo.kv_heads * DH;
        for (0..geo.heads) |head| for (0..s.rows) |row| for (0..p.blocks) |block| {
            const slot = (head * s.rows + row) * p.blocks + block;
            const kv = head / (geo.heads / geo.kv_heads);
            const from, const to = p.span(row, block);
            var top: f64 = -std.math.inf(f64); // Metal's max drops a NaN score; exp(NaN - top) carries it on
            var top_abs: f64 = 0; // the top score's terms, the magnitude its fp32 dot rounds against
            var key = from;
            while (key < to) : (key += 1) {
                const sc, const abs = p.dot(head, row, key);
                if (!std.math.isNan(sc) and sc > top) {
                    top = sc;
                    top_abs = abs;
                }
            }
            var mass: f64 = 0;
            var mass_abs: f64 = 0;
            const o = want[2 * slots + slot * DH ..][0..DH];
            const om = mag[2 * slots + slot * DH ..][0..DH];
            @memset(o, 0);
            @memset(om, 0);
            key = from;
            while (key < to) : (key += 1) {
                const w = @exp(p.score(head, row, key) - top);
                mass += w;
                mass_abs += @abs(w);
                for (0..DH) |d| {
                    const t = w * xv(p.v, key * kvw + kv * DH + d);
                    o[d] += t;
                    om[d] += @abs(t);
                }
            }
            want[slot] = top;
            mag[slot] = top_abs;
            want[slots + slot] = mass;
            mag[slots + slot] = mass_abs;
        };
    }

    pub fn simulate(p: *DecodeVec, acc: c.Accum, order: g.Order, out: []f64) !void {
        const s = p.shape;
        const geo = p.geo;
        const slots = geo.heads * s.rows * p.blocks;
        const kvw = geo.kv_heads * DH;
        const t = try p.a.alloc(f64, p.chunk);
        defer p.a.free(t);
        for (0..geo.heads) |head| for (0..s.rows) |row| for (0..p.blocks) |block| {
            const slot = (head * s.rows + row) * p.blocks + block;
            const kv = head / (geo.heads / geo.kv_heads);
            const from, const to = p.span(row, block);
            var top: f64 = -std.math.inf(f64);
            for (from..to) |key| {
                const sc = p.score(head, row, key);
                if (!std.math.isNan(sc)) top = @max(top, sc);
            }
            for (from..to, 0..) |key, i| t[i] = @exp(p.score(head, row, key) - top);
            out[slot] = top;
            out[slots + slot] = rows_in.sumIn(acc, order, t[0 .. to - from]);
            for (0..DH) |d| {
                for (from..to, 0..) |key, i| t[i] = @exp(p.score(head, row, key) - top) * xv(p.v, key * kvw + kv * DH + d);
                out[2 * slots + slot * DH + d] = rows_in.sumIn(acc, order, t[0 .. to - from]);
            }
        };
    }
};

/// The simdgroups' strided sums over the blocks, the fold through threadgroup memory and the division.
fn mergeDepth(s: c.Shape) usize {
    return s.k / 32 + 16;
}

/// ll_merge_vec: the partials of every block folded into a row's bf16 attention output.
pub const MergeVec = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    lib: mtl.Library,
    pipe: mtl.Pipeline,
    shape: c.Shape = .{ .rows = 0, .n = 0, .k = 0 },
    pm: []f32 = &.{},
    pl: []f32 = &.{},
    po: []f32 = &.{},
    in: [3]rig_.Ref = undefined,
    out: rig_.Ref = undefined,

    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue) !MergeVec {
        const lib = try family.glueLibrary(device, a);
        errdefer lib.deinit();
        return .{ .a = a, .rig = .{ .a = a, .device = device, .queue = queue }, .lib = lib, .pipe = try llama.pipelineOf(lib, device, "ll_merge_vec") };
    }

    fn drop(p: *MergeVec) void {
        for ([_]*[]f32{ &p.pm, &p.pl, &p.po }) |s| {
            p.a.free(s.*);
            s.* = &.{};
        }
        p.rig.reset();
    }

    pub fn deinit(p: *MergeVec) void {
        p.drop();
        p.rig.deinit();
        p.pipe.deinit();
        p.lib.deinit();
    }

    const vt = g.hooks(MergeVec);

    pub fn kernel(p: *MergeVec) g.Kernel {
        return .{ .spec = .{ .name = "llama glue ll_merge_vec", .accum = .f32, .depth = mergeDepth, .ops = 6, .out = .bf16, .tiles = .{ .{ .size = 16, .step = 1, .min = 1 }, .{}, .{ .size = 32, .step = 1, .min = 1 } }, .served = &.{ .{ .rows = 16, .n = family.nanbeige43.heads, .k = 128 }, .{ .rows = 4, .n = family.nanbeige41.heads, .k = 64 } }, .sums = softmax_sums }, .ctx = p, .vt = &vt };
    }

    /// rows rows of n heads over k blocks of partials, as decode leaves them: every fourth block past the first empty.
    pub fn prepare(p: *MergeVec, s: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (s.rows == 0 or s.rows > 16 or s.n == 0 or s.k == 0) return false;
        p.drop();
        p.shape = s;
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        const slots = s.n * s.rows * s.k;
        p.pm = try p.a.alloc(f32, slots);
        p.pl = try p.a.alloc(f32, slots);
        p.po = try p.a.alloc(f32, slots * DH);
        for (p.pm, p.pl, 0..) |*m, *l, i| {
            const empty = i % s.k != 0 and i % 4 == 3;
            m.* = if (empty) -std.math.inf(f32) else rng.float(f32) * 20 - 10;
            l.* = if (empty) 0 else 0.5 + rng.float(f32) * 64;
        }
        for (p.po, 0..) |*o, i| o.* = (rng.float(f32) * 2 - 1) * p.pl[i / DH];
        if (fill == .nonfinite) {
            p.po[3 * DH + 7] = std.math.nan(f32); // the first head's first row
            if (s.k > 1) p.pm[s.k + 1] = std.math.inf(f32); // the second slot's top: exp(inf - inf)
        }
        p.in[0] = try p.rig.input(std.mem.sliceAsBytes(p.pm));
        p.in[1] = try p.rig.input(std.mem.sliceAsBytes(p.pl));
        p.in[2] = try p.rig.input(std.mem.sliceAsBytes(p.po));
        p.out = try p.rig.output(s.rows * s.n * DH * 2);
        return true;
    }

    pub fn count(p: *MergeVec) usize {
        return p.shape.rows * p.shape.n * DH;
    }

    pub fn run(p: *MergeVec) !void {
        const s = p.shape;
        const pass = p.rig.begin();
        const e = pass.e;
        e.setPipeline(p.pipe);
        for (p.in, 0..) |r, slot| e.setBuffer(r.buf, r.off, slot);
        e.setValue(VArgs{ .rows = @intCast(s.rows), .pos = 0, .heads = @intCast(s.n), .groups = 1, .blocks = @intCast(s.k), .chunk = 32, .first = 0, .scale = 1 }, 3);
        e.setBuffer(p.out.buf, p.out.off, 4);
        e.dispatchGroups(mtl.Size.of(s.n, s.rows, 1), mtl.Size.of(1024, 1, 1));
        try p.rig.finish(pass);
    }

    fn fold(p: *MergeVec, acc: ?struct { a: c.Accum, o: g.Order }, want: []f64, mag: ?[]f64) !void {
        const s = p.shape;
        const t = try p.a.alloc(f64, s.k);
        defer p.a.free(t);
        for (0..s.n) |head| for (0..s.rows) |row| {
            const at = head * s.rows + row;
            var top: f64 = -std.math.inf(f64); // Metal's max drops a NaN partial
            for (0..s.k) |b| if (!std.math.isNan(p.pm[at * s.k + b])) {
                top = @max(top, p.pm[at * s.k + b]);
            };
            var total: f64 = 0;
            var total_abs: f64 = 0;
            for (0..s.k) |b| {
                const l: f64 = p.pl[at * s.k + b];
                t[b] = if (l > 0) l * @exp(@as(f64, p.pm[at * s.k + b]) - top) else 0;
                total += t[b];
                total_abs += @abs(t[b]);
            }
            if (acc) |x| total = rows_in.sumIn(x.a, x.o, t);
            for (0..DH) |d| {
                var sum: f64 = 0;
                var abs: f64 = 0;
                for (0..s.k) |b| {
                    const l: f64 = p.pl[at * s.k + b];
                    t[b] = if (l > 0) @exp(@as(f64, p.pm[at * s.k + b]) - top) * p.po[(at * s.k + b) * DH + d] else 0;
                    sum += t[b];
                    abs += @abs(t[b]);
                }
                if (acc) |x| sum = rows_in.sumIn(x.a, x.o, t);
                const o = (row * s.n + head) * DH + d;
                want[o] = if (acc == null) sum / total else bf(sum / total);
                if (mag) |m| m[o] = abs / total + @abs(sum) * total_abs / (total * total);
            }
        };
    }

    pub fn reference(p: *MergeVec, want: []f64, mag: []f64) !void {
        return p.fold(null, want, mag);
    }

    pub fn simulate(p: *MergeVec, acc: c.Accum, order: g.Order, out: []f64) !void {
        return p.fold(.{ .a = acc, .o = order }, out, null);
    }
};

/// The tensor-op accumulation over the keys, the rescales and the final division.
fn promptDepth(s: c.Shape) usize {
    return s.k / 16 + 32;
}

/// ll_prompt_attention for one geometry: rows query rows at positions k - rows.. over a cache of k positions.
pub const Prompt = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    pipe: mtl.Pipeline,
    geo: family.Geometry,
    name: []const u8,
    served: [2]c.Shape,
    shape: c.Shape = .{ .rows = 0, .n = 0, .k = 0 },
    q: []u16 = &.{},
    k: []u16 = &.{},
    v: []u16 = &.{},
    in: [3]rig_.Ref = undefined,
    out: rig_.Ref = undefined,

    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, geo: family.Geometry) !Prompt {
        const src = try family.attentionSource(device, a, geo);
        defer a.free(src);
        const name = try std.fmt.allocPrint(a, "llama {s} attention ll_prompt_attention", .{geo.name});
        errdefer a.free(name);
        const big = geo.heads == family.nanbeige43.heads;
        return .{ .a = a, .rig = .{ .a = a, .device = device, .queue = queue }, .pipe = try rig_.pipeline(device, src, family.attention_name), .geo = geo, .name = name, .served = .{ .{ .rows = if (big) 256 else 64, .n = geo.heads, .k = if (big) 1280 else 64 }, .{ .rows = 33, .n = geo.heads, .k = 33 } } };
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
        p.a.free(p.name);
    }

    const vt = g.hooks(Prompt);

    pub fn kernel(p: *Prompt) g.Kernel {
        return .{ .spec = .{ .name = p.name, .accum = .f32, .depth = promptDepth, .ops = 8, .out = .bf16, .tiles = .{ .{ .size = 32, .step = 1, .min = 1 }, .{}, .{ .size = 64, .step = 1, .min = 1 } }, .served = &p.served, .sums = softmax_sums }, .ctx = p, .vt = &vt };
    }

    /// Q padded to whole 32-row tiles and the cache to whole 64-key blocks, as the served buffers hold them.
    pub fn prepare(p: *Prompt, s: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (s.n != p.geo.heads or s.rows == 0 or s.rows > 2048 or s.k < s.rows) return false;
        p.drop();
        p.shape = s;
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        const geo = p.geo;
        const kvw = geo.kv_heads * DH;
        const slots = std.mem.alignForward(usize, s.k, 64);
        p.q = try p.a.alloc(u16, std.mem.alignForward(usize, s.rows, 32) * geo.heads * DH);
        p.k = try p.a.alloc(u16, slots * kvw);
        p.v = try p.a.alloc(u16, slots * kvw);
        random16(rng, p.q, -2, 2);
        random16(rng, p.k, -2, 2);
        random16(rng, p.v, -2, 2);
        if (fill == .nonfinite) {
            p.k[3 * kvw + 2] = c.toBf16(std.math.nan(f32)); // key 3: every row of the first key head's queries
            if (geo.kv_heads > 1) p.v[@min(10, s.k - 1) * kvw + DH + 4] = c.toBf16(std.math.nan(f32));
        }
        p.in[0] = try p.rig.input(std.mem.sliceAsBytes(p.q));
        p.in[1] = try p.rig.input(std.mem.sliceAsBytes(p.k));
        p.in[2] = try p.rig.input(std.mem.sliceAsBytes(p.v));
        p.out = try p.rig.output(s.rows * geo.heads * DH * 2);
        return true;
    }

    pub fn count(p: *Prompt) usize {
        return p.shape.rows * p.geo.heads * DH;
    }

    pub fn run(p: *Prompt) !void {
        const s = p.shape;
        const pass = p.rig.begin();
        const e = pass.e;
        e.setPipeline(p.pipe);
        for (p.in, 0..) |r, slot| e.setBuffer(r.buf, r.off, slot);
        e.setBuffer(p.out.buf, p.out.off, 3);
        e.setValue([2]i32{ @intCast(s.rows), @intCast(s.k - s.rows) }, 4);
        e.dispatchGroups(mtl.Size.of((s.rows + 31) / 32, p.geo.heads, 1), mtl.Size.of(DH / 64 * 32, 1, 1));
        try p.rig.finish(pass);
    }

    fn score(p: *const Prompt, head: usize, row: usize, key: usize) f64 {
        const geo = p.geo;
        const kv = head / (geo.heads / geo.kv_heads);
        var dot: f64 = 0;
        for (0..DH) |d| dot += xv(p.q, (row * geo.heads + head) * DH + d) * xv(p.k, key * geo.kv_heads * DH + kv * DH + d);
        return dot * @as(f64, tf.llama.config.headScale());
    }

    fn fold(p: *Prompt, acc: ?struct { a: c.Accum, o: g.Order }, want: []f64, mag: ?[]f64) !void {
        const s = p.shape;
        const geo = p.geo;
        const kvw = geo.kv_heads * DH;
        const p0 = s.k - s.rows;
        const t = try p.a.alloc(f64, s.k);
        defer p.a.free(t);
        for (0..s.rows) |row| for (0..geo.heads) |head| {
            const kv = head / (geo.heads / geo.kv_heads);
            const keys = p0 + row + 1;
            var top: f64 = -std.math.inf(f64); // the tile's max drops a NaN score; exp2(NaN - top) carries it on
            for (0..keys) |key| {
                const sc = p.score(head, row, key);
                if (!std.math.isNan(sc)) top = @max(top, sc);
            }
            var total: f64 = 0;
            var total_abs: f64 = 0;
            for (0..keys) |key| {
                t[key] = @exp(p.score(head, row, key) - top);
                total += t[key];
                total_abs += @abs(t[key]);
            }
            if (acc) |x| total = rows_in.sumIn(x.a, x.o, t[0..keys]);
            for (0..DH) |d| {
                var sum: f64 = 0;
                var abs: f64 = 0;
                for (0..keys) |key| {
                    t[key] = @exp(p.score(head, row, key) - top) * xv(p.v, key * kvw + kv * DH + d);
                    sum += t[key];
                    abs += @abs(t[key]);
                }
                if (acc) |x| sum = rows_in.sumIn(x.a, x.o, t[0..keys]);
                const o = (row * geo.heads + head) * DH + d;
                want[o] = if (acc == null) sum / total else bf(sum / total);
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
