//! Gemma 4's prompt and decode attention on the M5's tensor units (prompt_attention.metal), as kernels.zig builds them.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const contract = @import("contract");
const sources = @import("kernel_sources");
const rig_ = @import("rig.zig");
const c = contract.spec;
const g = contract.gates;
const ar = contract.attention_range;
const cfg = tf.gemma4.config;
const ring = tf.gemma4.state.ring;
const nan16 = 0x7fc0;
const log2e = 1.4426950408889634;
/// gemma_decode_attention's CD (keys a threadgroup) and BK (keys a step); the prompt kernel's RT (rows a tile).
const cd = 256;
const bk = 32;
const rt = 32;
/// Rows a prompt chunk takes (state.chunk_rows) and a decode call (forward.zig's attend_rows).
const prompt_rows = tf.gemma4.state.chunk_rows;
const decode_rows = tf.gemma4.state.max_rows;

/// A layer kind's gemma_prompt_attention, or a global layer's gemma_decode_attention then gemma_attention_merge.
const Kind = enum { prompt, decode };
/// As served; P spread over [0, 1] with values in [0, 1); that case's range without the cut; scores at 2 log2e.
const Case = enum { plain, spread, cutless, edit };

const Variant = struct { geo: cfg.Geometry, global: bool, kind: Kind, case: Case = .plain };

/// Every pipeline the engine launches (the 12B's assistant compiles the 12B's global one), then the cases and controls.
const variants = [_]Variant{
    .{ .geo = cfg.dense31b, .global = false, .kind = .prompt },
    .{ .geo = cfg.dense31b, .global = true, .kind = .prompt },
    .{ .geo = cfg.unified12b, .global = false, .kind = .prompt },
    .{ .geo = cfg.unified12b, .global = true, .kind = .prompt },
    .{ .geo = cfg.dense31b, .global = true, .kind = .decode },
    .{ .geo = cfg.unified12b, .global = true, .kind = .decode },
    .{ .geo = cfg.dense31b, .global = true, .kind = .prompt, .case = .spread },
    .{ .geo = cfg.dense31b, .global = true, .kind = .prompt, .case = .cutless },
    .{ .geo = cfg.dense31b, .global = true, .kind = .decode, .case = .spread },
    .{ .geo = cfg.dense31b, .global = true, .kind = .decode, .case = .cutless },
    .{ .geo = cfg.dense31b, .global = false, .kind = .prompt, .case = .edit },
    .{ .geo = cfg.dense31b, .global = true, .kind = .decode, .case = .edit },
};

fn nameOf(comptime v: Variant) []const u8 {
    const what = if (v.kind == .decode) " decode attention global +merge" else if (v.global) " prompt attention global" else " prompt attention sliding";
    const base = "metal gemma4 " ++ v.geo.name ++ what;
    return switch (v.case) {
        .plain => base,
        .spread => base ++ ", P spread over [0, 1]",
        .cutless => "control: " ++ base ++ ", P spread, its range without the operand cut",
        .edit => "control: " ++ base ++ ", scores at 2 log2e",
    };
}

const names = blk: {
    var out: [variants.len][]const u8 = undefined;
    for (variants, &out) |v, *n| n.* = nameOf(v);
    break :blk out;
};

/// Each kernel's planted control: one source edit gate 3 must catch.
fn edit(k: Kind) [2][]const u8 {
    return if (k == .prompt) .{ "s[i][e] * kLog2e", "s[i][e] * (2.0f * kLog2e)" } else .{ "s[f][e] * 1.44269504089f", "s[f][e] * 2.88539008178f" };
}

/// kernels.zig's header for prompt_attention.metal at a layer kind.
fn header(a: std.mem.Allocator, geo: cfg.Geometry, global: bool) ![]u8 {
    const dh: usize = if (global) cfg.global_dim else cfg.sliding_dim;
    const nk: usize = if (global) geo.global_kv_heads else geo.sliding_kv_heads;
    return std.fmt.allocPrint(a, "#define DH {d}\n#define NQ {d}\n#define G {d}\n#define SL {d}\n#define WINDOW {d}\n#define RING {d}\n", .{ dh, geo.query_heads, geo.query_heads / nk, dh / 128, @as(usize, if (global) 0 else cfg.window), @as(usize, if (global) 0 else ring) });
}

/// The library as kernels.zig builds it on this GPU: the header, then frags.source's text.
fn library(a: std.mem.Allocator, device: mtl.Device, geo: cfg.Geometry, global: bool) ![]u8 {
    const h = try header(a, geo, global);
    defer a.free(h);
    const body = try tf.frags.source(device, a, sources.gemma4_prompt_attention);
    defer a.free(body);
    return std.mem.concat(a, u8, &.{ h, body });
}

/// Gate 5's texts for a record named "gemma4 <geometry> attention <kind> <function>": here, and before the M5.
pub fn limitsText(a: std.mem.Allocator, device: mtl.Device, name: []const u8) !struct { here: []const u8, pre_m5: []const u8 } {
    const geo = if (std.mem.indexOf(u8, name, " 31B ") != null) cfg.dense31b else if (std.mem.indexOf(u8, name, " 12B ") != null) cfg.unified12b else return error.UnknownLaunch;
    const global = std.mem.indexOf(u8, name, " global ") != null;
    const h = try header(a, geo, global);
    defer a.free(h);
    const body = try std.mem.replaceOwned(u8, a, sources.gemma4_prompt_attention, "#include \"../nax.h\"", sources.nax);
    defer a.free(body);
    const pre = try std.mem.concat(a, u8, &.{ h, "#define TF_SIMD_FRAGS 1\n", body });
    return .{ .here = if (device.tensorUnits()) try std.mem.concat(a, u8, &.{ h, body }) else pre, .pre_m5 = pre };
}

fn bf(v: f64) f64 {
    return c.fromBf16(c.toBf16(@floatCast(v)));
}

/// Gate 4's value for key `j` of `n`: big halves that cancel, four ones left, so only exact fp32 sums give 4.
fn stressValue(j: usize, n: usize) f32 {
    const half = n / 2;
    if (j >= 2 * half) return 0;
    const at = if (j < half) j else j - half;
    const big: f32 = if (at % 2 == 0) 127 else 16256;
    const one: f32 = if (at < 8 and at % 2 == 0) 1 else 0;
    return if (j < half) big + one else -big;
}

/// gemma_prompt_attention over `blocks` blocks of `dh / 2` keys, its fp32 P cut by `cut` as it meets V.
fn promptForm(dh: usize, blocks: usize, cut: f64) ar.Form {
    const b: f64 = @floatFromInt(blocks);
    const ns: f64 = @floatFromInt(dh / 64);
    return .{
        .score = @as(f64, @floatFromInt(dh / 16)) * ar.mma_step, // one accumulator over DH / 16 mma steps
        .fast_exp = false,
        .chunk_exp = false,
        .p_rel = 0,
        .p_cut = cut,
        .num = (@as(f64, @floatFromInt(dh / 32)) * ar.mma_step + 1) * b, // DH / 32 P x V steps and a rescale a block
        .den = 3 * b + 3 + (ns - 1), // a 16-key fold (4 levels, 2 adds) a block, its rescale, then NS simdgroups' sums
        .score_round = 2, // s * kLog2e: the constant's rounding and the product's
        .div = 4, // fast 1 / t (2.5 ulps), then acc * inv
        .exp_ulps = .{ 4 * b, 2 * log2e + 0.5 }, // fast::exp2 (4 + 2 |x| ulps) of a weight and its block's rescales
    };
}

/// gemma_decode_attention's 256-key chunks (8 steps of 32 keys over 4 128-dim slices), then the merge over `chunks`.
fn decodeForm(chunks: usize, cut: f64) ar.Form {
    const n: f64 = @floatFromInt(chunks);
    const steps: f64 = cd / bk;
    return .{
        .score = 8 * ar.mma_step + 3, // 8 mma steps a slice, its 4 slices added in order
        .fast_exp = false,
        .chunk_exp = false,
        .p_rel = 0,
        .p_cut = cut,
        .num = (2 * ar.mma_step + 1) * steps + n, // 2 P x V steps and a rescale a step, then the merge's fma chain
        .den = 3 * steps + 3 + n, // two 16-key folds and a rescale a step, then the merge's fma chain
        .score_round = 2 + 4, // s * log2e (2), and PM = top * ln2 then the merge's max, each 2 of |w| + x
        .div = 3, // the merge's fast acc / lsum
        // 8 fast::exp2s (4 + 2 |x| ulps), the merge's fast::exp (4 + 2 |x|), PM's and the merge's x roundings
        .exp_ulps = .{ 4 * steps + 4, 2 * log2e + 0.5 + 2 + 0.5 + 2 },
    };
}

const prompt_sliding = [_]c.Shape{ .{ .rows = 77, .n = cfg.sliding_dim, .k = 0 }, .{ .rows = 64, .n = cfg.sliding_dim, .k = 1000 }, .{ .rows = 100, .n = cfg.sliding_dim, .k = 2000 }, .{ .rows = prompt_rows, .n = cfg.sliding_dim, .k = 0 } };
const prompt_global = [_]c.Shape{ .{ .rows = 77, .n = cfg.global_dim, .k = 0 }, .{ .rows = 64, .n = cfg.global_dim, .k = 700 }, .{ .rows = 33, .n = cfg.global_dim, .k = 1500 }, .{ .rows = prompt_rows, .n = cfg.global_dim, .k = 0 } };
const decode_global = [_]c.Shape{ .{ .rows = decode_rows, .n = cfg.global_dim, .k = 700 }, .{ .rows = 1, .n = cfg.global_dim, .k = 0 }, .{ .rows = 5, .n = cfg.global_dim, .k = 255 }, .{ .rows = 9, .n = cfg.global_dim, .k = 1100 }, .{ .rows = 1, .n = cfg.global_dim, .k = 3000 } };

fn keysDepth(s: c.Shape) usize {
    return s.n + s.k + s.rows;
}

const Probe = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    v: Variant,
    name: []const u8,
    pipe: mtl.Pipeline,
    merge: ?mtl.Pipeline = null,
    rows: usize = 0,
    p0: usize = 0,
    cap: usize = 0,
    stress: bool = false,
    kp: []u16 = &.{}, // keys and values by position [kv heads, positions, dh]
    vp: []u16 = &.{},
    q: []u16 = &.{}, // [rows (prompt: whole tiles), query heads, dh]
    hi: []i32 = &.{}, // each prompt row's last key (an image's rows see the whole image)
    point: []f64 = &.{},
    half: []f64 = &.{},
    down: []f64 = &.{},
    up: []f64 = &.{},
    refs: [5]rig_.Ref = undefined,
    parts: [4]rig_.Ref = undefined,
    out: rig_.Ref = undefined,

    fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, v: Variant, name: []const u8) !Probe {
        const text = try library(a, device, v.geo, v.global);
        defer a.free(text);
        const src = if (v.case != .edit) try a.dupe(u8, text) else blk: {
            const e = edit(v.kind);
            if (std.mem.count(u8, text, e[0]) != 1) return error.ControlEditMissing;
            break :blk try std.mem.replaceOwned(u8, a, text, e[0], e[1]);
        };
        defer a.free(src);
        const fun = if (v.kind == .prompt) "gemma_prompt_attention" else "gemma_decode_attention";
        var p: Probe = .{ .a = a, .rig = .{ .a = a, .device = device, .queue = queue }, .v = v, .name = name, .pipe = try rig_.pipeline(device, src, fun) };
        errdefer p.pipe.deinit();
        if (v.kind == .decode) {
            const glue = try tf.gemma4.kernels.glueSource(a, v.geo, true, false, 4);
            defer a.free(glue);
            p.merge = try rig_.pipeline(device, glue, "gemma_attention_merge");
        }
        return p;
    }

    fn free(p: *Probe) void {
        for ([_][]u16{ p.kp, p.vp, p.q }) |x| p.a.free(x);
        p.a.free(p.hi);
        for ([_][]f64{ p.point, p.half, p.down, p.up }) |x| p.a.free(x);
        p.kp = &.{};
        p.vp = &.{};
        p.q = &.{};
        p.hi = &.{};
        p.point = &.{};
        p.half = &.{};
        p.down = &.{};
        p.up = &.{};
    }

    fn deinit(p: *Probe) void {
        p.free();
        p.rig.deinit();
        p.pipe.deinit();
        if (p.merge) |m| m.deinit();
    }

    fn dh(p: *const Probe) usize {
        return if (p.v.global) cfg.global_dim else cfg.sliding_dim;
    }
    fn nq(p: *const Probe) usize {
        return p.v.geo.query_heads;
    }
    fn nk(p: *const Probe) usize {
        return if (p.v.global) p.v.geo.global_kv_heads else p.v.geo.sliding_kv_heads;
    }
    fn npos(p: *const Probe) usize {
        return p.p0 + p.rows;
    }
    fn chunks(p: *const Probe) usize {
        return (p.npos() + cd - 1) / cd;
    }

    fn spec(p: *const Probe) c.Spec {
        const served: []const c.Shape = if (p.v.kind == .decode) &decode_global else if (p.v.global) &prompt_global else &prompt_sliding;
        const rows: c.Tile = if (p.v.kind == .decode) .{ .size = 16 / (p.nq() / p.nk()) } else .{ .size = rt };
        const keys: c.Tile = .{ .size = if (p.v.kind == .decode) cd else p.dh() / 2, .min = 0 };
        // ops: gate 4's exact sums meet only the final division's roundings
        return .{ .name = p.name, .accum = .f32, .depth = keysDepth, .ops = 4, .out = .bf16, .tiles = .{ rows, .{}, keys }, .served = served };
    }

    /// The keys row `r` attends: a sliding row's window up to its last key, else every key up to its own.
    fn keysOf(p: *const Probe, r: usize) [2]usize {
        const pos = p.p0 + r;
        if (p.v.global) return .{ 0, pos };
        return .{ (pos + 1) -| cfg.window, @intCast(p.hi[r]) };
    }

    /// Row `r`'s form: its prompt tile's blocks (the tile's first window start to its last row's last key), or chunks.
    fn form(p: *const Probe, r: usize) ar.Form {
        const cut: f64 = if (p.v.case == .cutless) 0 else ar.operand_cut;
        if (p.v.kind == .decode) return decodeForm(p.chunks(), cut);
        const r0 = r / rt * rt;
        const last: usize = @intCast(p.hi[@min(r0 + rt, p.rows) - 1]);
        const lo = if (p.v.global) 0 else (p.p0 + r0 + 1) -| cfg.window;
        return promptForm(p.dh(), last / (p.dh() / 2) - lo / (p.dh() / 2) + 1, cut);
    }

    pub fn prepare(p: *Probe, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        const most: usize = if (p.v.kind == .decode) decode_rows else prompt_rows;
        if (shape.rows == 0 or shape.rows > most or shape.n != p.dh()) return false;
        p.rig.reset();
        p.free();
        p.rows = shape.rows;
        p.p0 = shape.k;
        p.stress = fill == .stress;
        const dh_ = p.dh();
        const n = p.npos();
        const pad = if (p.v.kind == .prompt) std.mem.alignForward(usize, p.rows, rt) else p.rows;
        p.kp = try p.a.alloc(u16, p.nk() * n * dh_);
        p.vp = try p.a.alloc(u16, p.nk() * n * dh_);
        p.q = try p.a.alloc(u16, pad * p.nq() * dh_);
        p.hi = try p.a.alloc(i32, p.rows);
        for ([_]*[]f64{ &p.point, &p.half, &p.down, &p.up }) |x| x.* = try p.a.alloc(f64, p.count());
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        for ([_][]u16{ p.kp, p.vp, p.q }) |x| for (x) |*y| {
            y.* = c.toBf16(rng.float(f32) * 2 - 1);
        };
        for (p.hi, 0..) |*h, r| h.* = @intCast(p.p0 + r);
        if (!p.v.global and p.rows > 16) { // an image's rows: positions [at, at + len) all see the whole image
            const at = p.rows / 4;
            @memset(p.hi[at..][0 .. p.rows / 3], @intCast(p.p0 + at + p.rows / 3 - 1));
        }
        switch (fill) {
            .random => if (p.v.case == .spread or p.v.case == .cutless) p.spread(rng),
            .nonfinite => {
                // keys at p0 every row sees; a NaN key at p0 + 2 that rows 0 and 1 mask
                p.q[(0 * p.nq() + 1) * dh_ + 3] = c.toBf16(std.math.nan(f32));
                p.vp[p.p0 * dh_ + 5] = c.toBf16(std.math.inf(f32));
                p.vp[p.p0 * dh_ + 6] = c.toBf16(-std.math.inf(f32));
                if (p.p0 + 2 < n) p.kp[(p.p0 + 2) * dh_] = c.toBf16(std.math.nan(f32));
            },
            .stress => {
                // every score 0, so every weight 1; the values every row sees cancel but for four ones
                @memset(p.q, 0);
                @memset(p.kp, 0);
                @memset(p.vp, 0);
                const first = if (p.v.global) 0 else (p.p0 + p.rows) -| cfg.window;
                for (0..p.nk()) |h| for (first..p.p0 + 1) |j| for (0..dh_) |e| {
                    p.vp[(h * n + j) * dh_ + e] = c.toBf16(stressValue(j - first, p.p0 + 1 - first));
                };
            },
        }
        @memset(p.q[p.rows * p.nq() * dh_ ..], nan16); // a tile's rows past the chunk: read, never written
        try p.bind();
        return true;
    }

    /// Scores q0 k0 alone, so P = e^(s - max) spreads over about [2^-6, 1]; values in [0, 1) add the cut's bias up.
    fn spread(p: *Probe, rng: std.Random) void {
        const dh_ = p.dh();
        for (0..p.rows * p.nq()) |r| {
            @memset(p.q[r * dh_ ..][0..dh_], 0);
            p.q[r * dh_] = c.toBf16(0.5 + rng.float(f32));
        }
        for (0..p.nk() * p.npos()) |j| {
            @memset(p.kp[j * dh_ ..][0..dh_], 0);
            p.kp[j * dh_] = c.toBf16(-3 * rng.float(f32));
            for (p.vp[j * dh_ ..][0..dh_]) |*x| x.* = c.toBf16(rng.float(f32));
        }
    }

    /// The caches as the engine holds them: a sliding layer's ring, a global layer's linear slots; then the call.
    fn bind(p: *Probe) !void {
        const dh_ = p.dh();
        const n = p.npos();
        const sliding = !p.v.global;
        p.cap = if (sliding) ring else std.mem.alignForward(usize, n, 256);
        const kc = try p.a.alloc(u16, p.nk() * p.cap * dh_);
        defer p.a.free(kc);
        const vc = try p.a.alloc(u16, p.nk() * p.cap * dh_);
        defer p.a.free(vc);
        // past the last key: NaN (prompt never reads it in); zeros as caches start (decode reads 32 value rows whole)
        if (p.v.kind == .prompt) {
            @memset(kc, nan16);
            @memset(vc, nan16);
        } else {
            @memset(kc, 0);
            @memset(vc, 0);
        }
        for (0..p.nk()) |h| for (0..n) |j| {
            const slot = if (sliding) j % ring else j;
            @memcpy(kc[(h * p.cap + slot) * dh_ ..][0..dh_], p.kp[(h * n + j) * dh_ ..][0..dh_]);
            @memcpy(vc[(h * p.cap + slot) * dh_ ..][0..dh_], p.vp[(h * n + j) * dh_ ..][0..dh_]);
        };
        p.refs[0] = try p.rig.input(std.mem.sliceAsBytes(p.q));
        p.refs[1] = try p.rig.input(std.mem.sliceAsBytes(kc));
        p.refs[2] = try p.rig.input(std.mem.sliceAsBytes(vc));
        p.refs[3] = try p.rig.input(std.mem.sliceAsBytes(p.hi));
        if (p.v.kind == .decode) {
            // forward.zig's meta at the sub-batch's row: the chunks and the rows, which the merge reads
            var meta: [tf.gemma4.state.meta_ints]i32 = @splat(0);
            meta[1] = @intCast(p.chunks());
            meta[3] = @intCast(p.rows);
            p.refs[4] = try p.rig.input(std.mem.asBytes(&meta));
            const slots = p.nq() * p.rows * p.chunks();
            p.parts[0] = try p.rig.scratch(slots * 4);
            p.parts[1] = try p.rig.scratch(slots * 4);
            p.parts[2] = try p.rig.scratch(slots * dh_ * 4);
            p.parts[3] = try p.rig.scratch(4);
        }
        p.out = try p.rig.output(p.count() * 2);
    }

    pub fn count(p: *Probe) usize {
        return p.rows * p.nq() * p.dh();
    }

    pub fn run(p: *Probe) !void {
        const pass = p.rig.begin();
        const e = pass.e;
        const dh_ = p.dh();
        e.setPipeline(p.pipe);
        for (p.refs[0..3], 0..) |r, i| e.setBuffer(r.buf, r.off, i);
        if (p.v.kind == .prompt) {
            // as prompt.zig: (tiles, query heads) groups, 32 threads a 64-column simdgroup
            e.setBuffer(p.out.buf, p.out.off, 3);
            e.setValue([3]i32{ @intCast(p.rows), @intCast(p.p0), @intCast(p.cap) }, 4);
            e.setBuffer(p.refs[3].buf, p.refs[3].off, 5);
            e.dispatchGroups(.{ .width = (p.rows + rt - 1) / rt, .height = p.nq() }, .{ .width = 32 * dh_ / 64 });
        } else {
            // as forward.zig: (chunks, key heads, tiles of 16 / G rows) groups, 32 threads a 128-dim slice; the merge
            e.setValue([4]i32{ @intCast(p.rows), @intCast(p.p0), @intCast(p.cap), @intCast(p.chunks()) }, 3);
            for (p.parts[0..3], 4..) |r, i| e.setBuffer(r.buf, r.off, i);
            const grp = p.nq() / p.nk();
            e.dispatchGroups(.{ .width = p.chunks(), .height = p.nk(), .depth = (p.rows * grp + 15) / 16 }, .{ .width = 32 * dh_ / 128 });
            e.setPipeline(p.merge.?);
            for (p.parts[0..3], 0..) |r, i| e.setBuffer(r.buf, r.off, i);
            e.setBuffer(p.refs[4].buf, p.refs[4].off, 3);
            e.setBuffer(p.out.buf, p.out.off, 4);
            e.setBuffer(p.parts[3].buf, p.parts[3].off, 5);
            e.setValue(@as(i32, 0), 6); // no o-projection sums: they follow the bf16 output, which is all checked here
            e.dispatchGroups(.{ .width = 1, .height = p.nq(), .depth = p.rows }, .{ .width = 32 });
        }
        try p.rig.finish(pass);
    }

    /// Every head's float64 point, half width and the cut's sides over the keys its row attends.
    pub fn reference(p: *Probe, want: []f64, mag: []f64) !void {
        const dh_ = p.dh();
        const n = p.npos();
        const grp = p.nq() / p.nk();
        const buf = try p.a.alloc(f64, 2 * n * dh_ + 2 * n + dh_);
        defer p.a.free(buf);
        const kf = buf[0 .. n * dh_];
        const vf = buf[n * dh_ .. 2 * n * dh_];
        const w = buf[2 * n * dh_ ..][0..n];
        const tau = buf[2 * n * dh_ + n ..][0..n];
        const qv = buf[buf.len - dh_ ..];
        for (0..p.nk()) |hk| {
            for (kf, vf, 0..) |*k, *v, i| {
                k.* = c.fromBf16(p.kp[hk * n * dh_ + i]);
                v.* = c.fromBf16(p.vp[hk * n * dh_ + i]);
            }
            for (0..p.rows) |r| {
                const keys = p.keysOf(r);
                const len = keys[1] + 1 - keys[0];
                const f = p.form(r);
                const ks = kf[keys[0] * dh_ ..];
                const vs = vf[keys[0] * dh_ ..];
                for (hk * grp..(hk + 1) * grp) |h| {
                    const o = (r * p.nq() + h) * dh_;
                    for (qv, 0..) |*x, e| x.* = c.fromBf16(p.q[o + e]);
                    ar.head(qv, ks, vs, len, dh_, 1, f, w, tau, want[o..][0..dh_], p.half[o..][0..dh_]);
                    ar.cutSides(f, w, tau, vs, len, dh_, p.down[o..][0..dh_], p.up[o..][0..dh_]);
                }
            }
        }
        @memcpy(mag, p.half);
        @memcpy(p.point, want);
    }

    /// The range's half width and the cut's wider side plus bf16's rounding; gate 4's exact sums keep the exact bound.
    pub fn bounds(p: *Probe, want: []const f64, _: []const f64, allow: []f64) !void {
        const s = p.spec();
        for (allow, want, p.half, p.down, p.up) |*l, w, h, d, u| {
            if (p.stress) {
                l.* = c.boundExact(s, w);
            } else if (!std.math.isFinite(w)) {
                l.* = 0;
            } else if (!std.math.isFinite(h)) {
                l.* = std.math.floatMax(f64); // scores too far apart for the range's first order: any finite value
            } else l.* = h + @max(d, u) + 0x1p-8 * (@abs(w) + h + @max(d, u)) + 0x1p-133;
        }
    }

    /// Each bf16 output within its range's ends as rounded, the cut on its own side; gate 4 and wide ranges are open.
    pub fn intervals(p: *Probe, low: []f64, high: []f64) !void {
        for (low, high, p.point, p.half, p.down, p.up) |*l, *hi, pt, h, d, u| {
            const open = p.stress or !std.math.isFinite(pt) or !std.math.isFinite(h);
            const e = if (open) [2]f64{ -std.math.floatMax(f64), std.math.floatMax(f64) } else ar.ends(pt, h, d, u);
            l.* = e[0];
            hi.* = e[1];
        }
    }

    /// Gate 4: every weight 1, so each output is its row's values summed in `acc` over their count.
    pub fn simulate(p: *Probe, acc: c.Accum, order: g.Order, out: []f64) !void {
        if (!p.stress) return error.StressRequiresEqualScores;
        const dh_ = p.dh();
        const n = p.npos();
        const grp = p.nq() / p.nk();
        var terms: std.ArrayList(f64) = .empty;
        defer terms.deinit(p.a);
        for (0..p.rows) |r| {
            const keys = p.keysOf(r);
            for (0..p.nk()) |hk| for (0..dh_) |e| {
                terms.clearRetainingCapacity();
                for (keys[0]..keys[1] + 1) |j| try terms.append(p.a, c.fromBf16(p.vp[(hk * n + j) * dh_ + e]));
                const v = bf(contract.rows.sumIn(acc, order, terms.items) / @as(f64, @floatFromInt(terms.items.len)));
                for (hk * grp..(hk + 1) * grp) |h| out[(r * p.nq() + h) * dh_ + e] = v;
            };
        }
    }
};

const vt = g.hooks(Probe);

/// Every variant and control, as tf-kernel-contract collects them.
pub const Suite = struct {
    a: std.mem.Allocator,
    device: mtl.Device,
    queue: mtl.Queue,
    kernels: *std.ArrayList(g.Kernel),
    probes: std.ArrayList(*Probe) = .empty,

    /// Only on the tensor units: before the M5 nax.h's fragments keep P whole, a path these ranges do not state.
    pub fn collect(s: *Suite, only: ?[]const u8) !void {
        if (!s.device.tensorUnits()) return;
        for (variants, names) |v, name| {
            if (only) |o| if (std.mem.indexOf(u8, name, o) == null) continue;
            const p = try s.a.create(Probe);
            errdefer s.a.destroy(p);
            p.* = try Probe.init(s.a, s.device, s.queue, v, name);
            errdefer p.deinit();
            try s.probes.append(s.a, p);
            const control = v.case == .cutless or v.case == .edit;
            try s.kernels.append(s.a, .{ .spec = p.spec(), .ctx = p, .vt = &vt, .control = if (control) .shapes else null });
        }
    }

    pub fn deinit(s: *Suite) void {
        for (s.probes.items) |p| {
            p.deinit();
            s.a.destroy(p);
        }
        s.probes.deinit(s.a);
    }
};
