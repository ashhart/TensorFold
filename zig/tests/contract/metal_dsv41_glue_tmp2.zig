//! DeepSeek V4.1's glue part two on Metal: fp4pack, fp4rt, pool2, cand, hcmix, hcpre, hcpost against float64.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const sources = @import("kernel_sources");
const contract = @import("contract");
const rig_ = @import("rig.zig");
const ref = @import("dsv41_glue_tmp_ref.zig");
const aux = @import("metal_dsv41_glue_tmp2_aux.zig");

const c = contract.spec;
const g = contract.gates;
const gl = tf.dsv41_metal.glue_tmp;
const base = "dsv41 glue_tmp 2 ";
const tg = 256;
const dim: usize = 5120; // the hyper-connection stream: hc_mult * hidden
const iters: u32 = 20;
const eps: f64 = 1e-20;
const hc_eps: f64 = 1e-6;
const alpha: f64 = 2.0;

pub const Fam = enum { pack, rt, pool, cand, mix, pre, post };

fn noSums(_: c.Shape) usize {
    return 0;
}

/// A lane's chain over every 256th element, the 5-level butterfly, the eight partials, then the divides after.
fn mixDepth(s: c.Shape) usize {
    return s.k / tg + 14;
}

/// The collapse: four products, three adds, one round to bf16.
fn preDepth(_: c.Shape) usize {
    return 7;
}

/// The expand: five products, four adds, one round to bf16.
fn postDepth(_: c.Shape) usize {
    return 9;
}

const pack_rule = "a NaN packs as the top code, an infinity takes the E8M0 byte 255 and packs every non-zero as 0";
const rt_rule = "a NaN or an infinity packs to a finite code: the maximum ignores a NaN, an infinity saturates";
const pool_rule = "an odd position pools its two slots as the reference's softmax does, an even position's row is not written," ++
    " and the carry holds every slot's last value";
const cand_rule = "a NaN key never wins a maximum, the last visible block is pinned, ties go to the lower block";
const mix_why = "the norm's sum of squares and the 24 matvecs, each lane chain, butterfly and partial in its fixed order";

pub fn spec(f: Fam, name: []const u8) c.Spec {
    const t16: c.Tile = .{ .size = 16 };
    const blocks: c.Tile = .{ .size = 256, .step = 32, .min = 32 };
    return switch (f) {
        .pack => .{ .name = name, .accum = .none, .depth = noSums, .out = .raw, .nan = .{ .stated = pack_rule }, .tiles = .{ t16, blocks, .{} }, .served = &.{
            .{ .rows = 1, .n = 512, .k = 1 }, .{ .rows = 16, .n = 512, .k = 1 }, .{ .rows = 5, .n = 256, .k = 1 },
        } },
        .rt => .{ .name = name, .accum = .none, .depth = noSums, .out = .bf16, .nan = .{ .stated = rt_rule }, .tiles = .{ t16, .{ .size = 256, .step = 16, .min = 16 }, .{} }, .served = &.{
            .{ .rows = 1, .n = 512, .k = 1 }, .{ .rows = 16, .n = 512, .k = 1 }, .{ .rows = 5, .n = 64, .k = 1 },
        } },
        .pool => .{ .name = name, .accum = .none, .depth = noSums, .out = .bf16, .nan = .{ .stated = pool_rule }, .tiles = .{ t16, .{ .size = 256, .step = 64, .min = 64 }, .{} }, .served = &.{
            .{ .rows = 1, .n = 512, .k = 1 }, .{ .rows = 16, .n = 512, .k = 1 }, .{ .rows = 5, .n = 64, .k = 1 }, .{ .rows = 3, .n = 33, .k = 1 },
        } },
        .cand => .{ .name = name, .accum = .none, .depth = noSums, .out = .raw, .nan = .{ .stated = cand_rule }, .tiles = .{ t16, .{}, .{ .size = 4096, .step = 64, .min = 64 } }, .served = &.{
            .{ .rows = 1, .n = 2048, .k = 64 },   .{ .rows = 4, .n = 2048, .k = 256 }, .{ .rows = 3, .n = 100, .k = 320 },
            .{ .rows = 2, .n = 2048, .k = 4096 },
        } },
        .mix => .{ .name = name, .accum = .f32, .depth = mixDepth, .ops = 260, .out = .f32, .tiles = .{ t16, .{}, .{} }, .served = &.{
            .{ .rows = 16, .n = 1, .k = dim }, .{ .rows = 1, .n = 1, .k = dim }, .{ .rows = 4, .n = 1, .k = dim },
        }, .sums = .{ .why = mix_why } },
        .pre => .{ .name = name, .accum = .f32, .depth = preDepth, .out = .bf16, .tiles = .{ t16, .{}, .{} }, .served = &.{
            .{ .rows = 16, .n = 1, .k = dim }, .{ .rows = 1, .n = 1, .k = dim }, .{ .rows = 5, .n = 1, .k = dim },
        }, .sums = .{ .why = "four weighted streams added in stream order 0, 1, 2, 3, one round to bf16" } },
        .post => .{ .name = name, .accum = .f32, .depth = postDepth, .out = .bf16, .tiles = .{ t16, .{}, .{} }, .served = &.{
            .{ .rows = 16, .n = 1, .k = dim }, .{ .rows = 1, .n = 1, .k = dim }, .{ .rows = 3, .n = 1, .k = dim },
        }, .sums = .{ .why = "the residual through comb in j order, then the post-weighted branch last, one round to bf16" } },
    };
}

fn pow2(e: i32) f32 {
    return std.math.pow(f32, 2, @floatFromInt(e));
}

fn spread(rng: std.Random, lo: i32, hi: i32) f32 {
    return (rng.float(f32) * 2 - 1) * pow2(rng.intRangeAtMost(i32, lo, hi));
}

fn at(r: rig_.Ref, extra: usize) gl.Ref {
    return .{ .buf = r.buf, .off = r.off + extra };
}

/// The pipeline of a glue_tmp kernel by its source name.
fn pipe(pl: *const gl.Pipelines, name: []const u8) mtl.Pipeline {
    for (gl.names, 0..) |n, i| if (std.mem.eql(u8, n, name)) return pl.p[i];
    unreachable;
}

const vt_exact = g.hooks(Proj);
const vt_plain = blk: {
    var v = g.hooks(Proj);
    v.check = null; // the hc families are checked against the float64 bound, not a stated rule
    break :blk v;
};

pub const Proj = struct {
    a: std.mem.Allocator,
    arena: std.heap.ArenaAllocator,
    rig: rig_.Rig,
    fam: Fam,
    ctl: u8,
    name: []const u8,
    pipes: gl.Pipelines,
    shape: c.Shape = undefined,
    x: []u16 = &.{},
    fnw: []f32 = &.{},
    scale: [3]f32 = .{ 1, 1, 1 },
    basew: []f32 = &.{},
    mixf: []f32 = &.{},
    postf: []f32 = &.{},
    comb: []f32 = &.{},
    res: []u16 = &.{},
    kv: []f32 = &.{},
    sc: []f32 = &.{},
    pos: []i32 = &.{},
    state0: []f32 = &.{},
    scores: []f32 = &.{},
    vis: []i32 = &.{},
    dest: []i32 = &.{},
    plants: usize = 0,
    in: [4]rig_.Ref = undefined,
    out: [3]rig_.Ref = undefined,
    out_len: [3]usize = .{ 0, 0, 0 },
    scratch: rig_.Ref = undefined,
    no: usize = 0,
    msg: [300]u8 = undefined,

    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, f: Fam, ctl: u8, name: []const u8) !Proj {
        var p: Proj = .{ .a = a, .arena = std.heap.ArenaAllocator.init(a), .rig = .{ .a = a, .device = device, .queue = queue }, .fam = f, .ctl = ctl, .name = name, .pipes = undefined };
        if (ctl == 0) {
            p.pipes = try gl.Pipelines.load(device, sources.dsv41.glue_tmp);
        } else {
            const text = try std.fmt.allocPrint(a, "#define TF_CTL {d}\n{s}", .{ ctl, sources.dsv41.glue_tmp });
            defer a.free(text);
            p.pipes = try gl.Pipelines.load(device, text);
        }
        return p;
    }

    pub fn deinit(p: *Proj) void {
        p.arena.deinit();
        p.pipes.deinit();
        p.rig.deinit();
    }

    /// Control 2 is a wrong value at every family; control 4 writes past the end but for fp4rt's, which lands inside.
    pub fn kernel(p: *Proj) g.Kernel {
        const control: ?g.Gate = switch (p.ctl) {
            0 => null,
            2 => .shapes,
            else => if (p.fam == .rt) .shapes else .repeat,
        };
        const exact = p.fam == .pack or p.fam == .rt or p.fam == .pool or p.fam == .cand;
        return .{ .spec = spec(p.fam, p.name), .ctx = p, .vt = if (exact) &vt_exact else &vt_plain, .control = control };
    }

    fn blocks(p: *const Proj) usize {
        return p.shape.n / 32;
    }

    pub fn count(p: *Proj) usize {
        const s = p.shape;
        return switch (p.fam) {
            .pack => s.rows * p.blocks() * 17,
            .rt => s.rows * s.n,
            .pool => s.rows * s.n + 4 * s.n,
            .cand => s.rows * s.k,
            .mix => s.rows * 24,
            .pre => s.rows * s.k,
            .post => s.rows * 4 * s.k,
        };
    }

    fn output(p: *Proj, len: usize) !rig_.Ref {
        p.out[p.no] = try p.rig.output(len);
        p.out_len[p.no] = len;
        p.no += 1;
        return p.out[p.no - 1];
    }

    /// Rows of destination: a permutation of the rows with every fifth one skipped (-1).
    fn destinations(p: *Proj, a: std.mem.Allocator) ![]i32 {
        const d = try a.alloc(i32, p.shape.rows);
        for (d, 0..) |*o, r| o.* = if (r % 5 == 4) -1 else @intCast(p.shape.rows - 1 - r);
        return d;
    }

    pub fn prepare(p: *Proj, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        const blk: usize = switch (p.fam) {
            .pack => 32,
            .rt => 16,
            else => 1,
        };
        if (shape.rows == 0 or shape.n == 0 or shape.n % blk != 0) return false;
        if (p.fam == .cand and shape.k == 0) return false;
        if ((p.fam == .mix or p.fam == .pre or p.fam == .post) and shape.k != dim) return false;
        p.rig.reset();
        _ = p.arena.reset(.retain_capacity);
        const a = p.arena.allocator();
        p.shape = shape;
        p.plants = 0;
        p.no = 0;
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        const nf = fill == .nonfinite;
        const n = shape.n;
        const nr = shape.rows;
        switch (p.fam) {
            .pack, .rt => {
                p.x = try a.alloc(u16, nr * n);
                ref.base.fillQuant(if (p.fam == .rt) .fp4_e4m3 else .fp4_e8m0, p.x, blk, seed);
                if (nf) p.plants = ref.base.plant(p.x, nr, n, 0, n);
                p.in[0] = try p.rig.input(std.mem.sliceAsBytes(p.x));
                p.dest = try p.destinations(a);
                p.in[1] = try p.rig.input(std.mem.sliceAsBytes(p.dest));
                if (p.fam == .pack) {
                    _ = try p.output(nr * p.blocks() * 16);
                    _ = try p.output(nr * p.blocks());
                } else {
                    _ = try p.output(nr * n * 2);
                }
            },
            .pool => try p.preparePool(a, rng, nf),
            .cand => try p.prepareCand(a, rng, nf),
            .mix, .pre, .post => try p.prepareHc(a, rng, fill),
        }
        return true;
    }

    /// The ratio-2 pool: the carry's start, positions that alternate from an even one, and gate 1's plants.
    fn preparePool(p: *Proj, a: std.mem.Allocator, rng: std.Random, nf: bool) !void {
        const n = p.shape.n;
        const nr = p.shape.rows;
        p.kv = try a.alloc(f32, nr * n);
        p.sc = try a.alloc(f32, nr * n);
        p.pos = try a.alloc(i32, nr);
        p.state0 = try a.alloc(f32, 4 * n);
        const p0: i32 = @intCast(rng.uintLessThan(u32, 1 << 20) & ~@as(u32, 1));
        for (p.pos, 0..) |*o, r| o.* = p0 + @as(i32, @intCast(r));
        for (p.state0[0 .. 2 * n]) |*o| o.* = spread(rng, -4, 4);
        for (p.state0[2 * n ..]) |*o| o.* = spread(rng, -3, 3);
        for (0..nr) |r| for (0..n) |d| {
            p.kv[r * n + d] = spread(rng, -4, 4);
            p.sc[r * n + d] = switch (rng.uintLessThan(u32, 4)) {
                0, 1 => spread(rng, -3, 3),
                2 => (rng.float(f32) * 2 - 1) * 80,
                else => @as(f32, @floatFromInt(rng.intRangeAtMost(i32, -2, 2))),
            };
        };
        if (nf and nr > 3) {
            // rows 0 and 2 fill slot 0, rows 1 and 3 pool: five columns whose pooled value is non-finite
            const put = [_]struct { r: usize, d: usize, kv: f32, sc: f32 }{
                .{ .r = 0, .d = 1, .kv = std.math.nan(f32), .sc = 0.5 },
                .{ .r = 0, .d = 2, .kv = std.math.inf(f32), .sc = 0.5 },
                .{ .r = 0, .d = 3, .kv = -std.math.inf(f32), .sc = 0.5 },
                .{ .r = 2, .d = 5, .kv = std.math.inf(f32), .sc = 0.25 },
                .{ .r = 2, .d = 6, .kv = -std.math.inf(f32), .sc = 0.25 },
            };
            for (put) |q| {
                p.kv[q.r * n + q.d] = q.kv;
                p.sc[q.r * n + q.d] = q.sc;
            }
            for ([_]usize{ 1, 2, 3, 5, 6 }) |d| {
                p.kv[n + d] = 1;
                p.sc[n + d] = 0.5;
                p.kv[3 * n + d] = 1;
                p.sc[3 * n + d] = 0.25;
            }
            p.plants = 5;
        }
        p.in[0] = try p.rig.input(std.mem.sliceAsBytes(p.kv));
        p.in[1] = try p.rig.input(std.mem.sliceAsBytes(p.sc));
        p.in[2] = try p.rig.input(std.mem.sliceAsBytes(p.pos));
        _ = try p.output(nr * n * 2);
        _ = try p.output(4 * n * 4);
    }

    /// The candidate blocks: rows of ties, of distinct values and of zeros, with partial and empty visibility.
    fn prepareCand(p: *Proj, a: std.mem.Allocator, rng: std.Random, nf: bool) !void {
        const nb = p.shape.k;
        const keys = 8 * nb;
        const nr = p.shape.rows;
        p.scores = try a.alloc(f32, nr * keys);
        p.vis = try a.alloc(i32, nr);
        for (0..nr) |r| {
            const row = p.scores[r * keys ..][0..keys];
            const vis: usize = if (r < 4) keys else switch (r % 8) {
                1 => keys / 2 + 1,
                5 => 0,
                else => keys,
            };
            p.vis[r] = @intCast(vis);
            for (row, 0..) |*v, i| v.* = if (i >= vis) -std.math.inf(f32) else switch (r % 4) {
                0 => @as(f32, @floatFromInt(rng.uintLessThan(u32, 512))) / 16 - 16,
                1 => 1.5, // every block ties: the tie rule is what the mask turns on
                2 => -@as(f32, @floatFromInt(i / 3)),
                else => switch (rng.uintLessThan(u32, 8)) {
                    0 => 1.0,
                    1...4 => 0.0,
                    else => -0.0,
                },
            };
        }
        if (nf) {
            const put = [_]f32{ std.math.nan(f32), std.math.inf(f32), -std.math.inf(f32), std.math.nan(f32) };
            for (put, 0..) |v, i| if (i < nr and p.vis[i] > 0) {
                const row = p.scores[i * keys ..][0..keys];
                for (row[0..@min(8, row.len)]) |*o| o.* = v;
                p.plants += 1;
            };
        }
        p.in[0] = try p.rig.input(std.mem.sliceAsBytes(p.scores));
        p.in[1] = try p.rig.input(std.mem.sliceAsBytes(p.vis));
        p.scratch = try p.rig.scratch(nr * nb * 4);
        _ = try p.output(nr * nb);
    }

    /// The hyper-connection families: the stream, the mixes' weights, and gate 4's cancelling integers.
    fn prepareHc(p: *Proj, a: std.mem.Allocator, rng: std.Random, fill: g.Fill) !void {
        const nr = p.shape.rows;
        const stream: usize = if (p.fam == .pre) 4 else 1;
        p.x = try a.alloc(u16, nr * stream * dim);
        p.fnw = try a.alloc(f32, if (p.fam == .mix) 24 * dim else 1);
        p.basew = try a.alloc(f32, if (p.fam == .mix) 24 else 1);
        p.mixf = try a.alloc(f32, nr * (if (p.fam == .mix) @as(usize, 1) else 4));
        p.postf = try a.alloc(f32, if (p.fam == .post) nr * 4 else 1);
        p.comb = try a.alloc(f32, if (p.fam == .post) nr * 16 else 1);
        p.res = try a.alloc(u16, if (p.fam == .post) nr * 4 * dim else 1);
        aux.fillHc(.{ .kind = switch (p.fam) {
            .mix => .mix,
            .pre => .pre,
            else => .post,
        }, .rows = nr, .x = p.x, .fnw = p.fnw, .basew = p.basew, .mixf = p.mixf, .postf = p.postf, .comb = p.comb, .res = p.res, .scale = &p.scale, .plants = &p.plants }, rng, fill);
        const spare = try p.rig.input(std.mem.sliceAsBytes(p.scale[0..1]));
        p.in[0] = try p.rig.input(std.mem.sliceAsBytes(p.x));
        p.in[1] = switch (p.fam) {
            .mix => try p.rig.input(std.mem.sliceAsBytes(p.fnw)),
            .pre => try p.rig.input(std.mem.sliceAsBytes(p.mixf)),
            else => try p.rig.input(std.mem.sliceAsBytes(p.res)),
        };
        p.in[2] = switch (p.fam) {
            .mix => try p.rig.input(std.mem.sliceAsBytes(&p.scale)),
            .post => try p.rig.input(std.mem.sliceAsBytes(p.postf)),
            else => spare,
        };
        p.in[3] = switch (p.fam) {
            .mix => try p.rig.input(std.mem.sliceAsBytes(p.basew)),
            .post => try p.rig.input(std.mem.sliceAsBytes(p.comb)),
            else => spare,
        };
        if (p.fam == .mix) {
            _ = try p.output(nr * 4 * 4);
            _ = try p.output(nr * 4 * 4);
            _ = try p.output(nr * 16 * 4);
        } else if (p.fam == .pre) {
            _ = try p.output(nr * dim * 2);
        } else {
            _ = try p.output(nr * 4 * dim * 2);
        }
    }

    fn reset(p: *Proj) void {
        if (p.fam != .pool) return;
        @memset(p.rig.bytes(p.out[0], p.out_len[0]), 0xff);
        @memcpy(p.rig.bytes(p.out[1], p.out_len[1]), std.mem.sliceAsBytes(p.state0));
    }

    /// Rows r0 to r0 + m of the prepared call as one launch.
    pub fn encode(p: *Proj, e: anytype, r0: usize, m: usize) void {
        const s = p.shape;
        const pl = &p.pipes;
        const nr: u32 = @intCast(m);
        const n: u32 = @intCast(s.n);
        switch (p.fam) {
            .pack => {
                e.setPipeline(pipe(pl, "tf_dsv41_g_fp4pack"));
                e.setBuffer(p.in[0].buf, p.in[0].off + r0 * s.n * 2, 0);
                e.setBuffer(p.out[0].buf, p.out[0].off, 1);
                e.setBuffer(p.out[1].buf, p.out[1].off, 2);
                e.setBuffer(p.in[1].buf, p.in[1].off + r0 * 4, 3);
                e.setValue([6]u32{ n / 32, n, n / 2, n / 32, 0, 1 }, 4);
                e.dispatchThreads(mtl.Size.of(n / 32, m, 1), mtl.Size.of(32, 1, 1));
            },
            .rt => {
                e.setPipeline(pipe(pl, "tf_dsv41_g_fp4rt"));
                e.setBuffer(p.in[0].buf, p.in[0].off + r0 * s.n * 2, 0);
                e.setBuffer(p.out[0].buf, p.out[0].off, 1);
                e.setBuffer(p.in[1].buf, p.in[1].off + r0 * 4, 2);
                e.setValue([3]u32{ n / 16, n, n }, 3);
                e.dispatchThreads(mtl.Size.of(n / 16, m, 1), mtl.Size.of(32, 1, 1));
            },
            .pool => {
                e.setPipeline(pipe(pl, "tf_dsv41_g_pool2"));
                e.setBuffer(p.in[0].buf, p.in[0].off + r0 * s.n * 4, 0);
                e.setBuffer(p.in[1].buf, p.in[1].off + r0 * s.n * 4, 1);
                e.setBuffer(p.out[1].buf, p.out[1].off, 2);
                e.setBuffer(p.in[2].buf, p.in[2].off + r0 * 4, 3);
                e.setBuffer(p.out[0].buf, p.out[0].off + r0 * s.n * 2, 4);
                e.setValue([2]u32{ nr, n }, 5);
                e.dispatchThreads(mtl.Size.of(s.n, 1, 1), mtl.Size.of(32, 1, 1));
            },
            .cand => {
                e.setPipeline(pipe(pl, "tf_dsv41_g_cand"));
                e.setBuffer(p.in[0].buf, p.in[0].off + r0 * s.k * 8 * 4, 0);
                e.setBuffer(p.in[1].buf, p.in[1].off + r0 * 4, 1);
                e.setBuffer(p.scratch.buf, p.scratch.off + r0 * s.k * 4, 2);
                e.setBuffer(p.out[0].buf, p.out[0].off + r0 * s.k, 3);
                e.setValue([3]u32{ @intCast(s.k * 8), @intCast(s.k), n }, 4);
                e.dispatchGroups(mtl.Size.of(m, 1, 1), mtl.Size.of(tg, 1, 1));
            },
            .mix => gl.hcmix(pl, e, at(p.in[0], r0 * dim * 2), .{ at(p.in[1], 0), at(p.in[2], 0), at(p.in[3], 0) }, .{ at(p.out[0], r0 * 16), at(p.out[1], r0 * 16), at(p.out[2], r0 * 64) }, nr, .{ .dim = dim, .iters = iters, .eps = @floatCast(eps), .hc_eps = @floatCast(hc_eps), .alpha = @floatCast(alpha) }),
            .pre => gl.hcpre(pl, e, at(p.in[0], r0 * 4 * dim * 2), at(p.in[1], r0 * 16), at(p.out[0], r0 * dim * 2), nr, dim),
            .post => gl.hcpost(pl, e, at(p.in[0], r0 * dim * 2), at(p.in[1], r0 * 4 * dim * 2), at(p.in[2], r0 * 16), at(p.in[3], r0 * 64), at(p.out[0], r0 * 4 * dim * 2), nr, dim),
        }
    }

    pub fn run(p: *Proj) !void {
        p.reset();
        const pass = p.rig.begin();
        p.encode(pass.e, 0, p.shape.rows);
        try p.rig.finish(pass);
    }

    pub fn reference(p: *Proj, want: []f64, mag: []f64) !void {
        return p.compute(.none, .sequential, want, mag);
    }

    pub fn simulate(p: *Proj, acc: c.Accum, order: g.Order, out: []f64) !void {
        const scratch = try p.arena.allocator().alloc(f64, @max(dim, p.count()));
        return p.compute(acc, order, out, scratch);
    }

    /// Every output of the prepared call in the order the rig gathers its buffers; float64, or gate 4's sums.
    fn compute(p: *Proj, acc: c.Accum, order: g.Order, want: []f64, mag: []f64) !void {
        const s = p.shape;
        const a = p.arena.allocator();
        @memset(want, 0);
        @memset(mag, 0);
        switch (p.fam) {
            .mix => {
                const scratch = try a.alloc(f64, dim);
                for (0..s.rows) |r| {
                    const o = ref.mix(.{ .x = p.x[r * dim ..][0..dim], .fnw = p.fnw, .scale = p.scale, .base = p.basew, .iters = iters, .eps = eps, .hc_eps = hc_eps, .alpha = alpha }, acc, order, scratch);
                    for (0..4) |j| {
                        want[r * 4 + j] = o.pre[j];
                        mag[r * 4 + j] = o.mpre[j];
                        want[s.rows * 4 + r * 4 + j] = o.post[j];
                        mag[s.rows * 4 + r * 4 + j] = o.mpost[j];
                    }
                    for (0..16) |j| {
                        want[2 * s.rows * 4 + r * 16 + j] = o.comb[j];
                        mag[2 * s.rows * 4 + r * 16 + j] = o.mcomb[j];
                    }
                }
            },
            .pre => for (0..s.rows) |r| for (0..dim) |q| {
                const v = ref.pre(p.mixf[r * 4 ..][0..4], p.x[r * 4 * dim ..][0 .. 4 * dim], dim, q, acc, order);
                want[r * dim + q] = v.v;
                mag[r * dim + q] = v.mag;
            },
            .post => for (0..s.rows) |r| for (0..4) |k| for (0..dim) |q| {
                const v = ref.post(p.comb[r * 16 ..][0..16], p.postf[r * 4 ..][0..4], p.res[r * 4 * dim ..][0 .. 4 * dim], p.x[r * dim ..][0..dim], dim, k, q, acc, order);
                want[r * 4 * dim + k * dim + q] = v.v;
                mag[r * 4 * dim + k * dim + q] = v.mag;
            },
            else => {},
        }
    }

    /// The packings and the candidate mask against the math byte for byte; the pool against float64 by column.
    pub fn check(p: *Proj, gate: g.Gate) !?[]const u8 {
        const s = p.shape;
        const a = p.arena.allocator();
        return switch (p.fam) {
            .pack => aux.pack(s, p.blocks(), p.x, p.dest, p.rig.bytes(p.out[0], s.rows * p.blocks() * 16), p.rig.bytes(p.out[1], s.rows * p.blocks()), p.plants, gate, a, &p.msg),
            .rt => aux.rt(s, p.x, p.dest, p.rig.bytes(p.out[0], s.rows * s.n * 2), p.plants, gate, a, &p.msg),
            .pool => aux.pool(s, p.state0, p.kv, p.sc, p.pos, p.rig.bytes(p.out[0], s.rows * s.n * 2), p.rig.bytes(p.out[1], 4 * s.n * 4), p.plants, gate, &p.msg),
            .cand => aux.cand(s, p.scores, p.vis, p.rig.bytes(p.out[0], s.rows * s.k), p.plants, gate, a, &p.msg),
            else => null,
        };
    }
};

/// Every served kernel and its two planted controls, plus the launch-split probe after the gates.
pub const Suite = struct {
    a: std.mem.Allocator,
    probes: std.ArrayList(*Proj) = .empty,

    const E = struct { f: Fam, ctl: u8, name: []const u8 };
    const list = [_]E{
        .{ .f = .pack, .ctl = 0, .name = base ++ "fp4 pack tf_dsv41_g_fp4pack" },
        .{ .f = .pack, .ctl = 2, .name = base ++ "control: fp4 pack the high nibble from the low value" },
        .{ .f = .pack, .ctl = 4, .name = base ++ "control: fp4 pack writes past its codes" },
        .{ .f = .rt, .ctl = 0, .name = base ++ "fp4 round trip tf_dsv41_g_fp4rt" },
        .{ .f = .rt, .ctl = 2, .name = base ++ "control: fp4 round trip the sign lost" },
        .{ .f = .rt, .ctl = 4, .name = base ++ "control: fp4 round trip overwrites its last block" },
        .{ .f = .pool, .ctl = 0, .name = base ++ "pool tf_dsv41_g_pool2" },
        .{ .f = .pool, .ctl = 2, .name = base ++ "control: pool the second slot's term lost" },
        .{ .f = .pool, .ctl = 4, .name = base ++ "control: pool writes past its carry" },
        .{ .f = .cand, .ctl = 0, .name = base ++ "cand tf_dsv41_g_cand" },
        .{ .f = .cand, .ctl = 2, .name = base ++ "control: cand ties given to the later block" },
        .{ .f = .cand, .ctl = 4, .name = base ++ "control: cand writes past its mask" },
        .{ .f = .mix, .ctl = 0, .name = base ++ "hc mix tf_dsv41_g_hcmix" },
        .{ .f = .mix, .ctl = 2, .name = base ++ "control: hc mix the Sinkhorn's passes and a square lost" },
        .{ .f = .mix, .ctl = 4, .name = base ++ "control: hc mix writes past its comb" },
        .{ .f = .pre, .ctl = 0, .name = base ++ "hc pre tf_dsv41_g_hcpre" },
        .{ .f = .pre, .ctl = 2, .name = base ++ "control: hc pre drops the fourth stream" },
        .{ .f = .pre, .ctl = 4, .name = base ++ "control: hc pre writes past its last row" },
        .{ .f = .post, .ctl = 0, .name = base ++ "hc post tf_dsv41_g_hcpost" },
        .{ .f = .post, .ctl = 2, .name = base ++ "control: hc post drops the fourth residual stream" },
        .{ .f = .post, .ctl = 4, .name = base ++ "control: hc post writes past its last stream" },
    };

    pub fn create(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, kernels: *std.ArrayList(g.Kernel), only: ?[]const u8) !*Suite {
        const s = try a.create(Suite);
        s.* = .{ .a = a };
        for (list) |entry| {
            if (only) |text| if (std.mem.indexOf(u8, entry.name, text) == null) continue;
            const p = try a.create(Proj);
            p.* = try Proj.init(a, device, queue, entry.f, entry.ctl, entry.name);
            try s.probes.append(a, p);
            try kernels.append(a, p.kernel());
        }
        return s;
    }

    pub fn destroy(s: *Suite) void {
        for (s.probes.items) |p| {
            p.deinit();
            s.a.destroy(p);
        }
        s.probes.deinit(s.a);
        s.a.destroy(s);
    }

    /// The pool carries state across rows by design, so it is left out of the split probe.
    fn invShapes(f: Fam) []const c.Shape {
        return switch (f) {
            .pack => &.{ .{ .rows = 9, .n = 512, .k = 1 }, .{ .rows = 6, .n = 256, .k = 1 } },
            .rt => &.{ .{ .rows = 9, .n = 512, .k = 1 }, .{ .rows = 6, .n = 64, .k = 1 } },
            .pool => &.{},
            .cand => &.{ .{ .rows = 9, .n = 2048, .k = 256 }, .{ .rows = 6, .n = 100, .k = 320 } },
            .mix => &.{.{ .rows = 9, .n = 1, .k = dim }},
            .pre => &.{.{ .rows = 9, .n = 1, .k = dim }},
            .post => &.{.{ .rows = 9, .n = 1, .k = dim }},
        };
    }

    /// Launches of 1, 7 and rows/3 rows equal the whole launch's bytes: no kernel carries state across rows.
    fn probe(s: *Suite, p: *Proj, shape: c.Shape, seed: u64, bad: *usize, launches: *usize) !void {
        if (!try p.prepare(shape, .random, seed)) return error.Refused;
        try p.run();
        const whole = try s.a.dupe(u8, try p.rig.outputBytes());
        defer s.a.free(whole);
        for ([_]usize{ 1, 7, @max(1, shape.rows / 3) }) |step| {
            for (p.out[0..p.no], p.out_len[0..p.no]) |o, l| @memset(p.rig.bytes(o, l), 0xff);
            const pass = p.rig.begin();
            var r0: usize = 0;
            while (r0 < shape.rows) : (r0 += step) {
                p.encode(pass.e, r0, @min(step, shape.rows - r0));
                launches.* += 1;
            }
            try p.rig.finish(pass);
            if (!std.mem.eql(u8, whole, try p.rig.outputBytes())) {
                bad.* += 1;
                std.debug.print("FAIL row invariance {s} {d}x{d}x{d}: {d} rows a launch differs from one launch\n", .{ p.name, shape.rows, shape.n, shape.k, step });
            }
        }
    }

    pub fn invariance(s: *Suite, seed: u64) !usize {
        var bad: usize = 0;
        var launches: usize = 0;
        var kernels: usize = 0;
        for (s.probes.items) |p| {
            if (p.ctl != 0) continue;
            kernels += 1;
            for (invShapes(p.fam), 0..) |shape, i| try s.probe(p, shape, seed + i, &bad, &launches);
        }
        if (kernels == 0) return 0;
        std.debug.print("{s} row invariance {s}: {d} kernels at 1, 7 and rows/3 rows a launch ({d} launches) byte-equal to one launch, {d} differ\n", .{ if (bad == 0) "PASS" else "FAIL", base, kernels, launches, bad });
        return bad;
    }
};
