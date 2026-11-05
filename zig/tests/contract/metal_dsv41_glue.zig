//! DeepSeek V4.1's glue kernels under the contract on Metal: RMS norm, embedding, joins, RoPE and act-quant.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const sources = @import("kernel_sources");
const contract = @import("contract");
const rig_ = @import("rig.zig");
const ref = @import("dsv41_glue_ref.zig");

const c = contract.spec;
const g = contract.gates;
const rows_ = contract.rows;
const gl = tf.dsv41_metal.glue;
const base = "dsv41 glue ";
const eps: f32 = 1e-20; // rms_norm_eps
const rope_positions = 66560; // past the YaRN original length, 65,536
const vocab: u32 = 300; // the embed probe's rank slice holds ids 100 to 399
const vocab_lo: i32 = 100;

pub const Fam = enum { rms_bf16, rms_f32, embed, join, mean, rope_sliding, rope_yarn, fp8, fp4_e8m0, fp4_e4m3 };

fn isQuant(f: Fam) bool {
    return f == .fp8 or f == .fp4_e8m0 or f == .fp4_e4m3;
}

fn isRope(f: Fam) bool {
    return f == .rope_sliding or f == .rope_yarn;
}

fn noSums(_: c.Shape) usize {
    return 0;
}

/// Per thread a chain over every 256th square, then the 5-step butterfly and the seven adds over simdgroups.
fn rmsDepth(s: c.Shape) usize {
    return (s.k + 255) / 256 + 12;
}

const rms_why = "a bf16 output follows the sum of squares as ss^-1/2: a bf16 sum's error is a few output ulps";
const quant_rule = "a block holding a NaN or an infinity is all NaN; every other output is the reference's round trip, exact";

pub fn spec(f: Fam, name: []const u8) c.Spec {
    const tiles16: c.Tile = .{ .size = 16 };
    return switch (f) {
        .rms_bf16, .rms_f32 => .{ .name = name, .accum = .f32, .depth = rmsDepth, .ops = 8, .out = .bf16, .tiles = .{ tiles16, .{}, .{ .size = 256, .step = 64, .min = 64 } }, .served = &.{
            .{ .rows = 16, .n = 5120, .k = 5120 }, .{ .rows = 1, .n = 5120, .k = 5120 }, .{ .rows = 4, .n = 1280, .k = 1280 },
            .{ .rows = 8, .n = 512, .k = 512 },    .{ .rows = 16, .n = 128, .k = 128 },  .{ .rows = 3, .n = 2048, .k = 2048 },
        }, .sums = .{ .bf16 = 0, .why = rms_why } },
        .embed => .{ .name = name, .accum = .none, .depth = noSums, .out = .bf16, .tiles = .{ tiles16, .{ .size = 64, .step = 2, .min = 2 }, .{} }, .served = &.{
            .{ .rows = 16, .n = 5120, .k = 4 }, .{ .rows = 1, .n = 5120, .k = 1 }, .{ .rows = 5, .n = 5120, .k = 4 }, .{ .rows = 9, .n = 128, .k = 4 },
        } },
        .join => .{ .name = name, .accum = .none, .depth = noSums, .ops = 1, .out = .bf16, .tiles = .{ tiles16, .{ .size = 64 }, .{} }, .served = &.{
            .{ .rows = 16, .n = 5120, .k = 1 }, .{ .rows = 1, .n = 5120, .k = 1 }, .{ .rows = 5, .n = 1280, .k = 1 }, .{ .rows = 8, .n = 2304, .k = 1 },
        } },
        .mean => .{ .name = name, .accum = .none, .depth = noSums, .out = .bf16, .tiles = .{ tiles16, .{ .size = 64 }, .{} }, .served = &.{
            .{ .rows = 16, .n = 5120, .k = 4 }, .{ .rows = 1, .n = 5120, .k = 4 }, .{ .rows = 6, .n = 512, .k = 4 },
        } },
        .rope_sliding, .rope_yarn => .{ .name = name, .accum = .none, .depth = noSums, .out = .bf16, .tiles = .{ tiles16, .{ .size = 8 }, .{} }, .served = &.{
            .{ .rows = 16, .n = 64, .k = 512 }, .{ .rows = 1, .n = 1, .k = 512 }, .{ .rows = 5, .n = 1, .k = 512 },
            .{ .rows = 8, .n = 32, .k = 128 },  .{ .rows = 6, .n = 1, .k = 128 },
        } },
        .fp8, .fp4_e8m0, .fp4_e4m3 => .{ .name = name, .accum = .none, .depth = noSums, .out = .bf16, .nan = .{ .stated = quant_rule }, .tiles = .{ tiles16, .{ .size = 256, .step = 32, .min = 32 }, .{} }, .served = &.{
            .{ .rows = 16, .n = 5120, .k = 5120 }, .{ .rows = 1, .n = 1280, .k = 1280 }, .{ .rows = 5, .n = 4096, .k = 4096 }, .{ .rows = 8, .n = 8192, .k = 8192 },
            .{ .rows = 16, .n = 512, .k = 512 },   .{ .rows = 6, .n = 128, .k = 128 },   .{ .rows = 2, .n = 2304, .k = 2304 },
        } },
    };
}

var table_cache: [2]?[][2]f32 = .{ null, null };

fn tableFor(f: Fam) ![][2]f32 {
    const i: usize = if (f == .rope_yarn) 1 else 0;
    if (table_cache[i]) |t| return t;
    const t = try gl.ropeTable(std.heap.page_allocator, if (i == 1) gl.RopeSpec.compressed else gl.RopeSpec.sliding, rope_positions);
    table_cache[i] = t;
    return t;
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

const vt_checked = g.hooks(Proj);
const vt_plain = blk: {
    var v = g.hooks(Proj);
    v.check = null;
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
    rows_f: ?rows_.Rows = null,
    x: []u16 = &.{},
    r32: []f32 = &.{},
    ids: []i32 = &.{},
    pos: []i32 = &.{},
    table: []u16 = &.{},
    tab: [][2]f32 = &.{},
    plants: usize = 0,
    in: [3]rig_.Ref = undefined,
    out: rig_.Ref = undefined,
    msg: [200]u8 = undefined,

    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, f: Fam, ctl: u8, name: []const u8) !Proj {
        var p: Proj = .{ .a = a, .arena = std.heap.ArenaAllocator.init(a), .rig = .{ .a = a, .device = device, .queue = queue }, .fam = f, .ctl = ctl, .name = name, .pipes = undefined };
        if (ctl == 0) {
            p.pipes = try gl.Pipelines.load(device);
        } else {
            const text = try std.fmt.allocPrint(a, "#define TF_CTL {d}\n{s}", .{ ctl, sources.dsv41.glue });
            defer a.free(text);
            p.pipes = try gl.Pipelines.loadSource(device, text);
        }
        return p;
    }

    pub fn deinit(p: *Proj) void {
        p.arena.deinit();
        p.pipes.deinit();
        p.rig.deinit();
    }

    pub fn kernel(p: *Proj) g.Kernel {
        const control: ?g.Gate = switch (p.ctl) {
            0 => null,
            1 => .nonfinite,
            2 => .shapes,
            3 => if (isQuant(p.fam)) .shapes else .sums,
            else => .repeat,
        };
        return .{ .spec = spec(p.fam, p.name), .ctx = p, .vt = if (isQuant(p.fam)) &vt_checked else &vt_plain, .control = control };
    }

    fn quantKind(p: *const Proj) ref.Kind {
        return switch (p.fam) {
            .fp8 => .fp8,
            .fp4_e8m0 => .fp4_e8m0,
            else => .fp4_e4m3,
        };
    }

    fn glQuant(p: *const Proj) gl.Quant {
        return switch (p.fam) {
            .fp8 => .fp8,
            .fp4_e8m0 => .fp4_e8m0,
            else => .fp4_e4m3,
        };
    }

    pub fn count(p: *Proj) usize {
        const s = p.shape;
        return switch (p.fam) {
            .rms_bf16, .rms_f32 => s.rows * s.k,
            .embed => s.rows * s.k * s.n,
            .join, .mean => s.rows * s.n,
            .rope_sliding, .rope_yarn => s.rows * s.n * 64,
            else => s.rows * s.n,
        };
    }

    fn bf16s(a: std.mem.Allocator, n: usize, rng: std.Random, lo: i32, hi: i32) ![]u16 {
        const v = try a.alloc(u16, n);
        for (v) |*o| o.* = c.toBf16(spread(rng, lo, hi));
        return v;
    }

    pub fn prepare(p: *Proj, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        p.rig.reset();
        _ = p.arena.reset(.retain_capacity);
        const a = p.arena.allocator();
        p.shape = shape;
        p.plants = 0;
        p.rows_f = null;
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        const nf = fill == .nonfinite;
        switch (p.fam) {
            .rms_bf16, .rms_f32 => {
                var r = try rows_.Rows.init(a, .rms, .{ .rows = shape.rows, .n = shape.k, .k = shape.k });
                r.eps = eps;
                r.fill(fill, seed);
                p.rows_f = r;
                p.in[0] = try p.rig.input(std.mem.sliceAsBytes(r.x));
                if (p.fam == .rms_f32) {
                    p.r32 = try a.alloc(f32, r.w.len);
                    for (p.r32, r.w) |*o, w| o.* = c.fromBf16(w);
                    p.in[1] = try p.rig.input(std.mem.sliceAsBytes(p.r32));
                } else p.in[1] = try p.rig.input(std.mem.sliceAsBytes(r.w));
            },
            .embed => {
                if (shape.n % 2 != 0 or shape.n == 0) return false;
                p.table = try bf16s(a, vocab * shape.n, rng, -2, 2);
                p.ids = try a.alloc(i32, shape.rows);
                for (p.ids) |*id| id.* = if (rng.uintLessThan(u32, 4) == 0) rng.intRangeLessThan(i32, -3, 520) else vocab_lo + @as(i32, @intCast(rng.uintLessThan(u32, vocab)));
                if (nf) {
                    for (0..@min(4, shape.rows)) |i| p.ids[i] = vocab_lo + @as(i32, @intCast(i));
                    p.plants = ref.plant(p.table, 4, shape.n, 0, shape.n);
                }
                p.in[0] = try p.rig.input(std.mem.sliceAsBytes(p.ids));
                p.in[1] = try p.rig.input(std.mem.sliceAsBytes(p.table));
            },
            .join => {
                const n = shape.rows * shape.n;
                p.r32 = try a.alloc(f32, n);
                for (p.r32) |*o| o.* = spread(rng, -6, 6);
                p.x = try bf16s(a, n, rng, -6, 6);
                if (nf) {
                    const d = shape.n;
                    p.r32[d / 3] = std.math.nan(f32);
                    p.r32[d + d / 2] = std.math.inf(f32);
                    p.x[d + d / 2] = c.toBf16(-std.math.inf(f32));
                    p.r32[2 * d + d - 1] = -std.math.inf(f32);
                    p.x[3 * d] = c.toBf16(std.math.inf(f32));
                    p.plants = 4;
                }
                p.in[0] = try p.rig.input(std.mem.sliceAsBytes(p.r32));
                p.in[1] = try p.rig.input(std.mem.sliceAsBytes(p.x));
            },
            .mean => {
                p.x = try bf16s(a, shape.rows * shape.k * shape.n, rng, -8, 8);
                if (nf) p.plants = ref.plant(p.x, shape.rows, shape.k * shape.n, 0, shape.k * shape.n);
                p.in[0] = try p.rig.input(std.mem.sliceAsBytes(p.x));
            },
            .rope_sliding, .rope_yarn => {
                if (shape.k < 64) return false;
                p.tab = try tableFor(p.fam);
                p.x = try bf16s(a, shape.rows * shape.n * shape.k, rng, -3, 3);
                p.pos = try a.alloc(i32, shape.rows);
                const edge = [_]i32{ 65535, 65536, 65537, rope_positions - 1 };
                for (p.pos, 0..) |*o, i| o.* = if (i % 4 == 0) edge[i / 4 % 4] else @intCast(rng.uintLessThan(u32, rope_positions));
                if (nf) p.plants = ref.plant(p.x, shape.rows, shape.n * shape.k, shape.k - 64, 64);
                p.in[0] = try p.rig.input(std.mem.sliceAsBytes(p.x));
                p.in[1] = try p.rig.input(std.mem.sliceAsBytes(p.pos));
                p.in[2] = try p.rig.input(std.mem.sliceAsBytes(p.tab));
            },
            .fp8, .fp4_e8m0, .fp4_e4m3 => {
                if (shape.n % 32 != 0 or shape.n == 0) return false;
                p.x = try a.alloc(u16, shape.rows * shape.n);
                ref.fillQuant(p.quantKind(), p.x, ref.block(p.quantKind()), seed);
                if (nf) p.plants = ref.plant(p.x, shape.rows, shape.n, 0, shape.n);
                p.in[0] = try p.rig.input(std.mem.sliceAsBytes(p.x));
            },
        }
        p.out = try p.rig.output(p.count() * 2);
        return true;
    }

    /// Rows r0 to r0 + n of the prepared call as one launch (tokens, or elements for the join); p.in[] are whole.
    pub fn encode(p: *Proj, e: anytype, r0: usize, n: usize) !void {
        const s = p.shape;
        const pl = &p.pipes;
        switch (p.fam) {
            .rms_bf16, .rms_f32 => gl.rms(pl, e, at(p.in[0], r0 * s.k * 2), at(p.in[1], 0), at(p.out, r0 * s.k * 2), @intCast(n), @intCast(s.k), eps, p.fam == .rms_f32),
            .embed => gl.embed(pl, e, at(p.in[0], r0 * 4), at(p.in[1], 0), at(p.out, r0 * s.k * s.n * 2), @intCast(n), @intCast(s.n), vocab, vocab_lo, @intCast(s.k)),
            .join => gl.moeJoin(pl, e, at(p.in[0], r0 * s.n * 4), at(p.in[1], r0 * s.n * 2), at(p.out, r0 * s.n * 2), @intCast(n * s.n)),
            .mean => gl.hcMean(pl, e, at(p.in[0], r0 * s.k * s.n * 2), at(p.out, r0 * s.n * 2), @intCast(n), @intCast(s.n), @intCast(s.k)),
            .rope_sliding, .rope_yarn => {
                const hd: u32 = @intCast(s.k);
                const h: u32 = @intCast(s.n);
                const call: gl.RopeCall = .{ .rows = @intCast(n), .heads = h, .ldx = h * hd, .hx = hd, .ox = hd - 64, .ldy = h * 64, .hy = 64, .oy = 0 };
                try gl.rope(pl, e, at(p.in[0], r0 * h * hd * 2), at(p.out, r0 * h * 64 * 2), at(p.in[1], r0 * 4), at(p.in[2], 0), call);
            },
            else => try gl.quant(pl, e, p.glQuant(), at(p.in[0], r0 * s.n * 2), at(p.out, r0 * s.n * 2), @intCast(n), @intCast(s.n)),
        }
    }

    pub fn run(p: *Proj) !void {
        const pass = p.rig.begin();
        try p.encode(pass.e, 0, p.shape.rows);
        try p.rig.finish(pass);
    }

    pub fn reference(p: *Proj, want: []f64, mag: []f64) !void {
        const s = p.shape;
        @memset(mag, 0);
        switch (p.fam) {
            .rms_bf16, .rms_f32 => try p.rows_f.?.reference(want, mag),
            .embed => for (0..s.rows) |t| for (0..s.k) |j| for (0..s.n) |d| {
                const id = p.ids[t] - vocab_lo;
                want[(t * s.k + j) * s.n + d] = if (id >= 0 and id < vocab) c.fromBf16(p.table[@as(usize, @intCast(id)) * s.n + d]) else 0;
            },
            .join => for (want, p.r32, p.x) |*w, r, sh| {
                w.* = @as(f64, r) + c.fromBf16(sh);
            },
            .mean => for (0..s.rows) |t| for (0..s.n) |d| {
                var sum: f64 = 0;
                var abs: f64 = 0;
                for (0..s.k) |j| {
                    const v: f64 = c.fromBf16(p.x[(t * s.k + j) * s.n + d]);
                    sum += v;
                    abs += @abs(v);
                }
                want[t * s.n + d] = sum / @as(f64, @floatFromInt(s.k));
                mag[t * s.n + d] = abs / @as(f64, @floatFromInt(s.k));
            },
            .rope_sliding, .rope_yarn => for (0..s.rows) |t| for (0..s.n) |h| for (0..32) |i| {
                const from = (t * s.n + h) * s.k + s.k - 64 + 2 * i;
                const xr: f64 = c.fromBf16(p.x[from]);
                const xi: f64 = c.fromBf16(p.x[from + 1]);
                const cs = p.tab[@as(usize, @intCast(p.pos[t])) * 32 + i];
                const co: f64 = cs[0];
                const si: f64 = cs[1];
                const o = (t * s.n + h) * 64 + 2 * i;
                want[o] = xr * co - xi * si;
                want[o + 1] = xr * si + xi * co;
                mag[o] = @abs(xr * co) + @abs(xi * si);
                mag[o + 1] = @abs(xr * si) + @abs(xi * co);
            },
            else => @memset(want, 0),
        }
    }

    /// Embedding gathers are exact; the fp32 steps of the mean and RoPE add 2^-24 of their term magnitudes.
    pub fn bounds(p: *Proj, want: []const f64, mag: []const f64, allow: []f64) !void {
        switch (p.fam) {
            .embed => @memset(allow, 0),
            .mean, .rope_sliding, .rope_yarn => {
                const steps: f64 = if (p.fam == .mean) @floatFromInt(p.shape.k + 1) else 4;
                for (allow, want, mag) |*l, w, m| {
                    const e32 = steps * 0x1p-24 * m;
                    l.* = e32 + 0x1p-8 * (@abs(w) + e32) + 0x1p-133;
                }
            },
            else => {},
        }
    }

    pub fn simulate(p: *Proj, acc: c.Accum, order: g.Order, out: []f64) !void {
        return (p.rows_f orelse return error.NoSums).simulate(acc, order, out);
    }

    fn nanBits(v: u16) bool {
        return (v & 0x7fff) > 0x7f80;
    }

    /// Every block against the oracle's bits (any NaN equals any NaN); gate 1 also needs the plants to have landed.
    pub fn check(p: *Proj, gate: g.Gate) !?[]const u8 {
        const s = p.shape;
        const kind = p.quantKind();
        const blk = ref.block(kind);
        const got = std.mem.bytesAsSlice(u16, p.rig.bytes(p.out, p.count() * 2));
        var xs: [32]f32 = undefined;
        var want: [32]u16 = undefined;
        var bad_blocks: usize = 0;
        for (0..s.rows * s.n / blk) |b| {
            const x = p.x[b * blk ..][0..blk];
            for (xs[0..blk], x) |*o, v| o.* = c.fromBf16(v);
            ref.roundTrip(kind, xs[0..blk], want[0..blk]);
            for (want[0..blk], got[b * blk ..][0..blk], 0..) |w, h, i| {
                if (w != h and !(nanBits(w) and nanBits(h))) return try std.fmt.bufPrint(&p.msg, "block {d} element {d} of {d}x{d}: got 0x{x:0>4}, the round trip gives 0x{x:0>4}", .{ b, i, s.rows, s.n, h, w });
            }
            if (nanBits(want[0])) bad_blocks += 1;
        }
        if (gate == .nonfinite and (p.plants < 5 or bad_blocks < 4)) return try std.fmt.bufPrint(&p.msg, "{d} plants, {d} NaN blocks: no proof", .{ p.plants, bad_blocks });
        return null;
    }
};

/// Every registered variant and the launch-split and in-place probes after the gates.
pub const Suite = struct {
    a: std.mem.Allocator,
    probes: std.ArrayList(*Proj) = .empty,

    const E = struct { f: Fam, ctl: u8, name: []const u8 };
    const list = [_]E{
        .{ .f = .rms_bf16, .ctl = 0, .name = base ++ "rms bf16 weight tf_dsv41_rms_bf16w" },
        .{ .f = .rms_f32, .ctl = 0, .name = base ++ "rms f32 weight tf_dsv41_rms_f32w" },
        .{ .f = .rms_bf16, .ctl = 1, .name = base ++ "control: rms non-finite inputs flushed" },
        .{ .f = .rms_bf16, .ctl = 2, .name = base ++ "control: rms last input dropped from the sum" },
        .{ .f = .rms_bf16, .ctl = 3, .name = base ++ "control: rms partial sums rounded to fp16" },
        .{ .f = .rms_bf16, .ctl = 4, .name = base ++ "control: rms writes past its output" },
        .{ .f = .embed, .ctl = 0, .name = base ++ "embed tf_dsv41_embed" },
        .{ .f = .embed, .ctl = 1, .name = base ++ "control: embed NaN and infinity rows flushed" },
        .{ .f = .embed, .ctl = 2, .name = base ++ "control: embed last pair of a row skipped" },
        .{ .f = .embed, .ctl = 4, .name = base ++ "control: embed writes past its output" },
        .{ .f = .join, .ctl = 0, .name = base ++ "moe join tf_dsv41_moe_join" },
        .{ .f = .join, .ctl = 1, .name = base ++ "control: join non-finite inputs flushed" },
        .{ .f = .join, .ctl = 2, .name = base ++ "control: join rounds the routed sum to bf16 first" },
        .{ .f = .join, .ctl = 4, .name = base ++ "control: join writes past its output" },
        .{ .f = .mean, .ctl = 0, .name = base ++ "hc mean tf_dsv41_hc_mean" },
        .{ .f = .mean, .ctl = 1, .name = base ++ "control: hc mean non-finite inputs flushed" },
        .{ .f = .mean, .ctl = 2, .name = base ++ "control: hc mean drops the last copy" },
        .{ .f = .mean, .ctl = 4, .name = base ++ "control: hc mean writes past its output" },
        .{ .f = .rope_sliding, .ctl = 0, .name = base ++ "rope sliding spectrum tf_dsv41_rope" },
        .{ .f = .rope_yarn, .ctl = 0, .name = base ++ "rope compressed YaRN spectrum tf_dsv41_rope" },
        .{ .f = .rope_yarn, .ctl = 1, .name = base ++ "control: rope non-finite inputs flushed" },
        .{ .f = .rope_yarn, .ctl = 2, .name = base ++ "control: rope reads the wrong pairs" },
        .{ .f = .rope_yarn, .ctl = 4, .name = base ++ "control: rope writes past its output" },
        .{ .f = .fp8, .ctl = 0, .name = base ++ "fp8 round trip tf_dsv41_fp8_rt" },
        .{ .f = .fp8, .ctl = 1, .name = base ++ "control: fp8 NaN not flagged" },
        .{ .f = .fp8, .ctl = 2, .name = base ++ "control: fp8 scale exponent floored" },
        .{ .f = .fp8, .ctl = 3, .name = base ++ "control: fp8 ties rounded away from zero" },
        .{ .f = .fp8, .ctl = 4, .name = base ++ "control: fp8 writes past its output" },
        .{ .f = .fp4_e8m0, .ctl = 0, .name = base ++ "fp4 e8m0 block 32 round trip tf_dsv41_fp4_rt32" },
        .{ .f = .fp4_e8m0, .ctl = 1, .name = base ++ "control: fp4 e8m0 NaN not flagged" },
        .{ .f = .fp4_e8m0, .ctl = 2, .name = base ++ "control: fp4 e8m0 scale exponent floored" },
        .{ .f = .fp4_e8m0, .ctl = 3, .name = base ++ "control: fp4 e8m0 ties rounded away from zero" },
        .{ .f = .fp4_e8m0, .ctl = 4, .name = base ++ "control: fp4 e8m0 writes past its output" },
        .{ .f = .fp4_e4m3, .ctl = 0, .name = base ++ "fp4 e4m3 block 16 round trip tf_dsv41_fp4_rt16" },
        .{ .f = .fp4_e4m3, .ctl = 1, .name = base ++ "control: fp4 e4m3 NaN not flagged" },
        .{ .f = .fp4_e4m3, .ctl = 2, .name = base ++ "control: fp4 e4m3 scale left unrounded to E4M3" },
        .{ .f = .fp4_e4m3, .ctl = 4, .name = base ++ "control: fp4 e4m3 writes past its output" },
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

    fn invShapes(f: Fam) []const c.Shape {
        return switch (f) {
            .rms_bf16, .rms_f32 => &.{ .{ .rows = 33, .n = 5120, .k = 5120 }, .{ .rows = 16, .n = 1280, .k = 1280 } },
            .embed => &.{.{ .rows = 20, .n = 5120, .k = 4 }},
            .join => &.{.{ .rows = 20, .n = 2304, .k = 1 }},
            .mean => &.{.{ .rows = 20, .n = 5120, .k = 4 }},
            .rope_sliding, .rope_yarn => &.{ .{ .rows = 17, .n = 64, .k = 512 }, .{ .rows = 9, .n = 32, .k = 128 } },
            else => &.{ .{ .rows = 20, .n = 5120, .k = 5120 }, .{ .rows = 9, .n = 512, .k = 512 } },
        };
    }

    /// Launches of 1, 7 and rows/3 rows, and one in place for RoPE and the round trips, equal the whole launch's bytes.
    fn probe(s: *Suite, p: *Proj, shape: c.Shape, seed: u64, bad: *usize, launches: *usize) !void {
        if (!try p.prepare(shape, .random, seed)) return error.Refused;
        try p.run();
        const len = p.count() * 2;
        const whole = try s.a.dupe(u8, p.rig.bytes(p.out, len));
        defer s.a.free(whole);
        for ([_]usize{ 1, 7, @max(1, shape.rows / 3) }) |step| {
            @memset(p.rig.bytes(p.out, len), 0x55);
            const pass = p.rig.begin();
            var r0: usize = 0;
            while (r0 < shape.rows) : (r0 += step) {
                try p.encode(pass.e, r0, @min(step, shape.rows - r0));
                launches.* += 1;
            }
            try p.rig.finish(pass);
            if (!std.mem.eql(u8, whole, p.rig.bytes(p.out, len))) {
                bad.* += 1;
                std.debug.print("FAIL row invariance {s} {d}x{d}x{d}: {d} rows a launch differs from one launch\n", .{ p.name, shape.rows, shape.n, shape.k, step });
            }
        }
        if (!isQuant(p.fam) and !isRope(p.fam)) return;
        const xlen = p.x.len * 2;
        const scr = try p.rig.scratch(xlen);
        @memcpy(p.rig.bytes(scr, xlen), std.mem.sliceAsBytes(p.x));
        const pass = p.rig.begin();
        const rows: u32 = @intCast(shape.rows);
        if (isQuant(p.fam)) {
            try gl.quant(&p.pipes, pass.e, p.glQuant(), at(scr, 0), at(scr, 0), rows, @intCast(shape.n));
        } else {
            const call = gl.ropeInPlace(rows, @intCast(shape.n), @intCast(shape.k), 32);
            try gl.rope(&p.pipes, pass.e, at(scr, 0), at(scr, 0), at(p.in[1], 0), at(p.in[2], 0), call);
        }
        try p.rig.finish(pass);
        launches.* += 1;
        const done = p.rig.bytes(scr, xlen);
        var same = true;
        if (isQuant(p.fam)) {
            same = std.mem.eql(u8, whole, done);
        } else for (0..shape.rows * shape.n) |rh| {
            const from = rh * shape.k * 2 + (shape.k - 64) * 2;
            if (!std.mem.eql(u8, whole[rh * 128 ..][0..128], done[from..][0..128])) same = false;
        }
        if (!same) {
            bad.* += 1;
            std.debug.print("FAIL in place {s} {d}x{d}x{d}: differs from the out-of-place bytes\n", .{ p.name, shape.rows, shape.n, shape.k });
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
        std.debug.print("{s} row invariance {s}: {d} kernels at 1, 7 and rows/3 rows a launch and in place ({d} launches) byte-equal to one launch, {d} differ\n", .{ if (bad == 0) "PASS" else "FAIL", base, kernels, launches, bad });
        return bad;
    }

    fn freq64(sp: gl.RopeSpec, i: usize) f64 {
        const dim: f64 = @floatFromInt(sp.dim);
        const f = 1.0 / std.math.pow(f64, sp.theta, @as(f64, @floatFromInt(2 * i)) / dim);
        if (sp.original == 0) return f;
        const orig: f64 = @floatFromInt(sp.original);
        const low = @max(@floor(dim * @log(orig / (sp.beta_fast * 2 * std.math.pi)) / (2 * @log(sp.theta))), 0);
        const high = @min(@ceil(dim * @log(orig / (sp.beta_slow * 2 * std.math.pi)) / (2 * @log(sp.theta))), dim - 1);
        const ramp = std.math.clamp((@as(f64, @floatFromInt(i)) - low) / @max(high - low, 1e-3), 0, 1);
        return f / sp.factor * ramp + f * (1 - ramp);
    }

    /// The RoPE tables against float64: frequencies within 4 fp32 ulps, cos and sin correctly rounded.
    pub fn tables(s: *Suite) !usize {
        if (s.probes.items.len == 0) return 0;
        var bad: usize = 0;
        for ([_]gl.RopeSpec{ gl.RopeSpec.sliding, gl.RopeSpec.compressed }) |sp| {
            var freqs: [32]f32 = undefined;
            gl.ropeFreqs(sp, &freqs);
            for (freqs, 0..) |f, i| {
                const want = freq64(sp, i);
                if (@abs(f - want) > 4 * std.math.floatEps(f32) * want) bad += 1;
            }
            var row: [32][2]f32 = undefined;
            for ([_]u32{ 0, 1, 777, 65535, 65536, 1048575 }) |pos| {
                gl.ropeRow(&freqs, pos, &row);
                for (freqs, row) |f, cs| {
                    const angle: f64 = @as(f32, @floatFromInt(pos)) * f;
                    const want: [2]f32 = .{ @floatCast(@cos(angle)), @floatCast(@sin(angle)) };
                    if (cs[0] != want[0] or cs[1] != want[1] or @abs(cs[0]) > 1 or @abs(cs[1]) > 1) bad += 1;
                }
            }
        }
        std.debug.print("{s} rope tables {s}: both spectra within 4 ulps of float64 frequencies, cos and sin correctly rounded, {d} misses\n", .{ if (bad == 0) "PASS" else "FAIL", base, bad });
        return bad;
    }
};
