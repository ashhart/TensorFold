//! DeepSeek V4.1's compressor kernels under the contract on Metal: the ratio-2 pool, FP4 round trip and key packing.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const sources = @import("kernel_sources");
const contract = @import("contract");
const rig_ = @import("rig.zig");
const ref = @import("dsv41_glue_ref.zig");

const c = contract.spec;
const g = contract.gates;
const cp = tf.dsv41_metal.compress;
const base = "dsv41 compress ";
const max_dim = 8192;

pub const Fam = enum { pool, rt, pack_keys, pack_q };

fn noSums(_: c.Shape) usize {
    return 0;
}

const pool_rule = "an odd position pools its two slots as the reference's softmax does, a NaN or infinity as it does (same class)," ++
    " an even position's row is not written, and the carry holds every slot's last value";
const rt_rule = "a block holding a NaN or an infinity is all NaN; every other output is the reference's round trip, exact";
const pack_rule = "a block holding a NaN or an infinity packs zero codes under the E8M0 NaN byte 255; every other byte is exact";

pub fn spec(f: Fam, name: []const u8) c.Spec {
    const blocks: c.Tile = .{ .size = 256, .step = 32, .min = 32 };
    return switch (f) {
        .pool => .{ .name = name, .accum = .none, .depth = noSums, .out = .bf16, .nan = .{ .stated = pool_rule }, .served = &.{
            .{ .rows = 16, .n = 512, .k = 1 }, .{ .rows = 1, .n = 512, .k = 1 }, .{ .rows = 4, .n = 512, .k = 1 }, .{ .rows = 700, .n = 512, .k = 1 },
            .{ .rows = 5, .n = 64, .k = 1 },   .{ .rows = 3, .n = 33, .k = 1 },  .{ .rows = 9, .n = 1, .k = 1 },   .{ .rows = 8, .n = 100, .k = 1 },
        } },
        .rt => .{ .name = name, .accum = .none, .depth = noSums, .out = .bf16, .nan = .{ .stated = rt_rule }, .tiles = .{ .{}, .{ .size = 256, .step = 16, .min = 16 }, .{} }, .served = &.{
            .{ .rows = 16, .n = 512, .k = 1 }, .{ .rows = 1, .n = 512, .k = 1 }, .{ .rows = 5, .n = 512, .k = 1 },
            .{ .rows = 64, .n = 512, .k = 1 }, .{ .rows = 8, .n = 64, .k = 1 },  .{ .rows = 3, .n = 16, .k = 1 },
        } },
        .pack_keys => .{ .name = name, .accum = .none, .depth = noSums, .out = .raw, .nan = .{ .stated = pack_rule }, .tiles = .{ .{}, blocks, .{} }, .served = &.{
            .{ .rows = 16, .n = 128, .k = 1 }, .{ .rows = 1, .n = 128, .k = 1 }, .{ .rows = 5, .n = 128, .k = 1 },
            .{ .rows = 64, .n = 128, .k = 1 }, .{ .rows = 4, .n = 256, .k = 1 }, .{ .rows = 3, .n = 32, .k = 1 },
        } },
        .pack_q => .{ .name = name, .accum = .none, .depth = noSums, .out = .raw, .nan = .{ .stated = pack_rule }, .tiles = .{ .{}, blocks, .{} }, .served = &.{
            .{ .rows = 16, .n = 8192, .k = 1 }, .{ .rows = 1, .n = 8192, .k = 1 }, .{ .rows = 5, .n = 4096, .k = 1 },
            .{ .rows = 8, .n = 128, .k = 1 },   .{ .rows = 3, .n = 32, .k = 1 },
        } },
    };
}

fn pow2(e: i32) f32 {
    return std.math.pow(f32, 2, @floatFromInt(e));
}

fn spread(rng: std.Random, lo: i32, hi: i32) f32 {
    return (rng.float(f32) * 2 - 1) * pow2(rng.intRangeAtMost(i32, lo, hi));
}

fn halfUlp(v: f64) f64 {
    const a = @abs(v);
    if (a < 0x1p-126) return 0x1p-134;
    return std.math.ldexp(@as(f64, 1), std.math.frexp(a).exponent - 1 - 8);
}

fn isNanBits(v: u16) bool {
    return (v & 0x7fff) > 0x7f80;
}

fn at(r: rig_.Ref, extra: usize) cp.Ref {
    return .{ .buf = r.buf, .off = r.off + extra };
}

/// The E2M1 code nearest to `a` (at least 0), a tie to the even code.
fn nearestCode(a: f64) u8 {
    var best: usize = 0;
    for (ref.e2m1, 0..) |t, i| {
        const d = @abs(a - t);
        const bd = @abs(a - ref.e2m1[best]);
        if (d < bd or (d == bd and i % 2 == 0 and best % 2 == 1)) best = i;
    }
    return @intCast(best);
}

/// One 32-value block's packed bytes from the math: sixteen code bytes and the E8M0 scale byte.
fn packBlock(x: []const u16, codes: *[16]u8) u8 {
    var amax: f32 = 0;
    for (x) |h| {
        const v = c.fromBf16(h);
        if (!std.math.isFinite(v)) {
            codes.* = @splat(0);
            return 255;
        }
        amax = @max(amax, @abs(v));
    }
    const inv6: f32 = 1.0 / 6.0;
    const e = ref.ceilLog2(@max(amax, 6 * 0x1p-126) * inv6);
    const s = std.math.pow(f64, 2, @floatFromInt(e));
    for (0..16) |j| {
        var byte: u8 = 0;
        for (0..2) |k| {
            const v = c.fromBf16(x[2 * j + k]);
            const code = nearestCode(@min(@as(f64, @abs(v)) / s, 6)) | @as(u8, if (std.math.signbit(v)) 8 else 0);
            byte |= code << @intCast(4 * k);
        }
        codes[j] = byte;
    }
    return @intCast(e + 127);
}

const vt = g.hooks(Proj);

pub const Proj = struct {
    a: std.mem.Allocator,
    arena: std.heap.ArenaAllocator,
    rig: rig_.Rig,
    fam: Fam,
    ctl: u8,
    name: []const u8,
    pipes: cp.Pipelines,
    /// the pipelines of glue_tmp, when the equality probe launches them in place of ours
    tmp: ?cp.Pipelines = null,
    shape: c.Shape = undefined,
    x: []u16 = &.{},
    kv: []f32 = &.{},
    sc: []f32 = &.{},
    pos: []i32 = &.{},
    dest: []i32 = &.{},
    state0: []f32 = &.{},
    plants: usize = 0,
    in: [4]rig_.Ref = undefined,
    o: [2]rig_.Ref = undefined,
    ol: [2]usize = .{ 0, 0 },
    no: usize = 0,
    msg: [240]u8 = undefined,

    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, f: Fam, ctl: u8, name: []const u8) !Proj {
        var p: Proj = .{ .a = a, .arena = std.heap.ArenaAllocator.init(a), .rig = .{ .a = a, .device = device, .queue = queue }, .fam = f, .ctl = ctl, .name = name, .pipes = undefined };
        if (ctl == 0) {
            p.pipes = try cp.Pipelines.load(device);
        } else {
            const text = try std.fmt.allocPrint(a, "#define TF_CTL {d}\n{s}", .{ ctl, sources.dsv41.compress });
            defer a.free(text);
            p.pipes = try cp.Pipelines.loadSource(device, text);
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
            2, 3 => .shapes,
            else => .repeat,
        };
        return .{ .spec = spec(p.fam, p.name), .ctx = p, .vt = &vt, .control = control };
    }

    pub fn count(p: *Proj) usize {
        return p.shape.rows * p.shape.n;
    }

    fn dim(p: *const Proj) usize {
        return p.shape.n;
    }

    fn output(p: *Proj, len: usize) !rig_.Ref {
        p.o[p.no] = try p.rig.output(len);
        p.ol[p.no] = len;
        p.no += 1;
        return p.o[p.no - 1];
    }

    /// Rows of destination: a permutation of the rows with every fifth one skipped (-1).
    fn destinations(p: *Proj, a: std.mem.Allocator) ![]i32 {
        const rows = p.shape.rows;
        const d = try a.alloc(i32, rows);
        for (d, 0..) |*o, r| o.* = if (r % 5 == 4) -1 else @intCast(rows - 1 - r);
        return d;
    }

    fn preparePool(p: *Proj, a: std.mem.Allocator, rng: std.Random, nf: bool) !void {
        const n = p.dim();
        const rows = p.shape.rows;
        p.kv = try a.alloc(f32, rows * n);
        p.sc = try a.alloc(f32, rows * n);
        p.pos = try a.alloc(i32, rows);
        p.state0 = try a.alloc(f32, 4 * n);
        var p0: i32 = @intCast(rng.uintLessThan(u32, 1 << 20));
        if (nf) p0 &= ~@as(i32, 1);
        for (p.pos, 0..) |*o, r| o.* = p0 + @as(i32, @intCast(r));
        const fresh = !nf and rng.uintLessThan(u32, 3) == 0;
        for (p.state0[0 .. 2 * n]) |*o| o.* = if (fresh) 0 else spread(rng, -4, 4);
        for (p.state0[2 * n ..]) |*o| o.* = if (fresh) -std.math.inf(f32) else spread(rng, -3, 3);
        for (0..rows) |r| {
            const mode = rng.uintLessThan(u32, 5);
            for (0..n) |d| {
                p.kv[r * n + d] = spread(rng, -4, 4);
                p.sc[r * n + d] = switch (mode) {
                    0, 1 => spread(rng, -3, 3),
                    2 => (rng.float(f32) * 2 - 1) * 80,
                    3 => if (r > 0) p.sc[(r - 1) * n + d] else 0.5,
                    else => @as(f32, @floatFromInt(rng.intRangeAtMost(i32, -2, 2))),
                };
            }
        }
        if (nf and rows > 3) {
            p.kv[1 * n + n / 3] = std.math.nan(f32);
            p.sc[3 * n + n / 2] = std.math.inf(f32);
            p.kv[0 * n + 0] = std.math.inf(f32);
            p.sc[0 * n + 0] = 0.5;
            p.sc[1 * n + 0] = 0.5;
            p.kv[1 * n + 0] = 1;
            p.kv[2 * n + n - 1] = std.math.inf(f32);
            p.kv[3 * n + n - 1] = -std.math.inf(f32);
            p.sc[2 * n + n - 1] = 0.25;
            p.sc[3 * n + n - 1] = 0.25;
            p.plants = 4;
        }
        p.in[0] = try p.rig.input(std.mem.sliceAsBytes(p.kv));
        p.in[1] = try p.rig.input(std.mem.sliceAsBytes(p.sc));
        p.in[2] = try p.rig.input(std.mem.sliceAsBytes(p.pos));
        _ = try p.output(rows * n * 2);
        _ = try p.output(4 * n * 4);
    }

    pub fn prepare(p: *Proj, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.n == 0 or shape.n > max_dim) return false;
        const blk: usize = switch (p.fam) {
            .pool => 1,
            .rt => 16,
            else => 32,
        };
        if (shape.n % blk != 0) return false;
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
        const rows = shape.rows;
        if (p.fam == .pool) {
            try p.preparePool(a, rng, nf);
            return true;
        }
        p.x = try a.alloc(u16, rows * n);
        ref.fillQuant(if (p.fam == .rt) .fp4_e4m3 else .fp4_e8m0, p.x, blk, seed);
        if (nf) p.plants = ref.plant(p.x, rows, n, 0, n);
        p.in[0] = try p.rig.input(std.mem.sliceAsBytes(p.x));
        switch (p.fam) {
            .rt => {
                p.dest = try p.destinations(a);
                p.in[1] = try p.rig.input(std.mem.sliceAsBytes(p.dest));
                _ = try p.output(rows * n * 2);
            },
            .pack_keys => {
                p.dest = try p.destinations(a);
                p.in[1] = try p.rig.input(std.mem.sliceAsBytes(p.dest));
                _ = try p.output(rows * (n / 2 + n / 32));
            },
            else => {
                _ = try p.output(rows * n / 2);
                _ = try p.output(rows * n / 32);
            },
        }
        return true;
    }

    /// Outputs back to the poison and the carry to its start: where a launch sequence begins.
    pub fn reset(p: *Proj) void {
        for (0..p.no) |i| @memset(p.rig.bytes(p.o[i], p.ol[i]), 0xff);
        if (p.fam == .pool) @memcpy(p.rig.bytes(p.o[1], p.ol[1]), std.mem.sliceAsBytes(p.state0));
    }

    /// Rows r0 to r0 + m of the prepared call as one launch, our kernels or (tmp) glue_tmp's.
    pub fn encode(p: *Proj, e: anytype, r0: usize, m: usize, tmp: bool) void {
        const pl = if (tmp) &p.tmp.? else &p.pipes;
        const n = p.shape.n;
        const rows: u32 = @intCast(m);
        const nn: u32 = @intCast(n);
        switch (p.fam) {
            .pool => cp.pool2(pl, e, at(p.in[0], r0 * n * 4), at(p.in[1], r0 * n * 4), at(p.o[1], 0), at(p.in[2], r0 * 4), at(p.o[0], r0 * n * 2), rows, nn),
            .rt => cp.fp4rt(pl, e, at(p.in[0], r0 * n * 2), at(p.o[0], 0), at(p.in[1], r0 * 4), rows, nn),
            .pack_keys => {
                const row: u32 = nn / 2 + nn / 32;
                cp.fp4pack(pl, e, at(p.in[0], r0 * n * 2), at(p.o[0], 0), at(p.o[0], 0), rows, .{ .blocks = nn / 32, .in_stride = nn, .code_row = row, .scale_row = row, .scale_off = nn / 2, .dest = at(p.in[1], r0 * 4) });
            },
            .pack_q => cp.fp4pack(pl, e, at(p.in[0], r0 * n * 2), at(p.o[0], r0 * n / 2), at(p.o[1], r0 * n / 32), rows, .{ .blocks = nn / 32, .in_stride = nn, .code_row = nn / 2, .scale_row = nn / 32 }),
        }
    }

    pub fn run(p: *Proj) !void {
        p.reset();
        const pass = p.rig.begin();
        p.encode(pass.e, 0, p.shape.rows, false);
        try p.rig.finish(pass);
    }

    pub fn reference(_: *Proj, want: []f64, mag: []f64) !void {
        @memset(want, 0);
        @memset(mag, 0);
    }

    fn fail(p: *Proj, comptime fmt: []const u8, args: anytype) ![]const u8 {
        return try std.fmt.bufPrint(&p.msg, fmt, args);
    }

    /// The pool against float64 by column: bounds on pooled rows, even rows untouched, the carry.
    fn checkPool(p: *Proj, gate: g.Gate) !?[]const u8 {
        const n = p.dim();
        const rows = p.shape.rows;
        const y = std.mem.bytesAsSlice(u16, p.rig.bytes(p.o[0], rows * n * 2));
        const st = std.mem.bytesAsSlice(u32, p.rig.bytes(p.o[1], 4 * n * 4));
        var nonfinite: usize = 0;
        var pooled: usize = 0;
        for (0..n) |d| {
            var k2 = [2]f32{ p.state0[d], p.state0[n + d] };
            var s2 = [2]f32{ p.state0[2 * n + d], p.state0[3 * n + d] };
            for (0..rows) |r| {
                const slot: usize = @intCast(p.pos[r] & 1);
                k2[slot] = p.kv[r * n + d];
                s2[slot] = p.sc[r * n + d];
                const got = y[r * n + d];
                if (slot == 0) {
                    if (got != 0xffff) return try p.fail("row {d} column {d}: an even position's row was written (0x{x:0>4})", .{ r, d, got });
                    continue;
                }
                const m = @max(@as(f64, s2[0]), @as(f64, s2[1]));
                const e0 = @exp(@as(f64, s2[0]) - m);
                const e1 = @exp(@as(f64, s2[1]) - m);
                const k0: f64 = k2[0];
                const k1: f64 = k2[1];
                const want = (k0 * e0 + k1 * e1) / (e0 + e1);
                const mag = (@abs(k0 * e0) + @abs(k1 * e1)) / (e0 + e1);
                const have: f64 = c.fromBf16(got);
                pooled += 1;
                if (!std.math.isFinite(want)) {
                    nonfinite += 1;
                    if (c.classOf(have) != c.classOf(want)) return try p.fail("row {d} column {d} of {d}x{d}: {t} where float64 gives {t}", .{ r, d, rows, n, c.classOf(have), c.classOf(want) });
                    continue;
                }
                const work = 10 * 0x1p-24 * mag;
                if (!(@abs(have - want) <= halfUlp(@abs(want) + work) + work)) return try p.fail("row {d} column {d} of {d}x{d}: {e:.6} where float64 gives {e:.6}", .{ r, d, rows, n, have, want });
            }
            for ([_]f32{ k2[0], k2[1], s2[0], s2[1] }, 0..) |v, i| {
                const h: f32 = @bitCast(st[i * n + d]);
                if (!(h == v or (std.math.isNan(h) and std.math.isNan(v)))) return try p.fail("carry word {d} of column {d}: {e:.6} where the last slot value is {e:.6}", .{ i, d, h, v });
            }
        }
        if (gate == .nonfinite and (p.plants < 4 or nonfinite < 3 or nonfinite == pooled)) return try p.fail("{d} plants, {d} of {d} pooled outputs non-finite: no proof", .{ p.plants, nonfinite, pooled });
        return null;
    }

    fn checkRt(p: *Proj, gate: g.Gate) !?[]const u8 {
        const n = p.dim();
        const rows = p.shape.rows;
        const got = std.mem.bytesAsSlice(u16, p.rig.bytes(p.o[0], rows * n * 2));
        const want = try p.a.alloc(u16, rows * n);
        defer p.a.free(want);
        const written = try p.a.alloc(bool, rows * n);
        defer p.a.free(written);
        @memset(want, 0xffff);
        @memset(written, false);
        var xs: [16]f32 = undefined;
        var bad_blocks: usize = 0;
        for (0..rows) |r| {
            if (p.dest[r] < 0) continue;
            const to: usize = @intCast(p.dest[r]);
            for (0..n / 16) |b| {
                for (&xs, p.x[r * n + 16 * b ..][0..16]) |*o, v| o.* = c.fromBf16(v);
                ref.roundTrip(.fp4_e4m3, &xs, want[to * n + 16 * b ..][0..16]);
                @memset(written[to * n + 16 * b ..][0..16], true);
                bad_blocks += @intFromBool(isNanBits(want[to * n + 16 * b]));
            }
        }
        for (want, got, written, 0..) |w, h, wr, i| {
            const ok = if (wr) (w == h or (isNanBits(w) and isNanBits(h))) else h == 0xffff;
            if (!ok) return try p.fail("row {d} element {d} of {d}x{d}: got 0x{x:0>4}, the round trip gives 0x{x:0>4}", .{ i / n, i % n, rows, n, h, w });
        }
        if (gate == .nonfinite and (p.plants < 5 or bad_blocks < 4)) return try p.fail("{d} plants, {d} NaN blocks: no proof", .{ p.plants, bad_blocks });
        return null;
    }

    fn checkPack(p: *Proj, gate: g.Gate) !?[]const u8 {
        const n = p.dim();
        const rows = p.shape.rows;
        const keys = p.fam == .pack_keys;
        const code_row = if (keys) n / 2 + n / 32 else n / 2;
        const scale_row = if (keys) code_row else n / 32;
        const scale_off = if (keys) n / 2 else 0;
        const cw = try p.a.alloc(u8, rows * code_row);
        defer p.a.free(cw);
        const sw = if (keys) cw else try p.a.alloc(u8, rows * scale_row);
        defer if (!keys) p.a.free(sw);
        @memset(cw, 0xff);
        @memset(sw, 0xff);
        var bad_blocks: usize = 0;
        for (0..rows) |r| {
            const to: usize = if (keys) (if (p.dest[r] < 0) continue else @intCast(p.dest[r])) else r;
            for (0..n / 32) |b| {
                const sb = packBlock(p.x[r * n + 32 * b ..][0..32], cw[to * code_row + 16 * b ..][0..16]);
                sw[to * scale_row + scale_off + b] = sb;
                bad_blocks += @intFromBool(sb == 255);
            }
        }
        const got_c = p.rig.bytes(p.o[0], rows * code_row);
        if (std.mem.indexOfDiff(u8, cw, got_c)) |i| return try p.fail("{s} byte {d} of {d}x{d}: got 0x{x:0>2}, the packing gives 0x{x:0>2}", .{ if (keys) "key" else "code", i, rows, n, got_c[i], cw[i] });
        if (!keys) {
            const got_s = p.rig.bytes(p.o[1], rows * scale_row);
            if (std.mem.indexOfDiff(u8, sw, got_s)) |i| return try p.fail("scale byte {d} of {d}x{d}: got 0x{x:0>2}, the packing gives 0x{x:0>2}", .{ i, rows, n, got_s[i], sw[i] });
        }
        if (gate == .nonfinite and (p.plants < 5 or bad_blocks < 4)) return try p.fail("{d} plants, {d} NaN blocks: no proof", .{ p.plants, bad_blocks });
        return null;
    }

    pub fn check(p: *Proj, gate: g.Gate) !?[]const u8 {
        return switch (p.fam) {
            .pool => p.checkPool(gate),
            .rt => p.checkRt(gate),
            else => p.checkPack(gate),
        };
    }
};

/// Every registered variant, the launch-split probe after the gates and the equality with glue_tmp's kernels.
pub const Suite = struct {
    a: std.mem.Allocator,
    device: mtl.Device,
    probes: std.ArrayList(*Proj) = .empty,

    const E = struct { f: Fam, ctl: u8, name: []const u8 };
    const pool_name = base ++ "pool tf_dsv41_c_pool2";
    const rt_name = base ++ "fp4 round trip tf_dsv41_c_fp4rt";
    const keys_name = base ++ "fp4 pack keys tf_dsv41_c_fp4pack";
    const list = [_]E{
        .{ .f = .pool, .ctl = 0, .name = pool_name },
        .{ .f = .pool, .ctl = 1, .name = base ++ "control: pool non-finite inputs flushed" },
        .{ .f = .pool, .ctl = 2, .name = base ++ "control: pool carry not written back" },
        .{ .f = .pool, .ctl = 3, .name = base ++ "control: pool slot taken from the row, not the position" },
        .{ .f = .pool, .ctl = 4, .name = base ++ "control: pool writes past its output" },
        .{ .f = .rt, .ctl = 0, .name = rt_name },
        .{ .f = .rt, .ctl = 1, .name = base ++ "control: fp4 round trip NaN not flagged" },
        .{ .f = .rt, .ctl = 2, .name = base ++ "control: fp4 round trip scale left unrounded to E4M3" },
        .{ .f = .rt, .ctl = 4, .name = base ++ "control: fp4 round trip writes past its output" },
        .{ .f = .pack_keys, .ctl = 0, .name = keys_name },
        .{ .f = .pack_keys, .ctl = 1, .name = base ++ "control: fp4 pack NaN not flagged" },
        .{ .f = .pack_keys, .ctl = 2, .name = base ++ "control: fp4 pack scale exponent floored" },
        .{ .f = .pack_keys, .ctl = 3, .name = base ++ "control: fp4 pack ties rounded away from zero" },
        .{ .f = .pack_keys, .ctl = 4, .name = base ++ "control: fp4 pack writes past its output" },
        .{ .f = .pack_q, .ctl = 0, .name = base ++ "fp4 pack queries tf_dsv41_c_fp4pack" },
    };

    pub fn create(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, kernels: *std.ArrayList(g.Kernel), only: ?[]const u8) !*Suite {
        const s = try a.create(Suite);
        s.* = .{ .a = a, .device = device };
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

    fn shapesOf(f: Fam) []const c.Shape {
        return switch (f) {
            .pool => &.{ .{ .rows = 33, .n = 512, .k = 1 }, .{ .rows = 200, .n = 100, .k = 1 }, .{ .rows = 9, .n = 33, .k = 1 } },
            .rt => &.{ .{ .rows = 33, .n = 512, .k = 1 }, .{ .rows = 9, .n = 64, .k = 1 } },
            .pack_keys => &.{ .{ .rows = 33, .n = 128, .k = 1 }, .{ .rows = 9, .n = 256, .k = 1 } },
            .pack_q => &.{ .{ .rows = 20, .n = 8192, .k = 1 }, .{ .rows = 9, .n = 512, .k = 1 } },
        };
    }

    /// Launches of 1, 7 and rows/3 rows in order (the carry crossing every launch) equal the whole launch's bytes.
    fn probe(s: *Suite, p: *Proj, shape: c.Shape, seed: u64, bad: *usize, launches: *usize) !void {
        if (!try p.prepare(shape, .random, seed)) return error.Refused;
        try p.run();
        const whole = try s.a.dupe(u8, try p.rig.outputBytes());
        defer s.a.free(whole);
        for ([_]usize{ 1, 7, @max(1, shape.rows / 3) }) |step| {
            p.reset();
            const pass = p.rig.begin();
            var r0: usize = 0;
            while (r0 < shape.rows) : (r0 += step) {
                p.encode(pass.e, r0, @min(step, shape.rows - r0), false);
                launches.* += 1;
            }
            try p.rig.finish(pass);
            if (!std.mem.eql(u8, whole, try p.rig.outputBytes())) {
                bad.* += 1;
                std.debug.print("FAIL row invariance {s} {d}x{d}: {d} rows a launch differs from one launch\n", .{ p.name, shape.rows, shape.n, step });
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
            for (shapesOf(p.fam), 0..) |shape, i| try s.probe(p, shape, seed + i, &bad, &launches);
        }
        if (kernels == 0) return 0;
        std.debug.print("{s} row invariance {s}: {d} kernels at 1, 7 and rows/3 rows a launch, the pool's carry across launches ({d} launches) byte-equal to one launch, {d} differ\n", .{ if (bad == 0) "PASS" else "FAIL", base, kernels, launches, bad });
        return bad;
    }

    /// Our kernels and glue_tmp's give the same bytes on finite inputs, outputs and carry alike.
    pub fn equality(s: *Suite, seed: u64) !usize {
        var any = false;
        for (s.probes.items) |p| any = any or p.ctl == 0;
        if (!any) return 0;
        const lib = try mtl.Library.fromSource(s.device, sources.dsv41.glue_tmp, mtl.CompileOptions.mlx());
        defer lib.deinit();
        var old: cp.Pipelines = .{
            .pool = try mtl.Pipeline.init(s.device, lib, "tf_dsv41_g_pool2", false),
            .rt = try mtl.Pipeline.init(s.device, lib, "tf_dsv41_g_fp4rt", false),
            .pack = try mtl.Pipeline.init(s.device, lib, "tf_dsv41_g_fp4pack", false),
        };
        defer old.deinit();
        var cases: usize = 0;
        var differ: usize = 0;
        for (s.probes.items) |p| {
            if (p.ctl != 0) continue;
            p.tmp = old;
            defer p.tmp = null;
            const extra = [_]c.Shape{ .{ .rows = 700, .n = 512, .k = 1 }, .{ .rows = 64, .n = 8192, .k = 1 } };
            const lists = [_][]const c.Shape{ spec(p.fam, p.name).served, shapesOf(p.fam), &extra };
            var i: usize = 0;
            for (lists) |group| for (group) |shape| {
                i += 1;
                for (0..3) |k| {
                    if (!try p.prepare(shape, .random, seed + 100 * i + k)) continue;
                    p.reset();
                    var pass = p.rig.begin();
                    p.encode(pass.e, 0, shape.rows, false);
                    try p.rig.finish(pass);
                    const mine = try s.a.dupe(u8, try p.rig.outputBytes());
                    defer s.a.free(mine);
                    p.reset();
                    pass = p.rig.begin();
                    p.encode(pass.e, 0, shape.rows, true);
                    try p.rig.finish(pass);
                    cases += 1;
                    if (std.mem.indexOfDiff(u8, mine, try p.rig.outputBytes())) |at_byte| {
                        differ += 1;
                        std.debug.print("FAIL glue_tmp equality {s} {d}x{d}: first difference at byte {d}\n", .{ p.name, shape.rows, shape.n, at_byte });
                    }
                }
            };
        }
        std.debug.print("{s} equality {s}: {d} cases of the pool, the round trip and both packings byte-equal to glue_tmp's kernels on finite inputs, {d} differ\n", .{ if (differ == 0) "PASS" else "FAIL", base, cases, differ });
        return differ;
    }
};
