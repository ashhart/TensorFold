//! DeepSeek V4.1 release FP8 dense rows (1-16 rows): pipelines and the launch encoder, wired into no forward.
const std = @import("std");
const mtl = @import("metal");
const sources = @import("kernel_sources");

pub const max_rows = 16;
pub const max_tiles = 32;
pub const Ref = struct { buf: mtl.Buffer, off: usize = 0 };
pub const Out = enum { f32, bf16 };

/// split 1: a tile is 32 K groups across a simdgroup's lanes; split 4: 8 groups, four columns a simdgroup.
pub const Call = struct { k: u32, n: u32, rows: u32, split: u32 = 1, out: Out = .f32 };

const Args = extern struct { K: u32, N: u32, R: u32, groups: u32, tiles: u32, blocks: u32 };
const row_tiers = [_]u32{ 1, 2, 4, 8, 16 };
/// The one-launch kernels' row tiers.
const whole_tiers = [_]u32{ 1, 2, 3, 4, 5, 6, 8, 16 };
const a_names = [_][:0]const u8{ "tf_dsv41_fp8_a_r1", "tf_dsv41_fp8_a_r2", "tf_dsv41_fp8_a_r4", "tf_dsv41_fp8_a_r8", "tf_dsv41_fp8_a_r16" };
const b_names = [_][:0]const u8{ "tf_dsv41_fp8_b_r1", "tf_dsv41_fp8_b_r2", "tf_dsv41_fp8_b_r4", "tf_dsv41_fp8_b_r8", "tf_dsv41_fp8_b_r16" };

/// The one-launch kernels by split (a: 1, b: 4), row tier and output (f32, bf16).
fn wholeName(comptime split: u8, comptime t: u32, comptime out: Out) [:0]const u8 {
    return std.fmt.comptimePrint("tf_dsv41_fp8_w{c}_r{d}_{s}", .{ split, t, @tagName(out) });
}

pub const Pipelines = struct {
    a: [5]mtl.Pipeline,
    b: [5]mtl.Pipeline,
    f32: mtl.Pipeline,
    bf16: mtl.Pipeline,
    /// [Kind][tier][out f32, bf16]
    whole: [3][whole_tiers.len][2]mtl.Pipeline,

    pub fn load(device: mtl.Device) !Pipelines {
        return loadSource(device, sources.dsv41.fp8_rows);
    }

    /// The same kernels from other text: the contract's planted controls prefix a TF_CTL define.
    pub fn loadSource(device: mtl.Device, text: []const u8) !Pipelines {
        const lib = try mtl.Library.fromSource(device, text, mtl.CompileOptions.mlx());
        defer lib.deinit();
        var p: Pipelines = undefined;
        for (a_names, 0..) |name, i| p.a[i] = try mtl.Pipeline.init(device, lib, name, false);
        for (b_names, 0..) |name, i| p.b[i] = try mtl.Pipeline.init(device, lib, name, false);
        p.f32 = try mtl.Pipeline.init(device, lib, "tf_dsv41_fp8_finish_f32", false);
        p.bf16 = try mtl.Pipeline.init(device, lib, "tf_dsv41_fp8_finish_bf16", false);
        inline for (.{ 'a', 'b', 'c' }, 0..) |sp, i| inline for (whole_tiers, 0..) |t, j| inline for (.{ Out.f32, Out.bf16 }, 0..) |o, k| {
            p.whole[i][j][k] = try mtl.Pipeline.init(device, lib, wholeName(sp, t, o), false);
        };
        return p;
    }

    pub fn deinit(p: *Pipelines) void {
        for (p.a) |x| x.deinit();
        for (p.b) |x| x.deinit();
        p.f32.deinit();
        p.bf16.deinit();
        for (p.whole) |sp| for (sp) |t| for (t) |x| x.deinit();
    }
};

fn tier(rows: u32) usize {
    for (row_tiers, 0..) |t, i| if (rows <= t) return i;
    unreachable;
}

fn wholeTier(rows: u32) usize {
    for (whole_tiers, 0..) |t, i| if (rows <= t) return i;
    unreachable;
}

pub fn tilesOf(k: u32, split: u32) u32 {
    const per: u32 = if (split == 1) 32 else 8;
    return (k / 32 + per - 1) / per;
}

/// The split of a K shape: 4 (four times the simdgroups, every lane busy) while its tiles fit one group, else 1.
pub fn splitOf(k: u32) u32 {
    return if (tilesOf(k, 4) <= max_tiles) 4 else 1;
}

pub fn check(c: Call) !void {
    if (c.rows == 0 or c.rows > max_rows or c.k == 0 or c.k % 32 != 0 or c.n == 0 or c.n % 32 != 0) return error.BadFp8Shape;
    if (c.split != 1 and c.split != 4) return error.BadFp8Split;
    if (tilesOf(c.k, c.split) > max_tiles) return error.BadFp8Shape;
}

/// Bytes of scratch a call needs: one fp32 root a tile, column and row.
pub fn scratchBytes(c: Call) usize {
    return @as(usize, tilesOf(c.k, c.split)) * c.n * c.rows * 4;
}

/// X [rows][K] bf16, W [N][K] E4M3, S [N/32][K/32] E8M0, Y [rows][N]; bits do not depend on rows or split.
pub fn encode(p: *const Pipelines, e: mtl.ComputeEncoder, x: Ref, w: Ref, s: Ref, scratch: Ref, y: Ref, c: Call) !void {
    try check(c);
    const tiles = tilesOf(c.k, c.split);
    const args = Args{ .K = c.k, .N = c.n, .R = c.rows, .groups = c.k / 32, .tiles = tiles, .blocks = c.n / 4 };
    const at = tier(c.rows);
    e.setPipeline(if (c.split == 1) p.a[at] else p.b[at]);
    e.setBuffer(x.buf, x.off, 0);
    e.setBuffer(w.buf, w.off, 1);
    e.setBuffer(s.buf, s.off, 2);
    e.setBuffer(scratch.buf, scratch.off, 3);
    e.setValue(args, 4);
    e.dispatchGroups(mtl.Size.of(args.blocks * tiles, 1, 1), mtl.Size.of(32, 1, 1));
    e.barrier();
    e.setPipeline(if (c.out == .f32) p.f32 else p.bf16);
    e.setBuffer(scratch.buf, scratch.off, 0);
    e.setBuffer(y.buf, y.off, 1);
    e.setValue(args, 2);
    e.dispatchThreads(mtl.Size.of(c.n, c.rows, 1), mtl.Size.of(32, 1, 1));
    // the scratch is free for the next call once the finish has read it
    e.barrier();
}

/// Launches of one call shape in grid y: the byte strides from one launch's x, weight, scale and output to the next.
pub const Batch = struct { n: u32 = 1, xs: i64 = 0, ws: i64 = 0, ss: i64 = 0, ys: i64 = 0 };

const WholeArgs = extern struct { K: u32, N: u32, R: u32, groups: u32, tiles: u32, blocks: u32, xs: i64, ws: i64, ss: i64, ys: i64 };

/// A one-launch block: a (split 1, 4 columns), b (split 4, 4 columns), c (split 4, 8 columns).
pub const Kind = enum(u2) {
    a,
    b,
    c,

    pub fn columns(k: Kind) u32 {
        return if (k == .c) 8 else 4;
    }
};

/// The block a call launches with: 8 columns (two a lane share each activation load) for short K up to 4 rows.
pub fn kindOf(c: Call) Kind {
    if (c.split == 1) return .a;
    const most: u32 = if (c.rows == 1) 9 else if (c.rows <= 4) 8 else 0;
    return if (tilesOf(c.k, c.split) <= most) .c else .b;
}

/// `encode`'s two launches as one: a group of `tiles` simdgroups a 4-column block, the same bits, no scratch.
pub fn encodeWhole(p: *const Pipelines, e: mtl.ComputeEncoder, x: Ref, w: Ref, s: Ref, y: Ref, c: Call, b: Batch) !void {
    return encodeWholeAs(p, e, x, w, s, y, c, b, kindOf(c));
}

/// `encodeWhole` with the block chosen by the caller; every kind gives the same bits.
pub fn encodeWholeAs(p: *const Pipelines, e: mtl.ComputeEncoder, x: Ref, w: Ref, s: Ref, y: Ref, c: Call, b: Batch, kind: Kind) !void {
    try check(c);
    if (b.n == 0 or (kind == .a) != (c.split == 1) or c.n % kind.columns() != 0) return error.BadFp8Shape;
    const tiles = tilesOf(c.k, c.split);
    const args = WholeArgs{ .K = c.k, .N = c.n, .R = c.rows, .groups = c.k / 32, .tiles = tiles, .blocks = c.n / kind.columns(), .xs = b.xs, .ws = b.ws, .ss = b.ss, .ys = b.ys };
    e.setPipeline(p.whole[@intFromEnum(kind)][wholeTier(c.rows)][@intFromEnum(c.out)]);
    e.setBuffer(x.buf, x.off, 0);
    e.setBuffer(w.buf, w.off, 1);
    e.setBuffer(s.buf, s.off, 2);
    e.setBuffer(y.buf, y.off, 3);
    e.setValue(args, 4);
    e.dispatchGroups(mtl.Size.of(args.blocks, b.n, 1), mtl.Size.of(32 * tiles, 1, 1));
}
