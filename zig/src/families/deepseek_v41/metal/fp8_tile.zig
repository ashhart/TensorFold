//! DeepSeek V4.1 release FP8 dense rows, up to 16, on simdgroup MMA: each row's bits are the row kernel's.
const std = @import("std");
const mtl = @import("metal");
const sources = @import("kernel_sources");

pub const Ref = struct { buf: mtl.Buffer, off: usize = 0 };
pub const Out = enum { f32, bf16 };
pub const Call = struct { k: u32, n: u32, rows: u32, out: Out = .f32 };
/// Launches of one call shape in grid y: the byte strides from one launch's x, weight, scale and output to the next.
pub const Batch = struct { n: u32 = 1, xs: i64 = 0, ws: i64 = 0, ss: i64 = 0, ys: i64 = 0 };

pub const max_rows = 16;
/// Simdgroups a threadgroup's K split takes at most, times its 8-column blocks (they share the staging memory).
pub const max_lanes = 16;
const max_per = 512;

/// How a launch is cut: 8-column blocks a threadgroup (1 or 2) and simdgroups splitting K (a power of two).
pub const Form = struct { nj: u32, sgs: u32 };

const Args = extern struct { K: u32, N: u32, R: u32, groups: u32, per: u32, sgs: u32, xs: i64, ws: i64, ss: i64, ys: i64 };

pub const Pipelines = struct {
    /// [8-row tiles - 1][8-column blocks - 1][out f32, bf16]
    p: [2][2][2]mtl.Pipeline,

    pub fn load(device: mtl.Device) !Pipelines {
        return loadSource(device, sources.dsv41.fp8_tile);
    }

    /// The same kernels from other text: the contract's planted controls prefix a TF_CTL define.
    pub fn loadSource(device: mtl.Device, text: []const u8) !Pipelines {
        const lib = try mtl.Library.fromSource(device, text, mtl.CompileOptions.mlx());
        defer lib.deinit();
        var out: Pipelines = undefined;
        var made: usize = 0;
        errdefer for (0..made) |i| out.p[i / 4][(i / 2) % 2][i % 2].deinit();
        inline for (.{ "r8", "r16" }, 0..) |t, i| inline for (.{ "c8", "c16" }, 0..) |cb, j| inline for (.{ "f32", "bf16" }, 0..) |o, k| {
            out.p[i][j][k] = try mtl.Pipeline.init(device, lib, "tf_dsv41_fp8_tile_" ++ t ++ "_" ++ cb ++ "_" ++ o, false);
            made += 1;
        };
        return out;
    }

    pub fn deinit(p: *Pipelines) void {
        for (p.p) |t| for (t) |c| for (c) |x| x.deinit();
    }
};

fn padded(groups: u32) u32 {
    return std.math.ceilPowerOfTwo(u32, groups) catch unreachable;
}

/// A shape's form: 16 columns for the widest N, each simdgroup an aligned run of at least 8 groups; bits never change.
pub fn formOf(k: u32, n: u32) Form {
    return formFor(k, n, 1);
}

/// `formOf` for a batch: 16 columns once its launches hold 2048 or more columns between them.
pub fn formFor(k: u32, n: u32, launches: u32) Form {
    const nj: u32 = if (n * launches >= 2048 and n % 16 == 0) 2 else 1;
    return .{ .nj = nj, .sgs = std.math.clamp(padded(k / 32) / 8, 1, max_lanes / nj) };
}

fn valid(c: Call, f: Form) bool {
    if (f.nj != 1 and f.nj != 2) return false;
    if (f.sgs == 0 or f.sgs * f.nj > max_lanes or !std.math.isPowerOfTwo(f.sgs) or f.sgs > padded(c.k / 32)) return false;
    return c.n % (8 * f.nj) == 0 and padded(c.k / 32) / f.sgs <= max_per;
}

/// Whether the tile can run a call: 1-16 rows, whole 32x32 blocks, its slices within the tree it holds.
pub fn fits(c: Call) bool {
    if (c.rows == 0 or c.rows > max_rows or c.k == 0 or c.k % 32 != 0 or c.n == 0 or c.n % 32 != 0) return false;
    return valid(c, formOf(c.k, c.n));
}

/// Rows the decode forward takes this tile from for `launches` launches of a call (above max_rows: never), by its
/// columns: below 1024 the row kernel reads the weights faster at every count; wide K = 1280 and 2304 from 5 rows.
pub fn minRows(c: Call, launches: u32) u32 {
    const cols = c.n * launches;
    if (cols < 1024) return max_rows + 1;
    if (cols < 2048) return 7;
    return if (c.k % 1024 == 0) 6 else 5;
}

/// Whether the decode forward runs a call here (by its rows and columns only); a row's bits are the same either way.
pub fn takes(c: Call, launches: u32) bool {
    return c.rows >= minRows(c, launches) and fits(c) and valid(c, formFor(c.k, c.n, launches));
}

/// X [rows][K] bf16, W [N][K] E4M3, S [N/32][K/32] E8M0, Y [rows][N]; equal to fp8_rows at any row count or form.
pub fn encode(p: *const Pipelines, e: mtl.ComputeEncoder, x: Ref, w: Ref, s: Ref, y: Ref, c: Call, b: Batch) !void {
    return encodeAs(p, e, x, w, s, y, c, b, formFor(c.k, c.n, b.n));
}

/// `encode` with the form chosen by the caller; every valid form gives the same bits.
pub fn encodeAs(p: *const Pipelines, e: mtl.ComputeEncoder, x: Ref, w: Ref, s: Ref, y: Ref, c: Call, b: Batch, f: Form) !void {
    if (!fits(c) or b.n == 0) return error.BadFp8Shape;
    if (!valid(c, f)) return error.BadFp8Split;
    const groups = c.k / 32;
    const args = Args{ .K = c.k, .N = c.n, .R = c.rows, .groups = groups, .per = padded(groups) / f.sgs, .sgs = f.sgs, .xs = b.xs, .ws = b.ws, .ss = b.ss, .ys = b.ys };
    e.setPipeline(p.p[if (c.rows > 8) 1 else 0][f.nj - 1][@backingInt(c.out)]);
    e.setBuffer(x.buf, x.off, 0);
    e.setBuffer(w.buf, w.off, 1);
    e.setBuffer(s.buf, s.off, 2);
    e.setBuffer(y.buf, y.off, 3);
    e.setValue(args, 4);
    e.dispatchGroups(mtl.Size.of(c.n / (8 * f.nj), b.n, 1), mtl.Size.of(32 * f.sgs, 1, 1));
}

test "the tile's form covers every release shape within its tree" {
    for ([_]u32{ 512, 1280, 2304, 4096, 5120, 6144, 8192, 15360 }) |k| {
        const c: Call = .{ .k = k, .n = 2048, .rows = 6 };
        try std.testing.expect(takes(c, 1));
        for ([_]Form{ formOf(k, 64), formFor(k, 2048, 1) }) |f| try std.testing.expect(valid(c, f) and padded(k / 32) % f.sgs == 0);
    }
    try std.testing.expect(!takes(.{ .k = 5120, .n = 1280, .rows = 6 }, 1) and takes(.{ .k = 5120, .n = 1280, .rows = 7 }, 1));
    try std.testing.expect(!takes(.{ .k = 5120, .n = 512, .rows = 16 }, 1) and fits(.{ .k = 5120, .n = 64, .rows = 1 }));
    try std.testing.expect(takes(.{ .k = 1280, .n = 4096, .rows = 5 }, 1) and !takes(.{ .k = 4096, .n = 1024, .rows = 5 }, 8));
    try std.testing.expect(!fits(.{ .k = 5120, .n = 64, .rows = 17 }));
    try std.testing.expect(!valid(.{ .k = 5120, .n = 64, .rows = 2 }, .{ .nj = 2, .sgs = 16 }));
}
