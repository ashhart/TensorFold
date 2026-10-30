//! DeepSeek V4.1 small matvecs on Metal: bf16 or fp32 weights, bf16 rows, bits independent of the launch.
const std = @import("std");
const mtl = @import("metal");
const sources = @import("kernel_sources");

pub const Ref = struct { buf: mtl.Buffer, off: usize = 0 };
pub const Weight = enum { bf16, f32 };
pub const Out = enum { f32, bf16 };
pub const max_tier_rows = 16;

pub const Call = struct { k: u32, n: u32, rows: u32, w: Weight = .bf16, out: Out = .f32 };

const Args = extern struct { k: u32, n: u32, rows: u32 };
/// Rows and columns a simdgroup computes in each launch geometry; the source instantiates the same pairs.
const tiers = [_]u32{ 1, 2, 4, 4, 8, 16 };
const cols = [_]u32{ 1, 1, 1, 2, 2, 2 };

fn name(comptime w: Weight, comptime o: Out, comptime t: usize) [:0]const u8 {
    const ws = if (w == .bf16) "bf16" else "f32";
    return std.fmt.comptimePrint("tf_dsv41_mv_{s}w_{s}o_r{d}c{d}", .{ ws, if (o == .f32) "f32" else "bf16", tiers[t], cols[t] });
}

/// The registered launch names, in the order of the pipelines.
pub const names = blk: {
    var out: [2 * 2 * tiers.len][:0]const u8 = undefined;
    var i: usize = 0;
    for ([_]Weight{ .bf16, .f32 }) |w| for ([_]Out{ .f32, .bf16 }) |o| for (0..tiers.len) |t| {
        out[i] = name(w, o, t);
        i += 1;
    };
    break :blk out;
};

pub const Pipelines = struct {
    p: [names.len]mtl.Pipeline,

    pub fn load(device: mtl.Device) !Pipelines {
        return loadSource(device, sources.deepseek_v41_mv);
    }

    /// The same kernels from other text: the contract's planted controls prefix a TF_CTL define.
    pub fn loadSource(device: mtl.Device, text: []const u8) !Pipelines {
        const lib = try mtl.Library.fromSource(device, text, mtl.CompileOptions.mlx());
        defer lib.deinit();
        var out: Pipelines = undefined;
        for (names, 0..) |n, i| out.p[i] = try mtl.Pipeline.init(device, lib, n, false);
        return out;
    }

    pub fn deinit(p: *Pipelines) void {
        for (p.p) |x| x.deinit();
    }
};

/// Columns from which a matvec is wide (the head): its weight reads dominate, so a simdgroup shares them over rows.
const wide_n = 8192;

/// The launch geometry of a call: any choice gives the same bits, so it follows only speed.
fn tierOf(rows: u32, n: u32) usize {
    if (rows > 16) return 5;
    if (n >= wide_n) return if (rows == 1) 0 else if (rows == 2) 1 else if (rows <= 4) 3 else 4;
    return if (rows <= 4) 0 else if (rows <= 8) 1 else 2;
}

pub fn check(c: Call) !void {
    if (c.rows == 0 or c.k == 0 or c.n == 0) return error.BadMvShape;
}

/// x [rows][k] bf16, w [n][k], y [rows][n]; one launch of ceil(n / cols) by ceil(rows / tier rows) simdgroups.
pub fn encode(p: *const Pipelines, e: mtl.ComputeEncoder, x: Ref, w: Ref, y: Ref, c: Call) !void {
    try check(c);
    const t = tierOf(c.rows, c.n);
    const at = (@as(usize, @intFromEnum(c.w)) * 2 + @intFromEnum(c.out)) * tiers.len + t;
    e.setPipeline(p.p[at]);
    e.setBuffer(x.buf, x.off, 0);
    e.setBuffer(w.buf, w.off, 1);
    e.setBuffer(y.buf, y.off, 2);
    e.setValue(Args{ .k = c.k, .n = c.n, .rows = c.rows }, 3);
    e.dispatchGroups(mtl.Size.of((c.n + cols[t] - 1) / cols[t], (c.rows + tiers[t] - 1) / tiers[t], 1), mtl.Size.of(32, 1, 1));
}
