//! FP8 in a compressed-tensors export's math (ck_lane8t.cu): e4m3 rows a token's scale, weights a column's, K whole.
const std = @import("std");
const driver = @import("driver.zig");
const module = @import("module.zig");
const launch_ = @import("launch.zig");
const stream_ = @import("stream.zig");
const kernels = @import("kernels.zig");

const Driver = driver.Driver;
const Module = module.Module;
const Function = module.Function;
const Stream = stream_.Stream;

/// A projection: e4m3 codes in the FP8 pack's order (modelopt.Packer.fp8Lane), an fp32 scale a column.
pub const Weight = struct { codes: u64, cs: u64, n: usize, k: usize };

/// The rows' tile by count: 16, 64 or 128 rows; every tile sums an output in the same order.
pub fn tileRows(m: usize) usize {
    return if (m <= 16) 16 else if (m <= 64) 64 else 128;
}

/// Dynamic shared bytes of a `bm`-row tile: four stages of the rows and one 64-column weight tile.
pub fn sharedBytes(bm: usize) u32 {
    return @intCast(4 * ((bm * 64 + 64 * 64 + 127) / 128 * 128));
}

pub const Lane8t = struct {
    mod: Module,
    tiles: [3]Function, // 16, 64, 128 rows
    quant: Function,

    pub fn load(d: *const Driver) !Lane8t {
        if (!kernels.available) return error.BuiltWithoutKernels;
        var mod = try Module.load(d, kernels.ck_lane8t);
        errdefer mod.unload();
        var l: Lane8t = .{ .mod = mod, .tiles = undefined, .quant = try mod.function("tf_ck_quant8t") };
        for ([_][:0]const u8{ "tf_ck8t_16", "tf_ck8t_64", "tf_ck8t_128" }, [_]usize{ 16, 64, 128 }, &l.tiles) |name, bm, *f| {
            f.* = try mod.function(name);
            try f.allowDynamicShared(sharedBytes(bm));
        }
        return l;
    }

    pub fn unload(l: *Lane8t) void {
        l.mod.unload();
    }

    /// bf16 rows [m, k] (`ldx` apart) to e4m3 codes [m, k] under scale amax / 448 a row (fp32 at `rs`).
    pub fn quantRows(l: *const Lane8t, s: Stream, x: u64, ldx: usize, m: usize, k: usize, codes: u64, rs: u64) !void {
        if (k % 16 != 0 or m == 0) return error.UnexpectedTensor;
        var a: launch_.Args = .{};
        a.add(x);
        a.add(@as(i32, @intCast(ldx)));
        a.add(@as(i32, @intCast(k)));
        a.add(codes);
        a.add(rs);
        try launch_.launch(l.quant, .{ .grid = .{ .x = @intCast(m) }, .block = .{ .x = 128 } }, s, &a);
    }

    /// out [m, n] bf16 = (codes x w) * rs[row] * cs[col]; the tile by `rows_tile` (16, 64, 128) or the count.
    pub fn matmulTile(l: *const Lane8t, s: Stream, codes: u64, rs: u64, m: usize, w: Weight, out: u64, rows_tile: ?usize) !void {
        if (w.n % 64 != 0 or w.k % 64 != 0) return error.UnexpectedTensor;
        if (m == 0) return;
        const bm = rows_tile orelse tileRows(m);
        const ti: usize = switch (bm) {
            16 => 0,
            64 => 1,
            128 => 2,
            else => return error.UnexpectedTensor,
        };
        const rows_t = (m + bm - 1) / bm;
        const group = @max(1, @min(rows_t, (12 << 20) / (bm * w.k)));
        var a: launch_.Args = .{};
        for ([_]u64{ codes, rs, w.codes, w.cs, out }) |v| a.add(v);
        for ([_]usize{ m, w.n, w.k, group }) |v| a.add(@as(i32, @intCast(v)));
        try launch_.launch(l.tiles[ti], .{ .grid = .{ .x = @intCast(rows_t * (w.n / 64)) }, .block = .{ .x = if (bm == 128) 256 else 128 }, .shared = sharedBytes(bm) }, s, &a);
    }

    pub fn matmul(l: *const Lane8t, s: Stream, codes: u64, rs: u64, m: usize, w: Weight, out: u64) !void {
        return l.matmulTile(s, codes, rs, m, w, out, null);
    }
};

test "tiles and their shared bytes" {
    try std.testing.expectEqual(@as(usize, 16), tileRows(1));
    try std.testing.expectEqual(@as(usize, 64), tileRows(17));
    try std.testing.expectEqual(@as(usize, 128), tileRows(65));
    try std.testing.expectEqual(@as(u32, 49152), sharedBytes(128));
    try std.testing.expectEqual(@as(u32, 20480), sharedBytes(16));
}
