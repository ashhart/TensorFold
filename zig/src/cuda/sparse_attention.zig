//! sparse_attention.cu for every CUDA family: query rows over listed cache rows (top-k, local and sink keys).
const std = @import("std");
const kernels = @import("kernels.zig");
const module = @import("module.zig");
const launch_ = @import("launch.zig");
const stream_ = @import("stream.zig");

const Driver = @import("driver.zig").Driver;
const Module = module.Module;
const Function = module.Function;
const Stream = stream_.Stream;

/// Head dim, query heads a key head at most (a warp's mma rows), warps a block and their K and V tiles' bytes.
pub const dim = 128;
pub const max_group = 16;
const warps = 4;
const smem: u32 = warps * 2 * 32 * dim * 2;
/// The largest list capacity: a row's count rounds up to whole 32-key tiles in i32, (n + 31) / 32.
pub const max_cap = std.math.maxInt(i32) - 31;

/// rows of h query heads over hk key heads; a (row, key head) lists up to `cap` of `keys` cache rows, `stride` apart.
pub const Shape = struct { rows: usize, h: usize, hk: usize, cap: usize, keys: usize, stride: usize, scale: f32 };

/// The shapes the kernel takes: whole groups of at most 16 query heads, rows a key head's 128 apart at least.
pub fn check(sh: Shape) error{SparseShape}!void {
    if (sh.hk == 0 or sh.h == 0 or sh.h % sh.hk != 0 or sh.h / sh.hk > max_group or sh.cap == 0) return error.SparseShape;
    if (sh.stride < sh.hk * dim or sh.stride % 8 != 0 or sh.rows > std.math.maxInt(i32) or sh.hk > 65535) return error.SparseShape;
    if (sh.cap > max_cap or sh.keys == 0 or sh.keys > std.math.maxInt(i32) or sh.stride > std.math.maxInt(i64)) return error.SparseShape;
}

pub const SparseAttention = struct {
    mod: Module,
    fn_: Function,

    pub fn load(d: *const Driver) !SparseAttention {
        if (!kernels.available) return error.BuiltWithoutKernels;
        var mod = try Module.load(d, kernels.sparse_attention);
        errdefer mod.unload();
        const f = try mod.function("tf_sparse_attention");
        try f.allowDynamicShared(smem);
        return .{ .mod = mod, .fn_ = f };
    }

    pub fn unload(g: *SparseAttention) void {
        g.mod.unload();
    }

    /// q, out [rows, h, 128] bf16; list i32 [rows, hk, cap]; counts taken in [0, cap], entries past [0, keys) dropped.
    pub fn run(g: *const SparseAttention, s: Stream, q: u64, k: u64, v: u64, list: u64, counts: u64, out: u64, sh: Shape) !void {
        try check(sh);
        if (sh.rows == 0) return;
        var a: launch_.Args = .{};
        for ([_]u64{ q, k, v, list, counts, out }) |x| a.add(x);
        a.add(sh.scale);
        for ([_]usize{ sh.h, sh.hk, sh.cap, sh.keys }) |x| a.add(@as(i32, @intCast(x)));
        a.add(@as(i64, @intCast(sh.stride)));
        const grid: launch_.Dim3 = .{ .x = @intCast(sh.rows), .y = @intCast(sh.hk), .z = 1 };
        try launch_.launch(g.fn_, .{ .grid = grid, .block = .{ .x = 32 * warps }, .shared = smem }, s, &a);
    }
};

test "check takes Kolibri 1.1's groups and refuses what the kernel can't fold" {
    const ok: Shape = .{ .rows = 3, .h = 48, .hk = 4, .cap = 2184, .keys = 17408, .stride = 4 * dim, .scale = 1 };
    try check(ok);
    var x = ok;
    x.h = 47;
    try std.testing.expectError(error.SparseShape, check(x));
    x = ok;
    x.h = 68; // 17 query heads a key head
    try std.testing.expectError(error.SparseShape, check(x));
    x = ok;
    x.cap = 0;
    try std.testing.expectError(error.SparseShape, check(x));
    x = ok;
    x.stride = 4 * dim - 8; // key heads would overlap
    try std.testing.expectError(error.SparseShape, check(x));
    x = ok;
    x.stride = 4 * dim + 4; // rows off the 16-byte copies
    try std.testing.expectError(error.SparseShape, check(x));
    x = ok;
    x.hk = 0;
    try std.testing.expectError(error.SparseShape, check(x));
    x = ok;
    x.h = 0; // no query heads: nothing would be written
    try std.testing.expectError(error.SparseShape, check(x));
    x = ok;
    x.keys = 0;
    try std.testing.expectError(error.SparseShape, check(x));
    x = ok;
    x.keys = @as(usize, std.math.maxInt(i32)) + 1; // entries are i32
    try std.testing.expectError(error.SparseShape, check(x));
    x = ok;
    x.cap = max_cap; // its count rounds up to whole 32-key tiles in i32
    try check(x);
    x.cap += 1;
    try std.testing.expectError(error.SparseShape, check(x));
}
