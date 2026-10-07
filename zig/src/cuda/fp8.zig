//! Block-FP8 projections (128 x 128 e4m3 blocks, fp32 scales) on fp8_lane.cu, chosen as the Python lane matmul chooses.
const std = @import("std");
const driver = @import("driver.zig");
const module = @import("module.zig");
const launch_ = @import("launch.zig");
const memory = @import("memory.zig");
const stream_ = @import("stream.zig");
const kernels = @import("kernels.zig");

const Driver = driver.Driver;
const Module = module.Module;
const Function = module.Function;
const Stream = stream_.Stream;

pub const block = 128; // a checkpoint scale's rows and inputs
pub const group = 64; // inputs a kernel stage reads, one column scale each
pub const fused_rows = 256; // rows from which a projection's K slices meet in one block (linear.py FUSED_ROWS)
const bn = 64;
const threads = 128; // WM 1 x WN 4 warps

/// The dynamic shared bytes of a BM-row tile (qmmf.cu Tile<FP8G, BM, 64, 1, 4, 4>::SMEM).
pub fn sharedBytes(bm: u32) u32 {
    const stage = (bm * group * 2 + bn * group + bn * 4 + 127) / 128 * 128;
    const partials = (bm / 16) * 2 * 4 * threads * 4;
    return @max(4 * stage, partials);
}

/// K slices for an (n, k) weight, from the shape alone so a row's bits never depend on the row count (qmm.split_k).
pub fn splitK(n: usize, k: usize) u32 {
    const tiles = (n + bn - 1) / bn;
    const groups = k / group;
    var sk: usize = 1;
    while (sk < 8 and tiles * sk < 192 and groups % (sk * 2) == 0 and groups / (sk * 2) >= 8) sk *= 2;
    return @intCast(sk);
}

/// The row tile: 16 or 32 rows, else 64-row tiles side by side (qmm.bucket).
pub fn bucket(m: usize) u32 {
    return if (m <= 16) 16 else if (m <= 32) 32 else 64;
}

/// Rows padded to whole 128-row blocks, as the kernel's 64-column tiles read them.
pub fn padded(n: usize) usize {
    return (n + block - 1) / block * block;
}

/// The k of byte `byte` of `lane` in a k32 step (fragment_index's kin).
fn kin(lane: usize, byte: usize) usize {
    const p = 16 * (byte / 4) + 4 * (lane % 4) + byte % 4;
    return 16 * (p / 16) + 2 * ((p % 16) / 4) + p % 2 + 8 * ((p % 4) / 2);
}

/// e4m3 codes [n, k] in the kernel's order [npad/64][k/64][8][32][2][8], rows past n zero (linear._fragment_order).
pub fn packCodes(out: []u8, codes: []const u8, n: usize, k: usize) void {
    const npad = padded(n);
    std.debug.assert(out.len == npad * k and codes.len == n * k and k % group == 0);
    var i: usize = 0;
    for (0..npad / 64) |t| for (0..k / group) |g| for (0..8) |a| for (0..32) |lane| for (0..2) |h| for (0..8) |b| {
        const row = t * 64 + a * 8 + lane / 4;
        const col = g * group + h * 32 + kin(lane, b);
        out[i] = if (row < n) codes[row * k + col] else 0;
        i += 1;
    };
}

/// Block scales [ceil(n/128), k/128] as the kernel's tiles [npad/64][k/64][64] of fp32, rows past n one (from_rows).
pub fn packScales(out: []f32, scale_inv: []const f32, n: usize, k: usize) void {
    const npad = padded(n);
    const kg = k / group;
    std.debug.assert(out.len == npad * kg and scale_inv.len == ((n + block - 1) / block) * (k / block));
    for (0..npad / 64) |t| for (0..kg) |g| for (0..64) |c| {
        const row = t * 64 + c;
        out[(t * kg + g) * 64 + c] = if (row < n) scale_inv[(row / block) * (k / block) + g * group / block] else 1.0;
    };
}

/// A packed projection on the device: codes and scale tiles.
pub const Weight = struct { codes: u64, scales: u64, n: u32, k: u32, npad: u32 };

/// fp8_lane's instances, by row tile and cluster, then the fused prompt kernel and the slice reduce.
pub const symbols = struct {
    pub const lane16 = "_ZN11tf_fp8_lane11qmmf_kernelILi3ELi16ELi64ELi1ELi4ELi4ELb0ELb0ELb0EEEvPK13__nv_bfloat16PKhS5_fPvPfiiiiiii";
    pub const lane16c = "_ZN11tf_fp8_lane11qmmf_kernelILi3ELi16ELi64ELi1ELi4ELi4ELb0ELb1ELb0EEEvPK13__nv_bfloat16PKhS5_fPvPfiiiiiii";
    pub const lane32 = "_ZN11tf_fp8_lane11qmmf_kernelILi3ELi32ELi64ELi1ELi4ELi4ELb0ELb0ELb0EEEvPK13__nv_bfloat16PKhS5_fPvPfiiiiiii";
    pub const lane32c = "_ZN11tf_fp8_lane11qmmf_kernelILi3ELi32ELi64ELi1ELi4ELi4ELb0ELb1ELb0EEEvPK13__nv_bfloat16PKhS5_fPvPfiiiiiii";
    pub const lane64 = "_ZN11tf_fp8_lane11qmmf_kernelILi3ELi64ELi64ELi1ELi4ELi4ELb0ELb0ELb0EEEvPK13__nv_bfloat16PKhS5_fPvPfiiiiiii";
    pub const lane64c = "_ZN11tf_fp8_lane11qmmf_kernelILi3ELi64ELi64ELi1ELi4ELi4ELb0ELb1ELb0EEEvPK13__nv_bfloat16PKhS5_fPvPfiiiiiii";
    pub const fused = "_ZN11tf_fp8_lane11qmmf_kernelILi3ELi64ELi64ELi1ELi4ELi4ELb0ELb0ELb1EEEvPK13__nv_bfloat16PKhS5_fPvPfiiiiiii";
    pub const reduce = "_ZN11tf_fp8_lane13reduce_kernelILb0EEEvPKfPvxif";
};

/// What one call launches, from the shapes alone: tile, slices, cluster or reduce (qmmf_cuda's choice).
pub const Plan = struct { bm: u32, sk: u32, fused: bool, cluster: bool, reduce: bool };

pub fn plan(m: usize, n: usize, k: usize, clusters: bool) Plan {
    const sk = splitK(n, k);
    const fused = sk > 1 and m >= fused_rows;
    const cluster = !fused and sk > 1 and sk <= 8 and clusters;
    return .{ .bm = if (fused) 64 else bucket(m), .sk = sk, .fused = fused, .cluster = cluster, .reduce = !fused and sk > 1 and !cluster };
}

pub const Lane = struct {
    mod: Module,
    fns: [8]Function, // lane16, lane16c, lane32, lane32c, lane64, lane64c, fused, reduce
    clusters: bool, // sm_90 on: slices meet in a cluster's shared memory

    pub fn load(d: *const Driver, capability_major: u32) !Lane {
        if (!kernels.available) return error.BuiltWithoutKernels;
        var mod = try Module.load(d, kernels.fp8_lane);
        errdefer mod.unload();
        var l: Lane = .{ .mod = mod, .fns = undefined, .clusters = capability_major >= 9 };
        inline for (.{ symbols.lane16, symbols.lane16c, symbols.lane32, symbols.lane32c, symbols.lane64, symbols.lane64c, symbols.fused, symbols.reduce }, 0..) |s, i| l.fns[i] = try mod.function(s);
        for ([_]u32{ 16, 16, 32, 32, 64, 64, 64 }, l.fns[0..7]) |bm, f| try f.allowDynamicShared(sharedBytes(bm));
        return l;
    }

    pub fn unload(l: *Lane) void {
        l.mod.unload();
    }

    /// Slice partials a call needs (fp32 [sk][m][n]) when it reduces them, else 0.
    pub fn partBytes(l: Lane, m: usize, w: Weight) usize {
        const p = plan(m, w.n, w.k, l.clusters);
        return if (p.reduce) @as(usize, p.sk) * m * w.n * 4 else 0;
    }

    /// y [m, n] bf16 = x [m, k] bf16 (rows `ldx` apart) times the weight, fp32 sums; `part` holds partialBytes.
    pub fn matmul(l: Lane, s: Stream, x: u64, ldx: usize, m: usize, w: Weight, y: u64, part: u64) !void {
        const p = plan(m, w.n, w.k, l.clusters);
        const f = if (p.fused) l.fns[6] else switch (p.bm) {
            16 => if (p.cluster) l.fns[1] else l.fns[0],
            32 => if (p.cluster) l.fns[3] else l.fns[2],
            else => if (p.cluster) l.fns[5] else l.fns[4],
        };
        const rows_t = (m + p.bm - 1) / p.bm;
        const band = @max(1, @min(rows_t, (12 << 20) / (@as(usize, p.bm) * w.k * 2)));
        var args: launch_.Args = .{};
        args.add(x);
        args.add(w.codes);
        args.add(w.scales);
        args.add(@as(f32, 1.0));
        args.add(y);
        args.add(if (p.reduce) part else @as(u64, 0));
        args.add(@as(i32, @intCast(m)));
        args.add(@as(i32, @intCast(w.n)));
        args.add(@as(i32, @intCast(w.k)));
        args.add(@as(i32, @intCast(p.sk)));
        args.add(@as(i32, @intCast(w.npad)));
        args.add(@as(i32, @intCast(if (m == 1) w.k else ldx)));
        args.add(@as(i32, @intCast(band)));
        try launch_.launch(f, .{
            .grid = .{ .x = @intCast(rows_t * ((w.n + bn - 1) / bn)), .y = 1, .z = if (p.fused) 1 else p.sk },
            .block = .{ .x = threads, .y = 1, .z = 1 },
            .shared = sharedBytes(p.bm),
            .cluster = if (p.cluster) .{ .x = 1, .y = 1, .z = p.sk } else null,
        }, s, &args);
        if (p.reduce) {
            const total: usize = m * w.n;
            var r: launch_.Args = .{};
            r.add(part);
            r.add(y);
            r.add(@as(i64, @intCast(total)));
            r.add(@as(i32, @intCast(p.sk)));
            r.add(@as(f32, 1.0));
            try launch_.launch(l.fns[7], .{ .grid = .{ .x = @intCast((total + 255) / 256), .y = 1, .z = 1 }, .block = .{ .x = 256, .y = 1, .z = 1 } }, s, &r);
        }
    }
};

test "tile shared bytes, slices and row tiles as the Python host picks them" {
    try std.testing.expectEqual(@as(u32, 25600), sharedBytes(16));
    try std.testing.expectEqual(@as(u32, 33792), sharedBytes(32));
    try std.testing.expectEqual(@as(u32, 50176), sharedBytes(64));
    // Kolibri 1's projections: qkv [7168, 2560], o [2560, 6144]
    try std.testing.expectEqual(@as(u32, 2), splitK(7168, 2560)); // values from qmm.split_k itself
    try std.testing.expectEqual(@as(u32, 8), splitK(2560, 6144));
    try std.testing.expectEqual(@as(u32, 4), splitK(512, 2560));
    try std.testing.expectEqual(@as(u32, 1), splitK(2560, 512));
    try std.testing.expectEqual(@as(u32, 1), splitK(32768, 2560));
    try std.testing.expectEqual(@as(u32, 16), bucket(1));
    try std.testing.expectEqual(@as(u32, 32), bucket(17));
    try std.testing.expectEqual(@as(u32, 64), bucket(300));
    const p = plan(300, 2560, 6144, true);
    try std.testing.expect(p.fused and p.bm == 64 and !p.cluster and !p.reduce);
    const q = plan(3, 2560, 6144, true);
    try std.testing.expect(!q.fused and q.cluster and q.bm == 16 and q.sk == 8);
    const r = plan(3, 2560, 6144, false);
    try std.testing.expect(r.reduce and !r.cluster);
}

test "the code repack is a permutation that puts each byte where fragment_index says" {
    const gpa = std.testing.allocator;
    const n = 130;
    const k = 128;
    const codes = try gpa.alloc(u8, n * k);
    defer gpa.free(codes);
    for (codes, 0..) |*c, i| c.* = @truncate(i * 7 + 1);
    const out = try gpa.alloc(u8, padded(n) * k);
    defer gpa.free(out);
    packCodes(out, codes, n, k);
    // byte (t=2, g=1, a=0, lane=9, h=1, b=5): row 128 + 9/4 = 130 -> padding; (t=0, g=1, a=3, lane=9, h=1, b=5): row 26
    const idx = ((((0 * 2 + 1) * 8 + 3) * 32 + 9) * 2 + 1) * 8 + 5;
    try std.testing.expectEqual(codes[26 * k + 64 + 32 + kin(9, 5)], out[idx]);
    var zeros: usize = 0;
    for (out) |b| zeros += @intFromBool(b == 0);
    try std.testing.expect(zeros >= (padded(n) - n) * k);
    // kin covers a k32 step once for each lane's quad: 32 inputs from 4 lanes x 8 bytes
    var seen: [32]bool = @splat(false);
    for (0..4) |lane| for (0..8) |b| {
        try std.testing.expect(!seen[kin(lane, b)]);
        seen[kin(lane, b)] = true;
    };
}

test "scale tiles: each 64-input group takes its 128 x 128 block's scale, padded rows one" {
    const n = 130;
    const k = 256;
    var inv: [2 * 2]f32 = .{ 1, 2, 3, 4 };
    var out: [256 * 4]f32 = undefined;
    packScales(&out, &inv, n, k);
    try std.testing.expectEqual(@as(f32, 2), out[(0 * 4 + 3) * 64 + 5]); // row 5, group 3 -> block (0, 1)
    try std.testing.expectEqual(@as(f32, 3), out[(2 * 4 + 0) * 64 + 1]); // row 129 -> block (1, 0)
    try std.testing.expectEqual(@as(f32, 1), out[(2 * 4 + 1) * 64 + 2]); // row 130: padding
}
