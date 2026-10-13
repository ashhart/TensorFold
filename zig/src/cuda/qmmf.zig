//! The lane matmul (qmmf.cu) for FP4, FP8, FP8G and BF16 weights, with its kernel, tile and slices.
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

pub const group = 64; // inputs a kernel stage reads
pub const fused_rows = 256; // rows from which a projection's K slices meet in one block (linear.py FUSED_ROWS)
const bn = 64;
const threads = 128; // WM 1 x WN 4 warps

/// The packed formats (qmmf.cu Mode): FP4 with e4m3 scales a 16, e4m3 with one scale or fp32 a 64, and bf16.
pub const Mode = enum(u3) { fp4 = 0, fp8 = 1, fp8g = 3, bf16 = 4 };

const modes = [_]Mode{ .fp4, .fp8, .fp8g, .bf16 };

fn modeIndex(mode: Mode) usize {
    return std.mem.indexOfScalar(Mode, &modes, mode).?;
}

/// The dynamic shared bytes of a BM-row tile (qmmf.cu Tile<MODE, BM, 64, 1, 4, 4>::SMEM).
pub fn sharedBytes(mode: Mode, bm: u32) u32 {
    const w: u32 = switch (mode) {
        .fp4 => bn * group / 2,
        .fp8, .fp8g => bn * group,
        .bf16 => bn * group * 2,
    };
    const s: u32 = if (mode == .fp4 or mode == .fp8g) bn * 4 else 0;
    const stage = (bm * group * 2 + w + s + 127) / 128 * 128;
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

/// A packed projection on the device: codes or words, scale tiles, the tensor scale, its shape.
pub const Weight = struct { mode: Mode, codes: u64, scales: u64, scale: f32 = 1.0, n: u32, k: u32, npad: u32 };

/// What one call launches, from the shapes alone: tile, slices, cluster or reduce (qmmf_cuda's choice).
pub const Plan = struct { bm: u32, sk: u32, fused: bool, cluster: bool, reduce: bool };

pub fn plan(m: usize, n: usize, k: usize, clusters: bool) Plan {
    const sk = splitK(n, k);
    const fused = sk > 1 and m >= fused_rows;
    const cluster = !fused and sk > 1 and sk <= 8 and clusters;
    return .{ .bm = if (fused) 64 else bucket(m), .sk = sk, .fused = fused, .cluster = cluster, .reduce = !fused and sk > 1 and !cluster };
}

/// qmmf.cu's instance names, per mode: row tiles 16, 32, 64 (each plain, then with a cluster), then the fused kernel.
fn symbol(comptime mode: Mode, comptime bm: u32, comptime cluster: bool, comptime fused: bool) [:0]const u8 {
    const name = "_ZN7tf_qmmf11qmmf_kernelILi{d}ELi{d}ELi64ELi1ELi4ELi4ELb0ELb{d}ELb{d}EEEvPK13__nv_bfloat16PKhS5_fPvPfiiiiiii";
    return std.fmt.comptimePrint(name, .{ @backingInt(mode), bm, @intFromBool(cluster), @intFromBool(fused) });
}

pub const reduce_symbol = "_ZN7tf_qmmf13reduce_kernelILb0EEEvPKfPvxif";
const row_tiles = [_]u32{ 16, 32, 64 };

pub const Lane = struct {
    mod: Module,
    fns: [modes.len][7]Function, // per mode: lane16, lane16c, lane32, lane32c, lane64, lane64c, fused
    reduce: Function,
    clusters: bool, // sm_90 on: slices meet in a cluster's shared memory

    pub fn load(d: *const Driver, capability_major: u32) !Lane {
        if (!kernels.available) return error.BuiltWithoutKernels;
        var mod = try Module.load(d, kernels.qmmf);
        errdefer mod.unload();
        var l: Lane = .{ .mod = mod, .fns = undefined, .reduce = try mod.function(reduce_symbol), .clusters = capability_major >= 9 };
        inline for (modes, 0..) |mode, mi| {
            inline for (row_tiles, 0..) |bm, ti| inline for (.{ false, true }, 0..) |cl, ci| {
                const f = try mod.function(symbol(mode, bm, cl, false));
                try f.allowDynamicShared(sharedBytes(mode, bm));
                l.fns[mi][ti * 2 + ci] = f;
            };
            const f = try mod.function(symbol(mode, 64, false, true));
            try f.allowDynamicShared(sharedBytes(mode, 64));
            l.fns[mi][6] = f;
        }
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

    /// y [m, n] bf16 = x [m, k] bf16 (rows `ldx` apart) times the weight, fp32 sums; `part` holds partBytes.
    pub fn matmul(l: Lane, s: Stream, x: u64, ldx: usize, m: usize, w: Weight, y: u64, part: u64) !void {
        const p = plan(m, w.n, w.k, l.clusters);
        const set = l.fns[modeIndex(w.mode)];
        const ti: usize = if (p.bm == 16) 0 else if (p.bm == 32) 1 else 2;
        const f = if (p.fused) set[6] else set[ti * 2 + @intFromBool(p.cluster)];
        const rows_t = (m + p.bm - 1) / p.bm;
        const band = @max(1, @min(rows_t, (12 << 20) / (@as(usize, p.bm) * w.k * 2)));
        var args: launch_.Args = .{};
        args.add(x);
        args.add(w.codes);
        args.add(w.scales);
        args.add(w.scale);
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
            .shared = sharedBytes(w.mode, p.bm),
            .cluster = if (p.cluster) .{ .x = 1, .y = 1, .z = p.sk } else null,
        }, s, &args);
        if (p.reduce) {
            const total: usize = m * w.n;
            var r: launch_.Args = .{};
            r.add(part);
            r.add(y);
            r.add(@as(i64, @intCast(total)));
            r.add(@as(i32, @intCast(p.sk)));
            r.add(w.scale);
            try launch_.launch(l.reduce, .{ .grid = .{ .x = @intCast((total + 255) / 256), .y = 1, .z = 1 }, .block = .{ .x = 256, .y = 1, .z = 1 } }, s, &r);
        }
    }
};

/// bf16 [n, k] in the BF16 mode's order [npad/64][k/64][8][4][32][4]: n8 tile, k16 step, lane, b0 then b1 halves.
pub fn packBf16(out: []u16, w: []const u16, n: usize, k: usize) void {
    const npad = (n + 127) / 128 * 128;
    std.debug.assert(out.len == npad * k and w.len == n * k and k % group == 0);
    for (out, 0..) |*o, i| {
        const e = i % 4;
        const lane = (i / 4) % 32;
        const kt = (i / 128) % 4;
        const jj = (i / 512) % 8;
        const tg = i / 4096;
        const row = (tg / (k / group)) * 64 + jj * 8 + lane / 4;
        const col = (tg % (k / group)) * group + kt * 16 + 2 * (lane % 4) + (e & 1) + 8 * (e >> 1);
        o.* = if (row < n) w[row * k + col] else 0;
    }
}

test "bf16 packing puts each lane's b0 and b1 pairs where the m16n8k16 B fragment reads them" {
    const gpa = std.testing.allocator;
    const n = 70;
    const k = 128;
    const w = try gpa.alloc(u16, n * k);
    defer gpa.free(w);
    for (w, 0..) |*x, i| x.* = @truncate(i * 7 + 3);
    const out = try gpa.alloc(u16, 128 * k);
    defer gpa.free(out);
    packBf16(out, w, n, k);
    // tile 0, group 1, n8 tile 5, k16 step 2, lane 13: row 40 + 3 = 43, k0 = 64 + 32 + 2 * 1 = 98
    const at = (((1 * 8 + 5) * 4 + 2) * 32 + 13) * 4;
    try std.testing.expectEqualSlices(u16, &.{ w[43 * k + 98], w[43 * k + 99], w[43 * k + 106], w[43 * k + 107] }, out[at..][0..4]);
    try std.testing.expectEqual(w[63 * k + 6], out[(((0 * 8 + 7) * 4 + 0) * 32 + 31) * 4]); // row 56 + 7 = 63
    try std.testing.expectEqual(@as(u16, 0), out[(((2 * 8 + 1) * 4 + 0) * 32 + 31) * 4]); // row 64 + 8 + 7: padding
}

test "tile shared bytes, slices and row tiles as the Python host picks them" {
    try std.testing.expectEqual(@as(u32, 25600), sharedBytes(.fp8g, 16));
    try std.testing.expectEqual(@as(u32, 33792), sharedBytes(.fp8g, 32));
    try std.testing.expectEqual(@as(u32, 50176), sharedBytes(.fp8g, 64));
    try std.testing.expectEqual(@as(u32, 17408), sharedBytes(.fp4, 16));
    try std.testing.expectEqual(@as(u32, 25600), sharedBytes(.fp4, 32));
    try std.testing.expectEqual(@as(u32, 41984), sharedBytes(.fp4, 64));
    try std.testing.expectEqual(@as(u32, 24576), sharedBytes(.fp8, 16)); // no scale tile: 4 x (2048 + 4096)
    try std.testing.expectEqual(@as(u32, 40960), sharedBytes(.bf16, 16)); // 4 x (2048 + 8192)
    try std.testing.expectEqual(@as(u32, 65536), sharedBytes(.bf16, 64));
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

test "instance names are qmmf.cu's" {
    try std.testing.expectEqualStrings("_ZN7tf_qmmf11qmmf_kernelILi3ELi16ELi64ELi1ELi4ELi4ELb0ELb1ELb0EEEvPK13__nv_bfloat16PKhS5_fPvPfiiiiiii", symbol(.fp8g, 16, true, false));
    try std.testing.expectEqualStrings("_ZN7tf_qmmf11qmmf_kernelILi0ELi64ELi64ELi1ELi4ELi4ELb0ELb0ELb1EEEvPK13__nv_bfloat16PKhS5_fPvPfiiiiiii", symbol(.fp4, 64, false, true));
}
