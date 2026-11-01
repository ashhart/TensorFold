//! The lane matmul (qmmf.cu) for FP4, FP8, MXFP8, FP8G and BF16 weights, with its kernel, tile and slices.
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
pub const Mode = enum(u3) { fp4 = 0, fp8 = 1, mxfp8 = 2, fp8g = 3, bf16 = 4 };

const modes = [_]Mode{ .fp4, .fp8, .mxfp8, .fp8g, .bf16 };

fn modeIndex(mode: Mode) usize {
    return std.mem.indexOfScalar(Mode, &modes, mode).?;
}

/// The dynamic shared bytes of a BM-row tile (qmmf.cu Tile<MODE, BM, 64, 1, 4, 4>::SMEM).
pub fn sharedBytes(mode: Mode, bm: u32) u32 {
    const w: u32 = switch (mode) {
        .fp4 => bn * group / 2,
        .fp8, .mxfp8, .fp8g => bn * group,
        .bf16 => bn * group * 2,
    };
    const s: u32 = if (mode == .fp4 or mode == .fp8g) bn * 4 else if (mode == .mxfp8) bn * 2 else 0;
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

/// A packed projection on the device: codes or words, scale tiles, the tensor scale, its shape, its input scale (act).
pub const Weight = struct { mode: Mode, codes: u64, scales: u64, scale: f32 = 1.0, n: u32, k: u32, npad: u32, act: f32 = 0, sk: u32 = 0 };

/// `w` with its K slices pinned to `sk`: 1, 2, 4 or 8, dividing its groups and leaving at least 8 a slice.
pub fn pinned(w: Weight, sk: u32) error{BadSplit}!Weight {
    const groups = w.k / group;
    if (sk == 0 or sk > 8 or sk & (sk - 1) != 0 or groups % sk != 0 or (sk != 1 and groups / sk < 8)) return error.BadSplit;
    var out = w;
    out.sk = sk;
    return out;
}

/// The base tile and slice meeting from shapes alone; Lane may enlarge a BF16 prompt tile on sm_12x.
pub const Plan = struct { bm: u32, sk: u32, fused: bool, cluster: bool, reduce: bool };

pub fn plan(m: usize, n: usize, k: usize, clusters: bool) Plan {
    return planSlices(m, splitK(n, k), clusters);
}

/// plan() for a packed weight: its pinned K slices if any, else its shape's.
pub fn planFor(m: usize, w: Weight, clusters: bool) Plan {
    return planSlices(m, if (w.sk != 0) w.sk else splitK(w.n, w.k), clusters);
}

fn planSlices(m: usize, sk: u32, clusters: bool) Plan {
    const fused = sk > 1 and m >= fused_rows;
    const cluster = !fused and sk > 1 and sk <= 8 and clusters;
    return .{ .bm = if (fused) 64 else bucket(m), .sk = sk, .fused = fused, .cluster = cluster, .reduce = !fused and sk > 1 and !cluster };
}

/// qmmf.cu's instance names, per mode: row tiles 16, 32, 64 (each plain, then with a cluster), then the fused kernel.
fn symbol(
    comptime mode: Mode,
    comptime bm: u32,
    comptime cluster: bool,
    comptime fused: bool,
    comptime wide: bool,
) [:0]const u8 {
    const name = "_ZN7tf_qmmf11qmmf_kernelILi{d}ELi{d}ELi64ELi1ELi4ELi4ELb{d}ELb{d}ELb{d}" ++
        "EEEvPK13__nv_bfloat16PKhS5_fPvPfiiiiiii";
    const flags = .{ @intFromBool(wide), @intFromBool(cluster), @intFromBool(fused) };
    return std.fmt.comptimePrint(name, .{ @backingInt(mode), bm, flags[0], flags[1], flags[2] });
}

/// The 16-row tile at 32 columns a block (MXFP8 and BF16): plain, then with a cluster.
fn halfSymbol(comptime mode: Mode, comptime cluster: bool, comptime wide: bool) [:0]const u8 {
    const name = "_ZN7tf_qmmf11qmmf_kernelILi{d}ELi16ELi32ELi1ELi4ELi4ELb{d}ELb{d}ELb0EEEvPK13__nv_bfloat16PKhS5_fPvPfiiiiiii";
    return std.fmt.comptimePrint(name, .{ @backingInt(mode), @intFromBool(wide), @intFromBool(cluster) });
}

/// Dynamic shared bytes of the half tile: a stage holds half the weights (and scales) of the 64-column tile's.
pub fn halfShared(mode: Mode) u32 {
    const w: u32 = if (mode == .bf16) 32 * group * 2 else 32 * group;
    const s: u32 = if (mode == .mxfp8) 32 * 2 else 0;
    const stage = (16 * group * 2 + w + s + 127) / 128 * 128;
    const partials = 1 * 1 * 4 * threads * 4;
    return @max(4 * stage, partials);
}

/// MXFP8/BF16 decode calls take half tiles below this many 64-column blocks (TF_QMMF_HALF_BELOW; 0 never): same sums.
pub const half_below_default: usize = 96;

fn halfBelow() usize {
    const v = std.c.getenv("TF_QMMF_HALF_BELOW") orelse return half_below_default;
    return std.fmt.parseInt(usize, std.mem.span(v), 10) catch half_below_default;
}

pub fn halfModeIndex(mode: Mode) ?usize {
    return switch (mode) {
        .mxfp8 => 0,
        .bf16 => 1,
        else => null,
    };
}

pub const reduce_symbols = [2][:0]const u8{
    "_ZN7tf_qmmf13reduce_kernelILb0EEEvPKfPvxif",
    "_ZN7tf_qmmf13reduce_kernelILb1EEEvPKfPvxif",
};
const row_tiles = [_]u32{ 16, 32, 64 };

/// What a call writes: bf16 rows, or the fp32 sums unrounded (a tensor-parallel rank's partial rows).
pub const Out = enum { bf16, f32 };

fn loadTiles(mod: Module, comptime mode: Mode, comptime wide: bool) ![7]Function {
    var set: [7]Function = undefined;
    inline for (row_tiles, 0..) |bm, ti| inline for (.{ false, true }, 0..) |cl, ci| {
        const f = try mod.function(symbol(mode, bm, cl, false, wide));
        try f.allowDynamicShared(sharedBytes(mode, bm));
        set[ti * 2 + ci] = f;
    };
    set[6] = try mod.function(symbol(mode, 64, false, true, wide));
    try set[6].allowDynamicShared(sharedBytes(mode, 64));
    return set;
}

/// BF16 prompt tile: 128x128, eight warps, three 32 KiB stages and one barrier per stage.
pub const prompt_shared: u32 = 3 * (128 + 128) * group * 2 + 128;
pub const prompt_rows: usize = 512;
pub const prompt_k32_shared: u32 = 3 * (128 + 128) * 32 * 2;

pub fn promptK32Symbol(comptime out: Out) [:0]const u8 {
    return std.fmt.comptimePrint("_ZN7tf_qmmf22qmmf_prompt_k32_kernelILb{d}EEEvPK13__nv_bfloat16PKhS5_fPvPfiiiiiii", .{@intFromBool(out == .f32)});
}

pub fn promptSymbol(comptime out: Out, comptime fused: bool) [:0]const u8 {
    return std.fmt.comptimePrint("_ZN7tf_qmmf11qmmf_kernelILi4ELi128ELi128ELi2ELi4ELi3ELb{d}ELb0ELb{d}EEEvPK13__nv_bfloat16PKhS5_fPvPfiiiiiii", .{ @intFromBool(out == .f32), @intFromBool(fused) });
}

/// Only large BF16 grids use the prompt tile; the packed allocation contains complete 128-column tiles.
pub fn promptEligible(m: usize, w: Weight) bool {
    return m >= prompt_rows and w.mode == .bf16 and w.n >= 128 and
        (w.k >= 512 or (w.n == 10240 and w.k == 320)) and w.npad % 128 == 0;
}

pub const Lane = struct {
    mod: Module,
    fns: [modes.len][7]Function, // per mode: lane16, lane16c, lane32, lane32c, lane64, lane64c, fused
    f32_fns: [modes.len][7]Function,
    reduce: [2]Function, // by Out
    half: [2][2][2]Function, // by halfModeIndex, Out, then plain / cluster
    half_below: usize = half_below_default,
    shared_down: [4]Function = undefined,
    prompt: ?[2][2]Function = null, // Out, then plain / fused; loaded only on sm_12x
    prompt_k32: ?[2]Function = null, // Out; unsplit BF16 prompt calls on sm_12x
    clusters: bool, // sm_90 on: slices meet in a cluster's shared memory

    pub fn load(d: *const Driver, capability_major: u32) !Lane {
        if (!kernels.available) return error.BuiltWithoutKernels;
        var mod = try Module.load(d, kernels.qmmf);
        errdefer mod.unload();
        var l: Lane = .{
            .mod = mod,
            .fns = undefined,
            .f32_fns = undefined,
            .reduce = undefined,
            .half = undefined,
            .clusters = capability_major >= 9,
        };
        inline for (modes, 0..) |mode, mi| {
            l.fns[mi] = try loadTiles(mod, mode, false);
            l.f32_fns[mi] = try loadTiles(mod, mode, true);
        }
        for (&l.reduce, reduce_symbols) |*f, name| f.* = try mod.function(name);
        inline for (.{ Mode.mxfp8, Mode.bf16 }, 0..) |mode, mi| inline for (.{ false, true }, 0..) |wide, oi| inline for (.{ false, true }, 0..) |cl, ci| {
            const f = try mod.function(halfSymbol(mode, cl, wide));
            try f.allowDynamicShared(halfShared(mode));
            l.half[mi][oi][ci] = f;
        };
        for (@import("qmmf_shared.zig").symbols, &l.shared_down) |name, *f| {
            f.* = try mod.function(name);
            try f.allowDynamicShared(sharedBytes(.bf16, 64));
        }
        if (capability_major == 12) {
            var prompt: [2][2]Function = undefined;
            inline for (.{ Out.bf16, Out.f32 }, 0..) |out, oi| inline for (.{ false, true }, 0..) |fused, fi| {
                const f = try mod.function(promptSymbol(out, fused));
                try f.allowDynamicShared(prompt_shared);
                prompt[oi][fi] = f;
            };
            l.prompt = prompt;
            var k32: [2]Function = undefined;
            inline for (.{ Out.bf16, Out.f32 }, 0..) |out, oi| {
                const f = try mod.function(promptK32Symbol(out));
                try f.allowDynamicShared(prompt_k32_shared);
                k32[oi] = f;
            }
            l.prompt_k32 = k32;
        }
        l.half_below = halfBelow();
        return l;
    }

    pub fn unload(l: *Lane) void {
        l.mod.unload();
    }

    /// Slice partials a call needs (fp32 [sk][m][n]) when it reduces them, else 0.
    pub fn partBytes(l: Lane, m: usize, w: Weight) usize {
        const p = planFor(m, w, l.clusters);
        return if (p.reduce) @as(usize, p.sk) * m * w.n * 4 else 0;
    }

    /// y [m, n] bf16 = x [m, k] bf16 (rows `ldx` apart) times the weight, fp32 sums; `part` holds partBytes.
    pub fn matmul(l: Lane, s: Stream, x: u64, ldx: usize, m: usize, w: Weight, y: u64, part: u64) !void {
        return l.matmulOut(s, x, ldx, m, w, y, .bf16, part);
    }

    /// Preserve FP32 outputs for a later residual or tensor-rank sum.
    pub fn matmulF32(l: Lane, s: Stream, x: u64, ldx: usize, m: usize, w: Weight, y: u64, part: u64) !void {
        return l.matmulOut(s, x, ldx, m, w, y, .f32, part);
    }

    pub const matmulF32Strided = @import("qmmf_shared.zig").matmulF32Strided;

    /// The actual prompt selection, including loaded instances and the pinned slice meeting plan.
    pub fn usesPrompt(l: Lane, m: usize, w: Weight) bool {
        const p = planFor(m, w, l.clusters);
        return promptEligible(m, w) and !p.cluster and !p.reduce and l.prompt != null;
    }

    /// matmul() writing y as `out` says: bf16, or fp32.
    pub fn matmulOut(l: Lane, s: Stream, x: u64, ldx: usize, m: usize, w: Weight, y: u64, out: Out, part: u64) !void {
        const p = planFor(m, w, l.clusters);
        if (p.reduce and part == 0) return error.MissingLanePart;
        const set = (if (out == .f32) l.f32_fns else l.fns)[modeIndex(w.mode)];
        const ti: usize = if (p.bm == 16) 0 else if (p.bm == 32) 1 else 2;
        const prompt = l.usesPrompt(m, w);
        const k32 = prompt and p.sk == 1 and l.prompt_k32 != null;
        const bm: usize = if (prompt) 128 else p.bm;
        const rows_t = (m + bm - 1) / bm;
        const hm = halfModeIndex(w.mode);
        const half = hm != null and p.bm == 16 and !p.fused and rows_t * ((w.n + bn - 1) / bn) * p.sk < l.half_below;
        const f = if (k32) l.prompt_k32.?[@backingInt(out)] else if (prompt) l.prompt.?[@backingInt(out)][@intFromBool(p.fused)] else if (half) l.half[hm.?][@backingInt(out)][@intFromBool(p.cluster)] else if (p.fused) set[6] else set[ti * 2 + @intFromBool(p.cluster)];
        const cols: usize = if (prompt) 128 else if (half) 32 else bn;
        const band = @max(1, @min(rows_t, (12 << 20) / (bm * w.k * 2)));
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
            .grid = .{ .x = @intCast(rows_t * ((w.n + cols - 1) / cols)), .y = 1, .z = if (p.fused) 1 else p.sk },
            .block = .{ .x = if (prompt) 256 else threads, .y = 1, .z = 1 },
            .shared = if (k32) prompt_k32_shared else if (prompt) prompt_shared else if (half) halfShared(w.mode) else sharedBytes(w.mode, p.bm),
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
            try launch_.launch(l.reduce[@backingInt(out)], .{
                .grid = .{ .x = @intCast((total + 255) / 256), .y = 1, .z = 1 },
                .block = .{ .x = 256, .y = 1, .z = 1 },
            }, s, &r);
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
    try std.testing.expectEqual(@as(u32, 25088), sharedBytes(.mxfp8, 16));
    try std.testing.expectEqual(@as(u32, 33280), sharedBytes(.mxfp8, 32));
    try std.testing.expectEqual(@as(u32, 49664), sharedBytes(.mxfp8, 64));
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
    try std.testing.expectEqualStrings("_ZN7tf_qmmf11qmmf_kernelILi4ELi64ELi64ELi1ELi4ELi4ELb1ELb0ELb1EEEvPK13__nv_bfloat16PKhS5_fPvPfiiiiiii", symbol(.bf16, 64, false, true, true));
    try std.testing.expectEqualStrings("_ZN7tf_qmmf11qmmf_kernelILi2ELi16ELi64ELi1ELi4ELi4ELb0ELb1ELb0EEEvPK13__nv_bfloat16PKhS5_fPvPfiiiiiii", symbol(.mxfp8, 16, true, false, false));
    try std.testing.expectEqualStrings("_ZN7tf_qmmf11qmmf_kernelILi3ELi16ELi64ELi1ELi4ELi4ELb0ELb1ELb0EEEvPK13__nv_bfloat16PKhS5_fPvPfiiiiiii", symbol(.fp8g, 16, true, false, false));
    try std.testing.expectEqualStrings("_ZN7tf_qmmf11qmmf_kernelILi0ELi64ELi64ELi1ELi4ELi4ELb0ELb0ELb1EEEvPK13__nv_bfloat16PKhS5_fPvPfiiiiiii", symbol(.fp4, 64, false, true, false));
    try std.testing.expectEqualStrings("_ZN7tf_qmmf11qmmf_kernelILi0ELi32ELi64ELi1ELi4ELi4ELb1ELb1ELb0EEEvPK13__nv_bfloat16PKhS5_fPvPfiiiiiii", symbol(.fp4, 32, true, false, true));
}

test "large BF16 tiles preserve pinned slices and leave decode and other formats unchanged" {
    const w = try pinned(Weight{ .mode = .bf16, .codes = 0, .scales = 0, .n = 8240, .k = 2560, .npad = 8320 }, 1);
    for ([_]usize{ 1, 16, 64, 65, 511, 512, 1536, 4218 }) |m| {
        try std.testing.expectEqual(m >= prompt_rows, promptEligible(m, w));
        try std.testing.expectEqual(@as(u32, 1), planFor(m, w, true).sk);
    }
    var other = w;
    other.mode = .fp8;
    try std.testing.expect(!promptEligible(4218, other));
    try std.testing.expectEqual(@as(u32, 98432), prompt_shared);
    const hc_up: Weight = .{ .mode = .bf16, .codes = 0, .scales = 0, .n = 10240, .k = 320, .npad = 10240 };
    try std.testing.expect(promptEligible(512, hc_up));
    try std.testing.expect(!promptEligible(511, hc_up));
    var narrow_up = hc_up;
    narrow_up.n = 5120;
    narrow_up.npad = 5120;
    try std.testing.expect(!promptEligible(512, narrow_up));
    const out = try pinned(Weight{ .mode = .bf16, .codes = 0, .scales = 0, .n = 2560, .k = 3072, .npad = 2560 }, 4);
    const p = planFor(4218, out, true);
    try std.testing.expect(promptEligible(4218, out) and p.fused and p.sk == 4 and !p.reduce and !p.cluster);
}

test "prompt dispatch changes CTA geometry and output instance without changing slices" {
    const abi = @import("abi.zig");
    const Trace = struct {
        var calls: usize = 0;
        var grid: u32 = 0;
        var block: u32 = 0;
        var shared: u32 = 0;
        var id: usize = 0;
        var slices: i32 = 0;
        fn run(f: abi.Function, gx: c_uint, _: c_uint, gz: c_uint, bx: c_uint, _: c_uint, _: c_uint, sm: c_uint, _: abi.Stream, args: ?[*]?*anyopaque, _: ?[*]?*anyopaque) callconv(.c) abi.Result {
            calls += 1;
            grid = gx;
            block = bx;
            shared = sm;
            id = @intFromPtr(f);
            slices = @as(*const i32, @ptrCast(@alignCast(args.?[9].?))).*;
            return if (gz == 1) abi.success else 1;
        }
    };
    var d: Driver = .{ .lib = undefined, .api = undefined };
    d.api.cuLaunchKernel = Trace.run;
    const base: Function = .{ .d = &d, .handle = @ptrFromInt(10) };
    const fast: Function = .{ .d = &d, .handle = @ptrFromInt(20) };
    const fused: Function = .{ .d = &d, .handle = @ptrFromInt(21) };
    const wide: Function = .{ .d = &d, .handle = @ptrFromInt(23) };
    const k32: Function = .{ .d = &d, .handle = @ptrFromInt(30) };
    const k32_wide: Function = .{ .d = &d, .handle = @ptrFromInt(31) };
    var lane: Lane = .{
        .mod = undefined,
        .fns = @splat(@splat(base)),
        .f32_fns = @splat(@splat(base)),
        .reduce = @splat(base),
        .half = @splat(@splat(@splat(base))),
        .half_below = 0,
        .prompt = .{ .{ fast, fused }, .{ fast, wide } },
        .prompt_k32 = .{ k32, k32_wide },
        .clusters = false,
    };
    const s: Stream = .{ .d = &d, .handle = null };
    var w = try pinned(Weight{ .mode = .bf16, .codes = 0, .scales = 0, .n = 8240, .k = 2560, .npad = 8320 }, 1);
    for ([_]usize{ 1, 65, 511, 512, 4218 }) |m| {
        Trace.calls = 0;
        try lane.matmul(s, 0, w.k, m, w, 0, 0);
        const fast_path = m >= prompt_rows;
        const bm: usize = if (fast_path) 128 else bucket(m);
        const cols: usize = if (fast_path) 128 else 64;
        try std.testing.expectEqual(@as(usize, 1), Trace.calls);
        try std.testing.expectEqual(@as(usize, if (fast_path) 30 else 10), Trace.id);
        try std.testing.expectEqual(@as(u32, if (fast_path) 256 else 128), Trace.block);
        try std.testing.expectEqual(if (fast_path) prompt_k32_shared else sharedBytes(.bf16, @intCast(bm)), Trace.shared);
        try std.testing.expectEqual((m + bm - 1) / bm * ((w.n + cols - 1) / cols), Trace.grid);
        try std.testing.expectEqual(@as(i32, 1), Trace.slices);
    }
    w = try pinned(Weight{ .mode = .bf16, .codes = 0, .scales = 0, .n = 2560, .k = 3072, .npad = 2560 }, 4);
    try lane.matmulF32(s, 0, w.k, 4218, w, 0, 0);
    try std.testing.expectEqual(@as(usize, 23), Trace.id);
    try std.testing.expectEqual(@as(i32, 4), Trace.slices);
    try std.testing.expectEqual(prompt_shared, Trace.shared);
    for ([_]u32{ 2, 4, 8 }) |sk| {
        w = try pinned(Weight{ .mode = .bf16, .codes = 0, .scales = 0, .n = 2560, .k = 6144, .npad = 2560 }, sk);
        for ([_]bool{ false, true }) |clusters| for ([_]Out{ .bf16, .f32 }) |out| for ([_]usize{ 512, 1536, 4218 }) |m| {
            lane.clusters = clusters;
            Trace.calls = 0;
            try lane.matmulOut(s, 0, w.k, m, w, 0, out, 0);
            try std.testing.expectEqual(@as(usize, 1), Trace.calls);
            try std.testing.expectEqual(@as(usize, if (out == .f32) 23 else 21), Trace.id);
            try std.testing.expectEqual(prompt_shared, Trace.shared);
            try std.testing.expectEqual((m + 127) / 128 * 20, Trace.grid);
            try std.testing.expectEqual(@as(i32, @intCast(sk)), Trace.slices);
        };
    }
    w = try pinned(w, 1);
    try lane.matmulF32(s, 0, w.k, 4218, w, 0, 0);
    try std.testing.expectEqual(@as(usize, 31), Trace.id);
    try std.testing.expectEqual(prompt_k32_shared, Trace.shared);
    lane.prompt_k32 = null;
    try lane.matmulF32(s, 0, w.k, 4218, w, 0, 0);
    try std.testing.expectEqual(@as(usize, 20), Trace.id);
    try std.testing.expectEqual(prompt_shared, Trace.shared);
    w = try pinned(w, 4);
    lane.prompt = null;
    try lane.matmulF32(s, 0, w.k, 4218, w, 0, 0);
    try std.testing.expectEqual(@as(usize, 10), Trace.id);
    try std.testing.expectEqual(@as(i32, 4), Trace.slices);
}

test "a cut weight pinned to the whole weight's K slices plans as the whole one at every row count" {
    const whole: Weight = .{ .mode = .bf16, .codes = 0, .scales = 0, .n = 16480, .k = 2560, .npad = 16512 };
    var cut = whole;
    cut.n = 8240;
    cut.npad = 8320;
    try std.testing.expectEqual(@as(u32, 1), splitK(whole.n, whole.k));
    try std.testing.expectEqual(@as(u32, 2), splitK(cut.n, cut.k)); // unpinned, the half would sum two slices
    const kept = try pinned(cut, splitK(whole.n, whole.k));
    for ([_]usize{ 1, 16, 64, 1536 }) |m| for ([_]bool{ false, true }) |cl| {
        try std.testing.expectEqual(plan(m, whole.n, whole.k, cl), planFor(m, kept, cl));
        try std.testing.expectEqual(plan(m, cut.n, cut.k, cl), planFor(m, cut, cl));
    };
    try std.testing.expect(planFor(1536, cut, false).sk != planFor(1536, kept, false).sk);
    try std.testing.expectEqual(@as(u32, 4), (try pinned(cut, 4)).sk); // 40 groups: 10 a slice
    for ([_]u32{ 0, 3, 8, 16 }) |bad| try std.testing.expectError(error.BadSplit, pinned(cut, bad)); // 8: 5 a slice
}
