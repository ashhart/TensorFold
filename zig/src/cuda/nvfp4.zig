//! NVFP4 checkpoints (e2m1 codes, e4m3 scales a 16 inputs, fp32 global scale) for the lane matmul and the prompt GEMM.
const std = @import("std");
const driver = @import("driver.zig");
const module = @import("module.zig");
const launch_ = @import("launch.zig");
const stream_ = @import("stream.zig");
const kernels = @import("kernels.zig");
const abi = @import("abi.zig");
const qmmf = @import("qmmf.zig");

const Driver = driver.Driver;
const Module = module.Module;
const Function = module.Function;
const Stream = stream_.Stream;

pub const group = qmmf.group;

/// Rows padded as qmm.pack pads them: whole 128-row blocks.
pub fn padded(n: usize) usize {
    return (n + 127) / 128 * 128;
}

/// Codes [n, k/2] (low nibble first) as qmm.pack tiles them: int32 words [npad/64][k/64][8][32][2], rows past n zero.
pub fn packWords(out: []u32, codes: []const u8, n: usize, k: usize) void {
    const npad = padded(n);
    const kg = k / group;
    std.debug.assert(out.len == npad / 64 * kg * 512 and codes.len == n * k / 2 and k % group == 0);
    const offs = [8]usize{ 0, 8, 16, 24, 1, 9, 17, 25 };
    for (out, 0..) |*o, idx| {
        const v = idx & 1;
        const lane = (idx >> 1) & 31;
        const j = (idx >> 6) & 7;
        const tg = idx >> 9;
        const g = tg % kg;
        const row = (tg / kg) * 64 + j * 8 + (lane >> 2);
        var word: u32 = 0;
        if (row < n) for (offs, 0..) |off, p| {
            const input = g * 64 + 32 * v + 2 * (lane & 3) + off;
            const byte = codes[row * (k / 2) + input / 2];
            const nib: u32 = if (input & 1 == 0) byte & 0xF else byte >> 4;
            word |= nib << @intCast(4 * p);
        };
        o.* = word;
    }
}

/// e4m3 scales [n, k/16] as a 64-column tile's scales together [npad/64][k/64][64][4], rows past n zero.
pub fn packScales(out: []u8, scales: []const u8, n: usize, k: usize) void {
    const npad = padded(n);
    const kg = k / group;
    std.debug.assert(out.len == npad * (k / 16) and scales.len == n * (k / 16));
    for (0..npad / 64) |t| for (0..kg) |g| for (0..64) |c| for (0..4) |q| {
        const row = t * 64 + c;
        out[((t * kg + g) * 64 + c) * 4 + q] = if (row < n) scales[row * (k / 16) + g * 4 + q] else 0;
    };
}

/// The lane matmul's view of a packed NVFP4 projection.
pub fn weight(words: u64, scales: u64, global_scale: f32, n: usize, k: usize) qmmf.Weight {
    return .{ .mode = .fp4, .codes = words, .scales = scales, .scale = global_scale, .n = @intCast(n), .k = @intCast(k), .npad = @intCast(padded(n)) };
}

/// prompt16.cu's PROMPT_TILE 4 (FP4, per-tensor FP8; 128 x 128) and prompt_lean, FP4's tile with the same bits.
pub const Prompt = struct {
    mod: Module,
    fp4: Function,
    fp8: Function,
    fp4_f32: Function, // FP4 writing the fp32 sums unrounded
    lean: Function, // prompt_lean FP4, bf16 out
    lean_f32: Function, // and fp32 out
    sms: usize, // the current device's multiprocessors: prompt_lean's blocks (one a multiprocessor, tiles in turn)

    pub const Tile = enum { lean, classic };

    const bm = 128;
    const bn = 128;
    const threads = 128;
    const name = "_ZN11tf_prompt1613prompt_kernelILi{d}ELi128ELi128ELi2ELi2ELi2ELb{d}" ++
        "EEEvPK13__nv_bfloat16PKhS5_fPviiiiii";
    pub const symbols = [3][:0]const u8{
        std.fmt.comptimePrint(name, .{ 0, 0 }),
        std.fmt.comptimePrint(name, .{ 1, 0 }),
        std.fmt.comptimePrint(name, .{ 0, 1 }),
    };
    const lean_bm = 128;
    const lean_bn = 256;
    const lean_threads = 256;
    const lean_stages = 3;
    const lean_name = "_ZN11tf_prompt1611prompt_leanILi0ELi128ELi256ELi1ELi8ELi3ELb{d}" ++
        "EEEvPK13__nv_bfloat16PKhS5_fPviiiiii";
    pub const lean_symbols = [2][:0]const u8{ std.fmt.comptimePrint(lean_name, .{0}), std.fmt.comptimePrint(lean_name, .{1}) };

    /// prompt16.cu LeanTile<FP4, 128, 256, 1, 8, 3>::SMEM: three stages, then a full and an empty barrier each.
    pub fn leanSharedBytes() u32 {
        const stage: u32 = (lean_bm * group * 2 + lean_bn * group / 2 + lean_bn * 4 + 127) / 128 * 128;
        return lean_stages * stage + 2 * lean_stages * 8;
    }

    /// prompt16.cu Tile<MODE, 128, 128, 2, 2, 2>::SMEM.
    pub fn sharedBytes(mode: qmmf.Mode) u32 {
        const w: u32 = if (mode == .fp4) bn * group / 2 else bn * group;
        const s: u32 = if (mode == .fp4) bn * 4 else 0;
        return 2 * ((bm * group * 2 + w + s + 127) / 128 * 128);
    }

    pub fn load(d: *const Driver) !Prompt {
        if (!kernels.available) return error.BuiltWithoutKernels;
        var mod = try Module.load(d, kernels.prompt16);
        errdefer mod.unload();
        const fp4 = try mod.function(symbols[0]);
        try fp4.allowDynamicShared(sharedBytes(.fp4));
        const fp8 = try mod.function(symbols[1]);
        try fp8.allowDynamicShared(sharedBytes(.fp8));
        const fp4_f32 = try mod.function(symbols[2]);
        try fp4_f32.allowDynamicShared(sharedBytes(.fp4));
        const lean = try mod.function(lean_symbols[0]);
        try lean.allowDynamicShared(leanSharedBytes());
        const lean_f32 = try mod.function(lean_symbols[1]);
        try lean_f32.allowDynamicShared(leanSharedBytes());
        var dev: abi.Device = 0;
        try d.check(d.api.cuCtxGetDevice(&dev), "cuCtxGetDevice");
        var sms: c_int = 0;
        try d.check(d.api.cuDeviceGetAttribute(&sms, .multiprocessor_count, dev), "cuDeviceGetAttribute");
        return .{ .mod = mod, .fp4 = fp4, .fp8 = fp8, .fp4_f32 = fp4_f32, .lean = lean, .lean_f32 = lean_f32, .sms = @intCast(@max(sms, 1)) };
    }

    pub fn unload(p: *Prompt) void {
        p.mod.unload();
    }

    /// y [m, n] bf16 = prompt rows x [m, k] bf16 (rows `ldx` apart) times an FP4 or FP8 weight (linear._prompt).
    pub fn matmul(p: Prompt, s: Stream, x: u64, ldx: usize, m: usize, w: qmmf.Weight, y: u64) !void {
        return p.matmulOut(s, x, ldx, m, w, y, .bf16);
    }

    /// matmul() writing y as `out` says: bf16, or fp32 (FP4 only).
    pub fn matmulOut(p: Prompt, s: Stream, x: u64, ldx: usize, m: usize, w: qmmf.Weight, y: u64, out: qmmf.Out) !void {
        return p.matmulTile(s, x, ldx, m, w, y, out, .lean);
    }

    /// matmulOut() on a chosen FP4 tile (FP8 has the classic one only); both write the same bits.
    pub fn matmulTile(p: Prompt, s: Stream, x: u64, ldx: usize, m: usize, w: qmmf.Weight, y: u64, out: qmmf.Out, tile: Tile) !void {
        if (w.mode == .fp4 and tile == .lean) return run(s, x, ldx, m, w, y, if (out == .f32) p.lean_f32 else p.lean, .{
            .bm = lean_bm,
            .bn = lean_bn,
            .threads = lean_threads,
            .shared = leanSharedBytes(),
            .blocks = p.sms,
        });
        const f = switch (w.mode) {
            .fp4 => if (out == .f32) p.fp4_f32 else p.fp4,
            .fp8 => if (out == .f32) return error.EpilogueNotBuilt else p.fp8,
            else => return error.FormatMismatch,
        };
        return run(s, x, ldx, m, w, y, f, .{ .bm = bm, .bn = bn, .threads = threads, .shared = sharedBytes(w.mode) });
    }

    const Shape = struct { bm: usize, bn: usize, threads: u32, shared: u32, blocks: usize = std.math.maxInt(usize) };

    fn run(s: Stream, x: u64, ldx: usize, m: usize, w: qmmf.Weight, y: u64, f: Function, t: Shape) !void {
        const rows_t = (m + t.bm - 1) / t.bm;
        const band = @max(1, @min(rows_t, (12 << 20) / (t.bm * w.k * 2)));
        var args: launch_.Args = .{};
        args.add(x);
        args.add(w.codes);
        args.add(w.scales);
        args.add(w.scale);
        args.add(y);
        args.add(@as(i32, @intCast(m)));
        args.add(@as(i32, @intCast(w.n)));
        args.add(@as(i32, @intCast(w.k)));
        args.add(@as(i32, @intCast(w.npad)));
        args.add(@as(i32, @intCast(if (m == 1) w.k else ldx)));
        args.add(@as(i32, @intCast(band)));
        try launch_.launch(f, .{
            .grid = .{ .x = @intCast(@min(t.blocks, rows_t * ((w.n + t.bn - 1) / t.bn))), .y = 1, .z = 1 },
            .block = .{ .x = t.threads, .y = 1, .z = 1 },
            .shared = t.shared,
        }, s, &args);
    }
};

test "the prompt tile's shared bytes and instance names" {
    try std.testing.expectEqual(@as(u32, 41984), Prompt.sharedBytes(.fp4));
    try std.testing.expectEqual(@as(u32, 76848), Prompt.leanSharedBytes()); // 3 x (16384 + 8192 + 1024) + 48
    try std.testing.expectEqualStrings("_ZN11tf_prompt1611prompt_leanILi0ELi128ELi256ELi1ELi8ELi3ELb0EEEvPK13__nv_bfloat16PKhS5_fPviiiiii", Prompt.lean_symbols[0]);
    try std.testing.expectEqual(@as(u32, 49152), Prompt.sharedBytes(.fp8)); // 2 x (16384 + 8192): no scale tile
    try std.testing.expectEqualStrings("_ZN11tf_prompt1613prompt_kernelILi0ELi128ELi128ELi2ELi2ELi2ELb0EEEvPK13__nv_bfloat16PKhS5_fPviiiiii", Prompt.symbols[0]);
    try std.testing.expectEqualStrings(
        "_ZN11tf_prompt1613prompt_kernelILi0ELi128ELi128ELi2ELi2ELi2ELb1EEEvPK13__nv_bfloat16PKhS5_fPviiiiii",
        Prompt.symbols[2],
    );
}

test "word tiling reads each nibble where qmm.pack puts it" {
    const gpa = std.testing.allocator;
    const n = 130;
    const k = 128;
    const codes = try gpa.alloc(u8, n * k / 2);
    defer gpa.free(codes);
    for (codes, 0..) |*c, i| c.* = @truncate(i * 13 + 5);
    const out = try gpa.alloc(u32, padded(n) / 64 * (k / group) * 512);
    defer gpa.free(out);
    packWords(out, codes, n, k);
    // idx: v 1, lane 6, j 2, g 1, t 0 -> row 2 * 8 + 1 = 17; its nibble p 4 is input 64 + 32 + 2 * 2 + 1 = 101
    const idx = (((0 * 2 + 1) * 8 + 2) * 32 + 6) * 2 + 1;
    const byte = codes[17 * (k / 2) + 101 / 2];
    try std.testing.expectEqual(@as(u32, byte >> 4), (out[idx] >> 16) & 0xF);
    try std.testing.expectEqual(@as(u32, 0), out[((2 * 2 + 0) * 8 + 4) * 64]); // row 128 + 32 = 160: padding
}

test "scale tiles keep a column's four scales of each 64 inputs together" {
    const n = 2;
    const k = 128;
    var s: [n * k / 16]u8 = undefined;
    for (&s, 0..) |*b, i| b.* = @truncate(i + 1);
    var out: [128 * k / 16]u8 = undefined;
    packScales(&out, &s, n, k);
    try std.testing.expectEqual(s[1 * 8 + 4 + 2], out[((0 * 2 + 1) * 64 + 1) * 4 + 2]); // row 1, group 1, quarter 2
    try std.testing.expectEqual(@as(u8, 0), out[((0 * 2 + 0) * 64 + 5) * 4]); // row 5: padding
}
