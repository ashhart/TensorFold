//! exl3.cu's EXL3 projections: bf16 rows rotated into fp32, then 1-16 rows a launch over the trellis tiles as stored.
const std = @import("std");
const driver = @import("driver.zig");
const module = @import("module.zig");
const launch_ = @import("launch.zig");
const memory = @import("memory.zig");
const stream_ = @import("stream.zig");

const Driver = driver.Driver;
const Function = module.Function;
const Stream = stream_.Stream;

/// core/exl3_format.zig's Codebook, in the order exl3.cu numbers them.
pub const Codebook = enum(u32) { inst3 = 0, mcg = 1, mul1 = 2 };

/// One matrix on the device as stored: trellis int16 [k/16, n/16, 16 bits], suh fp16 [k], svh fp16 [n].
pub const Matrix = struct {
    trellis: u64,
    suh: u64,
    svh: u64,
    k: usize,
    n: usize,
    bits: u32,
    codebook: Codebook,

    /// Whole widths 1-8 over whole 128-blocks; exl3.cu takes no half widths and no partial blocks.
    pub fn check(m: Matrix) !void {
        if (m.k == 0 or m.n == 0 or m.k % 128 != 0 or m.n % 128 != 0 or m.bits < 1 or m.bits > 8) return error.InvalidExl3Matrix;
    }

    pub fn trellisBytes(k: usize, n: usize, bits: u32) usize {
        return k / 16 * (n / 16) * 32 * @as(usize, bits);
    }

    fn args(m: Matrix) MatrixArgs {
        return .{ .k = @intCast(m.k), .n = @intCast(m.n), .bits = m.bits, .tiles_n = @intCast(m.n / 16), .split = @intCast(split(m.k, m.n)) };
    }
};

/// exl3.cu's TfExl3Rows and TfExl3Matrix, passed by value.
pub const RowsArgs = extern struct { rows: u32, x_stride: u32, y_stride: u32, y_f32: u32 };
pub const MatrixArgs = extern struct { k: u32, n: u32, bits: u32, tiles_n: u32, split: u32 };

/// Decode rows a launch; longer decode runs go window by window.
pub const window = 16;
/// Prompt rows a chunk: rotated to fp16 together, then 64 rows a block on the tensor units.
pub const prompt_chunk = 256;
/// Blocks a launch aims for: a matrix with fewer 128-column blocks splits its input blocks into parts.
const target_blocks = 96;

/// Parts of the input blocks, from the shape alone: enough for ~target_blocks blocks, at least 4 input blocks a part.
pub fn split(k: usize, n: usize) usize {
    const cols = n / 128;
    const kb = k / 128;
    if (cols == 0 or cols >= target_blocks) return 1;
    const want = @min((target_blocks + cols - 1) / cols, kb / 4);
    if (want <= 1) return 1;
    const per = (kb + want - 1) / want;
    return (kb + per - 1) / per;
}

/// The kernels and their scratch (fp16 rotated rows up to `k_max` inputs): one stream's projections at a time.
pub const Kernels = struct {
    decode_fns: [3]Function, // 1-16 rows on the tensor units, by codebook
    prompt_fns: [3]Function, // 64 rows a block on the tensor units, by codebook
    rotate_f_fn: Function, // decode rows, each 128-block in fragment order
    rotate_h_fn: Function, // prompt rows, plain order
    codebook_fn: Function,
    unpack_fn: Function,
    rotated: memory.DeviceBuffer, // a decode window's or prompt chunk's rotated rows (fp16, prompt_chunk x k_max)
    parts: memory.DeviceBuffer, // a split matrix's part totals: 16 rows x 128 columns per (column block, part)
    tickets: memory.DeviceBuffer, // per column block, parts finished; the last part resets its count to zero
    k_max: usize,

    pub fn resolve(d: *const Driver, mod: module.Module, k_max: usize) !Kernels {
        var k: Kernels = .{ .decode_fns = undefined, .prompt_fns = undefined, .rotate_f_fn = try mod.function("tf_exl3_rotate_f"), .rotate_h_fn = try mod.function("tf_exl3_rotate_h"), .codebook_fn = try mod.function("tf_exl3_codebook"), .unpack_fn = try mod.function("tf_exl3_unpack"), .rotated = undefined, .parts = undefined, .tickets = undefined, .k_max = k_max };
        for (&k.decode_fns, &k.prompt_fns, 0..) |*f, *p, c| {
            var name: [32]u8 = undefined;
            f.* = try mod.function(try std.fmt.bufPrintSentinel(&name, "tf_exl3_tc_{d}", .{c}, 0));
            p.* = try mod.function(try std.fmt.bufPrintSentinel(&name, "tf_exl3_prompt_{d}", .{c}, 0));
        }
        k.rotated = try memory.DeviceBuffer.alloc(d, prompt_chunk * k_max * 2);
        errdefer k.rotated.free();
        // split * cols stays under target_blocks + cols, so 2 * target_blocks parts always fit
        k.parts = try memory.DeviceBuffer.alloc(d, 2 * target_blocks * window * 128 * 4);
        errdefer k.parts.free();
        k.tickets = try memory.DeviceBuffer.alloc(d, target_blocks * 4);
        errdefer k.tickets.free();
        try d.check(d.api.cuMemsetD32_v2(k.tickets.ptr, 0, target_blocks), "cuMemsetD32");
        return k;
    }

    pub fn deinit(k: *Kernels) void {
        k.rotated.free();
        k.parts.free();
        k.tickets.free();
    }

    /// Decode rows: bf16 rows `ldx` apart -> out (rows, n), 16 a window on the tensor units, the same at every count.
    pub fn rows(k: *const Kernels, s: Stream, x: u64, ldx: usize, m: Matrix, out: u64, count: usize, f32_out: bool) !void {
        try k.fits(m, ldx);
        const elem: usize = if (f32_out) 4 else 2;
        var at: usize = 0;
        while (at < count) : (at += window) {
            const r = @min(window, count - at);
            try k.rotate(k.rotate_f_fn, s, x + at * ldx * 2, ldx, m, r);
            var a: launch_.Args = .{};
            a.add(k.rotated.ptr);
            a.add(m.trellis);
            a.add(m.svh);
            a.add(RowsArgs{ .rows = @intCast(r), .x_stride = @intCast(m.k), .y_stride = @intCast(m.n), .y_f32 = @intFromBool(f32_out) });
            a.add(m.args());
            a.add(out + at * m.n * elem);
            a.add(k.parts.ptr);
            a.add(k.tickets.ptr);
            try launch_.launch(k.decode_fns[@backingInt(m.codebook)], .{ .grid = .{ .x = @intCast(m.n / 128), .y = @intCast(split(m.k, m.n)), .z = 1 }, .block = .{ .x = 256 } }, s, &a);
        }
    }

    /// Prompt rows (docs/recipes/exl3.md, Prompts): fp16 rows on the tensor units, fp32 sums, bf16 or fp32 out.
    pub fn prompt(k: *const Kernels, s: Stream, x: u64, ldx: usize, m: Matrix, out: u64, count: usize, f32_out: bool) !void {
        const elem: usize = if (f32_out) 4 else 2;
        try k.fits(m, ldx);
        var at: usize = 0;
        while (at < count) : (at += prompt_chunk) {
            const r = @min(prompt_chunk, count - at);
            try k.rotate(k.rotate_h_fn, s, x + at * ldx * 2, ldx, m, r);
            var a: launch_.Args = .{};
            a.add(k.rotated.ptr);
            a.add(@as(u32, @intCast(r)));
            a.add(m.trellis);
            a.add(m.svh);
            a.add(m.args());
            a.add(out + at * m.n * elem);
            a.add(@as(u32, @intFromBool(f32_out)));
            try launch_.launch(k.prompt_fns[@backingInt(m.codebook)], .{ .grid = .{ .x = @intCast(m.n / 128), .y = @intCast((r + 63) / 64), .z = 1 }, .block = .{ .x = 256 } }, s, &a);
        }
    }

    fn fits(k: *const Kernels, m: Matrix, ldx: usize) !void {
        try m.check();
        if (m.k > k.k_max) return error.Exl3InputsOverScratch;
        if (ldx < m.k) return error.RowStride;
    }

    /// (x * suh) H times 2^-4 as fp16 rows into the scratch, one warp per (row, 128-block), in `f`'s order.
    fn rotate(k: *const Kernels, f: Function, s: Stream, x: u64, ldx: usize, m: Matrix, count: usize) !void {
        var a: launch_.Args = .{};
        a.add(x);
        a.add(@as(u32, @intCast(ldx)));
        a.add(m.suh);
        a.add(k.rotated.ptr);
        a.add(@as(u32, @intCast(m.k)));
        try launch_.launch(f, .{ .grid = .{ .x = @intCast(m.k / 128), .y = @intCast(count), .z = 1 }, .block = .{ .x = 32 } }, s, &a);
    }

    /// Every state's fp16 weight under `c` (65536 halves) into `out`.
    pub fn codebook(k: *const Kernels, s: Stream, c: Codebook, out: u64) !void {
        var a: launch_.Args = .{};
        a.add(@backingInt(c));
        a.add(out);
        try launch_.launch(k.codebook_fn, .{ .grid = .{ .x = 256, .y = 1, .z = 1 }, .block = .{ .x = 256 } }, s, &a);
    }

    /// W_q [k, n] fp16 (the rotated domain) into `out`, a warp per tile; suh and svh are not read.
    pub fn unpack(k: *const Kernels, s: Stream, m: Matrix, out: u64) !void {
        if (m.k % 16 != 0 or m.n % 16 != 0 or m.bits < 1 or m.bits > 8) return error.InvalidExl3Matrix;
        var a: launch_.Args = .{};
        a.add(m.trellis);
        a.add(m.args());
        a.add(@backingInt(m.codebook));
        a.add(out);
        try launch_.launch(k.unpack_fn, .{ .grid = .{ .x = @intCast(m.n / 16), .y = @intCast(m.k / 16), .z = 1 }, .block = .{ .x = 32 } }, s, &a);
    }
};

comptime {
    std.debug.assert(@sizeOf(RowsArgs) == 16 and @sizeOf(MatrixArgs) == 20);
}

test "a matrix exl3.cu cannot take is refused before any launch" {
    const base: Matrix = .{ .trellis = 0, .suh = 0, .svh = 0, .k = 4096, .n = 2048, .bits = 4, .codebook = .mul1 };
    try base.check();
    var m = base;
    m.n = 2000;
    try std.testing.expectError(error.InvalidExl3Matrix, m.check());
    m = base;
    m.bits = 9;
    try std.testing.expectError(error.InvalidExl3Matrix, m.check());
}

test "the trellis size is 16 bits a 16-value word row, 32 bytes a bit" {
    // GLM-5.3-Flash's routed gate_proj at K=4: trellis [256, 128, 64] int16
    try std.testing.expectEqual(@as(usize, 256 * 128 * 64 * 2), Matrix.trellisBytes(4096, 2048, 4));
}

test "the split comes from the shape alone and leaves no part empty" {
    try std.testing.expectEqual(@as(usize, 6), split(4096, 2048)); // a routed gate_proj: 16 column blocks, 32 input blocks
    try std.testing.expectEqual(@as(usize, 3), split(16384, 4096)); // o_proj
    try std.testing.expectEqual(@as(usize, 1), split(1536, 16384)); // q_b_proj: 128 column blocks already
    try std.testing.expectEqual(@as(usize, 1), split(512, 384));
    var k: usize = 128;
    while (k <= 16384) : (k += 128) {
        var n: usize = 128;
        while (n <= 16384) : (n += 128) {
            const p = split(k, n);
            const per = (k / 128 + p - 1) / p;
            try std.testing.expect(p * (n / 128) <= 2 * target_blocks and (p - 1) * per < k / 128);
        }
    }
}
