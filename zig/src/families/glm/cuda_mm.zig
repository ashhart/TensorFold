//! GLM's bf16 linears on tensor cores (glm_mm): weights tiled once at load, any row count, row-stable K slices.
const std = @import("std");
const cuda = @import("cuda");

/// A linear's tiles: bf16 pairs (glm_mm_tile16), or `bits`-bit codes (glm_mm_tile4/6/8) with fp16 scales at `s`.
pub const Tiled = struct { w: u64, n: usize, k: usize, s: u64 = 0, bits: u8 = 16 };

/// The widths glm_mm packs to, by kernel set: bf16 first.
const widths = [_]u8{ 16, 4, 6, 8 };

fn widthIndex(bits: u8) !usize {
    return std.mem.indexOfScalar(u8, &widths, bits) orelse error.BadBits;
}

/// Bytes a packed linear's codes take (its scales follow, 256-aligned): what fits in place of its bf16 tiles.
pub fn packedBytes(n: usize, k: usize, bits: u8) [2]usize {
    return .{ std.mem.alignForward(usize, n * k * bits / 8, 256), n * (k / 64) * 2 };
}

/// The linears packed at load: each one's tiles pointer (sorted) and its bits; a linear not here is bf16.
pub const Packed = struct {
    ptrs: []const u64 = &.{},
    bits: []const u8 = &.{},

    /// The tiled weight `w` [n][k] as the matmul reads it: packed codes with their scales after them, or bf16.
    pub fn tiled(p: *const Packed, w: u64, n: usize, k: usize) Tiled {
        var lo: usize = 0;
        var hi: usize = p.ptrs.len;
        while (lo < hi) {
            const mid = (lo + hi) / 2;
            if (p.ptrs[mid] < w) lo = mid + 1 else hi = mid;
        }
        if (lo == p.ptrs.len or p.ptrs[lo] != w) return .{ .w = w, .n = n, .k = k };
        const b = p.bits[lo];
        return .{ .w = w, .n = n, .k = k, .s = w + packedBytes(n, k, b)[0], .bits = b };
    }
};

/// glm_mm's bf16 kernels by row bucket, the K-slice reduce and the tiling.
pub const Kernels = struct {
    by_rows: [4]cuda.Function, // bf16: 16, 32, 64, 128 rows a block
    packed_rows: [3][4]cuda.Function, // 4, 6 and 8 bits, as by_rows
    prompt: cuda.Function, // 128-row blocks, row blocks fastest: a prompt's blocks share each weight tile in L2
    reduce: cuda.Function,
    tile: cuda.Function,
    tile_packed: [3]cuda.Function, // glm_mm_tile4/6/8: packed rows and their scales into the tiles
    pdl: bool, // dependent launches: a matmul's weights stream in while the kernel before it finishes

    pub fn load(m: cuda.Module) !Kernels {
        var k: Kernels = .{ .by_rows = undefined, .packed_rows = undefined, .tile_packed = undefined, .prompt = try m.function("glm_mm16_rows"), .reduce = try m.function("glm_mm_reduce"), .tile = try m.function("glm_mm_tile16"), .pdl = true };
        inline for (.{ "glm_mm16_16", "glm_mm16_32", "glm_mm16_64", "glm_mm16_128" }, 0..) |name, i| {
            k.by_rows[i] = try m.function(name);
            try k.by_rows[i].allowDynamicShared(smem(bucket(i)));
        }
        inline for (widths[1..], 0..) |b, wi| {
            inline for (.{ 16, 32, 64, 128 }, 0..) |bm, i| {
                k.packed_rows[wi][i] = try m.function(std.fmt.comptimePrint("glm_mm{d}_{d}", .{ b, bm }));
                try k.packed_rows[wi][i].allowDynamicShared(sharedBytes(b, bm));
            }
            k.tile_packed[wi] = try m.function(std.fmt.comptimePrint("glm_mm_tile{d}", .{b}));
        }
        try k.prompt.allowDynamicShared(smem(128));
        if (std.c.getenv("TF_GLM_PDL")) |v| k.pdl = !std.mem.eql(u8, std.mem.span(v), "0");
        return k;
    }
};

const split_target = 192; // column tiles times K slices a launch aims at (the GB10's SMs, a few blocks each)

fn bucket(i: usize) usize {
    return @as(usize, 16) << @intCast(i);
}

fn smem(bm: usize) u32 {
    return sharedBytes(16, bm);
}

/// A block's shared memory: three stages of `bm` rows of x and a 64-column tile of weights (`bits / 2` words a lane).
pub fn sharedBytes(bits: u8, bm: usize) u32 {
    const wpc: usize = if (bits == 16) 8 else bits / 2;
    return @intCast(3 * (bm * 128 + 8 * 32 * wpc * 4 + 128));
}

/// K slices fixed by the shape: column tiles times slices reach the target, each slice at least 4 groups of 64.
pub fn splitK(n: usize, k: usize) usize {
    const tiles = (n + 63) / 64;
    const steps = k / 64;
    var sk: usize = 1;
    while (sk < 8 and tiles * sk < split_target and steps % (sk * 2) == 0 and steps / (sk * 2) >= 4) sk *= 2;
    return sk;
}

/// Bytes of fp32 partials a call of `m` rows needs (0: one slice; windows past 64 rows go 64 at a time).
pub fn partBytes(m: usize, w: Tiled) usize {
    const sk = splitK(w.n, w.k);
    return if (sk > 1) sk * @min(m, 64) * w.n * 4 else 0;
}

/// x [m][k] (`ldx` apart) by `w` -> out [m][n] (`wide`: fp32); `one`: prompt blocks at any rows, sums per row alike.
pub fn matmul(k: *const Kernels, s: cuda.Stream, x: u64, ldx: usize, w: Tiled, out: u64, part: u64, m: usize, wide: bool, one: bool) !void {
    if (w.k % 64 != 0 or w.n % 8 != 0 or x % 16 != 0 or ldx % 8 != 0) return error.BadShape;
    var done: usize = 0;
    const wi = try widthIndex(w.bits);
    while (done < m) { // windows to 64 rows a launch with K slices; prompts in one launch of 128-row blocks, one slice
        const big = one or m - done >= 128;
        const rows = if (big) m - done else @min(m - done, 64);
        const i: usize = if (big) 3 else if (rows <= 16) 0 else if (rows <= 32) 1 else 2;
        const bm = bucket(i);
        const sk = if (big) 1 else splitK(w.n, w.k); // a prompt's blocks share each weight tile in L2: no slices
        const o = out + done * w.n * @as(usize, if (wide) 4 else 2);
        var a: cuda.Args = .{};
        a.add(x + done * ldx * 2);
        a.add(@as(c_longlong, @intCast(ldx)));
        a.add(w.w);
        a.add(w.s);
        a.add(o);
        a.add(part);
        a.add(@as(c_int, @intFromBool(wide)));
        a.add(@as(c_int, @intCast(rows)));
        a.add(@as(c_int, @intCast(w.n)));
        a.add(@as(c_int, @intCast(w.k)));
        a.add(@as(c_int, @intCast(sk)));
        const cols: u32 = @intCast((w.n + 63) / 64);
        const row_blocks: u32 = @intCast((rows + bm - 1) / bm);
        if (big and wi == 0) {
            try cuda.launch.launch(k.prompt, .{ .grid = .{ .x = row_blocks, .y = cols }, .block = .{ .x = 256 }, .shared = smem(128), .pdl = k.pdl }, s, &a);
        } else { // packed prompts take the 128-row bucket's blocks, one slice
            const f = if (wi == 0) k.by_rows[i] else k.packed_rows[wi - 1][i];
            try cuda.launch.launch(f, .{ .grid = .{ .x = cols, .y = row_blocks, .z = @intCast(sk) }, .block = .{ .x = if (bm <= 32) 128 else 256 }, .shared = sharedBytes(w.bits, bm), .pdl = k.pdl }, s, &a);
        }
        if (sk > 1) {
            const total = rows * w.n;
            var r: cuda.Args = .{};
            r.add(part);
            r.add(o);
            r.add(@as(c_longlong, @intCast(total)));
            r.add(@as(c_int, @intCast(sk)));
            r.add(@as(c_int, @intFromBool(wide)));
            try cuda.launch.launch(k.reduce, .{ .grid = .{ .x = @intCast((total + 255) / 256) }, .block = .{ .x = 256 } }, s, &r);
        }
        done += rows;
    }
}

/// Packed rows (cuda_qpack's) and fp16 scales [n][k / 64] at `src`, `ss` tiled into `dst`, `ds` (`bits` 4/6/8).
pub fn tilePacked(k: *const Kernels, s: cuda.Stream, bits: u8, src: u64, ss: u64, dst: u64, ds: u64, n: usize, kk: usize) !void {
    const wi = try widthIndex(bits);
    if (wi == 0) return error.BadBits;
    const chunks = n / 8 * (kk / 64) * 32;
    var a: cuda.Args = .{};
    a.add(src);
    a.add(ss);
    a.add(dst);
    a.add(ds);
    a.add(@as(c_int, @intCast(n)));
    a.add(@as(c_int, @intCast(kk)));
    try cuda.launch.launch(k.tile_packed[wi - 1], .{ .grid = .{ .x = @intCast((chunks + 255) / 256) }, .block = .{ .x = 256 } }, s, &a);
}

/// A row-major bf16 [n][k] linear at `src` tiled into `dst` (as many bytes).
pub fn tile(k: *const Kernels, s: cuda.Stream, src: u64, dst: u64, n: usize, kk: usize) !void {
    const chunks = n / 8 * (kk / 64) * 32;
    var a: cuda.Args = .{};
    a.add(src);
    a.add(dst);
    a.add(@as(c_int, @intCast(n)));
    a.add(@as(c_int, @intCast(kk)));
    try cuda.launch.launch(k.tile, .{ .grid = .{ .x = @intCast((chunks + 255) / 256) }, .block = .{ .x = 256 } }, s, &a);
}

test "K slices as the shapes take them" {
    try std.testing.expectEqual(@as(usize, 1), splitK(12288, 4096)); // 192 column tiles already
    try std.testing.expectEqual(@as(usize, 4), splitK(4096, 2048)); // 64 tiles, 32 groups: 4 slices of 8
}
