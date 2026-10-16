//! GLM's bf16 linears on tensor cores (glm_mm): weights tiled once at load, any row count, row-stable K slices.
const std = @import("std");
const cuda = @import("cuda");

/// A linear's tiled weights: [n / 8][k / 64][32 lanes][8 words] bf16 pairs (glm_mm_tile16's layout).
pub const Tiled = struct { w: u64, n: usize, k: usize };

/// glm_mm's bf16 kernels by row bucket, the K-slice reduce and the tiling.
pub const Kernels = struct {
    by_rows: [4]cuda.Function, // 16, 32, 64, 128 rows a block
    prompt: cuda.Function, // 128-row blocks, row blocks fastest: a prompt's blocks share each weight tile in L2
    reduce: cuda.Function,
    tile: cuda.Function,
    pdl: bool, // dependent launches: a matmul's weights stream in while the kernel before it finishes

    pub fn load(m: cuda.Module) !Kernels {
        var k: Kernels = .{ .by_rows = undefined, .prompt = try m.function("glm_mm16_rows"), .reduce = try m.function("glm_mm_reduce"), .tile = try m.function("glm_mm_tile16"), .pdl = true };
        inline for (.{ "glm_mm16_16", "glm_mm16_32", "glm_mm16_64", "glm_mm16_128" }, 0..) |name, i| {
            k.by_rows[i] = try m.function(name);
            try k.by_rows[i].allowDynamicShared(smem(bucket(i)));
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
    return @intCast(3 * (bm * 128 + 8 * 32 * 8 * 4 + 128));
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

/// x [m][k] (rows `ldx` apart) by `w` -> out [m][n], bf16 or fp32 (`wide`); `part`: partBytes(m, w).
pub fn matmul(k: *const Kernels, s: cuda.Stream, x: u64, ldx: usize, w: Tiled, out: u64, part: u64, m: usize, wide: bool) !void {
    if (w.k % 64 != 0 or w.n % 8 != 0 or x % 16 != 0 or ldx % 8 != 0) return error.BadShape;
    var done: usize = 0;
    while (done < m) { // windows to 64 rows a launch with K slices; prompts in one launch of 128-row blocks, one slice
        const big = m - done >= 128;
        const rows = if (big) m - done else @min(m - done, 64);
        const i: usize = if (big) 3 else if (rows <= 16) 0 else if (rows <= 32) 1 else 2;
        const bm = bucket(i);
        const sk = if (big) 1 else splitK(w.n, w.k); // a prompt's blocks share each weight tile in L2: no slices
        const o = out + done * w.n * @as(usize, if (wide) 4 else 2);
        var a: cuda.Args = .{};
        a.add(x + done * ldx * 2);
        a.add(@as(c_longlong, @intCast(ldx)));
        a.add(w.w);
        a.add(@as(u64, 0));
        a.add(o);
        a.add(part);
        a.add(@as(c_int, @intFromBool(wide)));
        a.add(@as(c_int, @intCast(rows)));
        a.add(@as(c_int, @intCast(w.n)));
        a.add(@as(c_int, @intCast(w.k)));
        a.add(@as(c_int, @intCast(sk)));
        const cols: u32 = @intCast((w.n + 63) / 64);
        const row_blocks: u32 = @intCast((rows + bm - 1) / bm);
        if (big) {
            try cuda.launch.launch(k.prompt, .{ .grid = .{ .x = row_blocks, .y = cols }, .block = .{ .x = 256 }, .shared = smem(128), .pdl = k.pdl }, s, &a);
        } else try cuda.launch.launch(k.by_rows[i], .{ .grid = .{ .x = cols, .y = row_blocks, .z = @intCast(sk) }, .block = .{ .x = if (bm <= 32) 128 else 256 }, .shared = smem(bm), .pdl = k.pdl }, s, &a);
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
