//! Paged E4M3 cache writes and attention equal contiguous E4M3 through shuffled pages across row 512.
const std = @import("std");
const glm = @import("glm_cuda");
const check = @import("check.zig");
const Rig = @import("quant_ref_tests.zig").RigT;
const latent = glm.latent;
const lw = 512;
const heads = 32;

fn position(r: *Rig, value: usize) !u64 {
    const v: i32 = @intCast(value);
    return r.dev(std.mem.asBytes(&v));
}

fn partials(r: *Rig, chunks: usize, rows: usize) !latent.Partials {
    return .{ .po = try r.zeros(chunks * rows * heads * lw * 4), .pm = try r.zeros(chunks * rows * heads * 4), .pl = try r.zeros(chunks * rows * heads * 4) };
}

pub fn run(r: *Rig, set: glm.glue.Set) !void {
    const keys = 513;
    const capacity = 1024;
    const stride = lw + 8;
    const x = try r.bfs(keys * stride, 1.0);
    @memset(x[0..lw], 0);
    const input = try r.dev(std.mem.sliceAsBytes(x));
    const contiguous = try r.zeros(try latent.cacheBytes(capacity, lw, true));
    const contiguous_scales = contiguous + capacity * lw;
    const part_bytes = 512 * (lw + 4) + 128 * 128 * 2;
    const page_bytes = 2 * part_bytes;
    const arena = try r.zeros(3 * page_bytes);
    const paged = arena + part_bytes;
    const paged_scales = paged + 512 * lw;
    const ids = [_]i32{ 2, 0 };
    const table: latent.Table = .{ .pt = try r.dev(std.mem.asBytes(&ids)), .ps = page_bytes, .logical = true };
    const at = try position(r, 0);
    try latent.latentWriteFormat(set, r.s, input, stride, contiguous, contiguous_scales, .{}, at, keys, lw);
    try latent.latentWriteFormat(set, r.s, input, stride, paged, paged_scales, table, at, keys, lw);
    const codes = try r.back(contiguous, keys * lw);
    const scales = try r.back(contiguous_scales, keys * 4);
    for (ids, 0..) |physical, logical| {
        const first = logical * 512;
        const rows = @min(512, keys - first);
        const offset: u64 = @as(u64, @intCast(physical)) * page_bytes;
        try check.sameBytes("GLM paged FP8 codes equal contiguous", try r.back(paged + offset, rows * lw), codes[first * lw ..][0 .. rows * lw]);
        try check.sameBytes("GLM paged FP8 scales equal contiguous", try r.back(paged_scales + offset, rows * 4), scales[first * 4 ..][0 .. rows * 4]);
    }
    const untouched = try r.back(arena + page_bytes, page_bytes);
    for (untouched) |byte| if (byte != 0xff) return error.Fp8WriteTouchedUnmappedPage;

    for ([_]usize{ 3, 64 }) |rows| {
        const hb: usize = if (rows < 64) 16 else 32;
        const q = try r.dev(std.mem.sliceAsBytes(try r.bfs(rows * heads * lw, 1.0)));
        const nch = latent.chunksFor(keys);
        const output_bytes = rows * heads * lw * 2;
        const fp_out = try r.zeros(output_bytes);
        const page_out = try r.zeros(output_bytes);
        const pos = try position(r, keys - rows);
        try latent.attentionFormat(set, r.s, q, contiguous, contiguous_scales, .{}, pos, try partials(r, nch, rows), fp_out, try r.zeros(4), rows, heads, lw, nch, 0.125, hb);
        try latent.attentionFormat(set, r.s, q, paged, paged_scales, table, pos, try partials(r, nch, rows), page_out, try r.zeros(4), rows, heads, lw, nch, 0.125, hb);
        try check.sameBytes("GLM paged FP8 dense equals contiguous at both head widths", try r.back(page_out, output_bytes), try r.back(fp_out, output_bytes));

        const width = 7;
        const tokens = try r.a.alloc(i32, rows * width);
        const counts = try r.a.alloc(i32, rows);
        for (0..rows) |row| {
            @memcpy(tokens[row * width ..][0..width], &[_]i32{ 512, 0, 511, 42, 500, 13, 222 });
            counts[row] = if (row == 1) 0 else width;
        }
        const dt = try r.dev(std.mem.sliceAsBytes(tokens));
        const dc = try r.dev(std.mem.sliceAsBytes(counts));
        const sparse_fp = try r.zeros(output_bytes);
        const sparse_page = try r.zeros(output_bytes);
        try latent.sparseAttentionFormat(set, r.s, q, contiguous, contiguous_scales, .{}, dt, dc, width, try partials(r, 1, rows), sparse_fp, rows, heads, lw, 0.125);
        try latent.sparseAttentionFormat(set, r.s, q, paged, paged_scales, table, dt, dc, width, try partials(r, 1, rows), sparse_page, rows, heads, lw, 0.125);
        try check.sameBytes("GLM paged FP8 sparse equals contiguous at both head widths", try r.back(sparse_page, output_bytes), try r.back(sparse_fp, output_bytes));
    }
    check.pass("GLM paged FP8 shuffled pages: codes/scales and dense/sparse 16/32-head attention equal contiguous", .{});
}
