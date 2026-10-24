//! GLM FP8 codes/scales against a CPU format reference, then the existing BF16 attention math and FP64 bounds.
const std = @import("std");
const glm = @import("glm_cuda");
const check = @import("check.zig");
const ref = @import("glm_ref_tests.zig");
const attn = @import("glm_ref_attn.zig");
const Rig = @import("quant_ref_tests.zig").RigT;
const latent = glm.latent;
const codec = glm.kv8_ref;
const lw = 512;
const heads = 32;
fn partials(r: *Rig, chunks: usize, rows: usize) !latent.Partials {
    return .{ .po = try r.zeros(chunks * rows * heads * lw * 4), .pm = try r.zeros(chunks * rows * heads * 4), .pl = try r.zeros(chunks * rows * heads * 4) };
}
fn position(r: *Rig, value: usize) !u64 {
    const p: i32 = @intCast(value);
    return r.dev(std.mem.asBytes(&p));
}
pub fn run(r: *Rig, set: glm.glue.Set) !void {
    for ([_]usize{ 3, 513 }) |keys| {
        const capacity = keys + 17;
        const stride = lw + 8;
        const x = try r.bfs(keys * stride, 1.0);
        @memset(x[0..lw], 0);
        @memset(x[stride..][0..lw], 1);
        const input = try r.dev(std.mem.sliceAsBytes(x));
        const fp_cache = try r.zeros(try latent.cacheBytes(capacity, lw, true));
        const scales = fp_cache + capacity * lw;
        try latent.latentWriteFormat(set, r.s, input, stride, fp_cache, scales, .{}, try position(r, 0), keys, lw);
        const want_codes = try r.a.alloc(u8, keys * lw);
        const want_scales = try r.a.alloc(f32, keys);
        const decoded = try r.a.alloc(u16, capacity * lw);
        @memset(decoded, 0x7fc0);
        const cd = try r.a.alloc(f64, keys * lw);
        for (0..keys) |row| {
            const codes = want_codes[row * lw ..][0..lw];
            want_scales[row] = try codec.pack(x[row * stride ..][0..lw], codes);
            for (codes, 0..) |code, col| {
                const at = row * lw + col;
                decoded[at] = codec.decoded(code, want_scales[row]);
                cd[at] = codec.bf(decoded[at]);
            }
        }
        try check.sameBytes("GLM FP8 row codes", try r.back(fp_cache, keys * lw), want_codes);
        try check.sameBytes("GLM FP8 row scales", try r.back(scales, keys * 4), std.mem.sliceAsBytes(want_scales));
        for ([_]usize{ 1, 7 }) |chunk| {
            if (keys > 3 and chunk == 1) continue;
            const split = try r.zeros(try latent.cacheBytes(capacity, lw, true));
            const ss = split + capacity * lw;
            var row: usize = 0;
            while (row < keys) : (row += chunk) try latent.latentWriteFormat(set, r.s, input + row * stride * 2, stride, split, ss, .{}, try position(r, row), @min(chunk, keys - row), lw);
            try check.sameBytes("GLM FP8 chunk-independent row codes", try r.back(split, keys * lw), want_codes);
            try check.sameBytes("GLM FP8 chunk-independent scales", try r.back(ss, keys * 4), std.mem.sliceAsBytes(want_scales));
        }
        const bf_cache = try r.dev(std.mem.sliceAsBytes(decoded));
        const rows = 3;
        const q = try r.bfs(rows * heads * lw, 1.0);
        const dq = try r.dev(std.mem.sliceAsBytes(q));
        const pos = keys - rows;
        const at = try position(r, pos);
        const nch = latent.chunksFor(keys);
        const key_sets = try r.a.alloc([]const u32, rows);
        for (key_sets, 0..) |*slot, row| {
            const ids = try r.a.alloc(u32, pos + row + 1);
            for (ids, 0..) |*id, i| id.* = @intCast(i);
            slot.* = ids;
        }
        const want = try attn.reference(r, q, cd, key_sets, 0.125);
        for ([_]usize{ 16, 32 }) |hb| {
            const fp_out = try r.zeros(rows * heads * lw * 2);
            const bf_out = try r.zeros(rows * heads * lw * 2);
            try latent.attentionFormat(set, r.s, dq, fp_cache, scales, .{}, at, try partials(r, nch, rows), fp_out, try r.zeros(4), rows, heads, lw, nch, 0.125, hb);
            try latent.attention(set, r.s, dq, bf_cache, .{}, at, try partials(r, nch, rows), bf_out, try r.zeros(4), rows, heads, lw, nch, 0.125, hb);
            const wide = try r.back(fp_out, rows * heads * lw * 2);
            try check.sameBytes("GLM FP8 reader equals reconstructed BF16 cache", wide, try r.back(bf_out, wide.len));
            try ref.closeCols(r, "GLM FP8 reconstructed attention vs FP64", fp_out, false, want.want, want.tol, lw);
            for (0..rows) |row| {
                const one = try r.zeros(heads * lw * 2);
                try latent.attentionFormat(set, r.s, dq + row * heads * lw * 2, fp_cache, scales, .{}, try position(r, pos + row), try partials(r, nch, 1), one, try r.zeros(4), 1, heads, lw, nch, 0.125, hb);
                try check.sameBytes("GLM FP8 concurrent row equals solo row", try r.back(one, heads * lw * 2), wide[row * heads * lw * 2 ..][0 .. heads * lw * 2]);
            }
        }
        const tokens = [_]i32{ 0, -1, -1, -1, -1, -1, 0, 1, 2 };
        const counts = [_]i32{ 1, 0, 3 };
        const dt = try r.dev(std.mem.asBytes(&tokens));
        const dc = try r.dev(std.mem.asBytes(&counts));
        const sparse_fp = try r.zeros(rows * heads * lw * 2);
        const sparse_bf = try r.zeros(rows * heads * lw * 2);
        try latent.sparseAttentionFormat(set, r.s, dq, fp_cache, scales, .{}, dt, dc, 3, try partials(r, 1, rows), sparse_fp, rows, heads, lw, 0.125);
        try latent.sparseAttention(set, r.s, dq, bf_cache, .{}, dt, dc, 3, try partials(r, 1, rows), sparse_bf, rows, heads, lw, 0.125);
        try check.sameBytes("GLM FP8 sparse reader and count-zero preservation", try r.back(sparse_fp, rows * heads * lw * 2), try r.back(sparse_bf, rows * heads * lw * 2));
    }
    check.pass("GLM FP8 synthetic format, masking, row/chunk identity and reconstructed BF16/FP64 attention", .{});
}
