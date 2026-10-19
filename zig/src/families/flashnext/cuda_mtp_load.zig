//! The draft vocabulary borrows exact rows of the checkpoint's dense or affine language head.
const std = @import("std");
const cuda = @import("cuda");
const wt = @import("cuda_prompt_wt.zig");
const core = @import("core");
const projection = @import("cuda_projection.zig");

pub const Mode = enum { bf16, affine4 };
pub fn mode() !Mode {
    const value = if (std.c.getenv("TF_FLASHNEXT_DRAFT_HEAD")) |v| std.mem.span(v) else "bf16";
    if (std.mem.eql(u8, value, "bf16")) return .bf16;
    if (std.mem.eql(u8, value, "affine4")) return .affine4;
    return error.BadDraftHead;
}
pub fn draft(a: std.mem.Allocator, d: *cuda.Driver, store: *wt.Store, artifact: *const core.proposal_vocab.Artifact, role: Mode) !wt.Q4 {
    const ids = artifact.ids();
    const affine = role == .affine4;
    const tensor = try store.tensor("language_model.lm_head.weight");
    if (tensor.dtype == .bf16) {
        if (!tensor.is(.bf16, &.{ wt.vocab, wt.dims })) return error.UnexpectedTensor;
        var selected = try (core.proposal_rows.Source{ .identity = artifact.identity(), .storage = .bf16, .recipe = .qlinear_decode, .codes = .{ .bytes = tensor.bytes, .stride = wt.dims * 2 } }).extract(a, artifact, 1, std.math.maxInt(usize));
        defer selected.deinit();
        const rows = selected.codes;
        if (affine) return quantized(a, d, rows, ids.len);
        return projection.dense(a, d, .{ .dtype = .bf16, .rank = 2, .shape = .{ ids.len, wt.dims, 1, 1, 1 }, .bytes = rows });
    }
    var buf: [120]u8 = undefined;
    const head = try wt.join3(a, store, &buf, "language_model.lm_head", "");
    defer a.free(head.w);
    defer a.free(head.s);
    defer a.free(head.b);
    const wrow = wt.dims / 2;
    const grow = wt.dims / 32 * 2;
    if (head.w.len != wt.vocab * wrow or head.s.len != wt.vocab * grow) return error.UnexpectedTensor;
    var selected = try (core.proposal_rows.Source{ .identity = artifact.identity(), .storage = .affine, .recipe = .qlinear_decode, .scale_dtype = .bf16, .bias_dtype = .bf16, .codes = .{ .bytes = head.w, .stride = wrow }, .scales = .{ .bytes = head.s, .stride = grow }, .biases = .{ .bytes = head.b, .stride = grow } }).extract(a, artifact, 1, std.math.maxInt(usize));
    defer selected.deinit();
    return wt.uploadQ4(a, d, selected.codes, selected.scales, selected.biases, ids.len, wt.dims / 8);
}

fn quantized(a: std.mem.Allocator, d: *cuda.Driver, rows: []const u8, n: usize) !wt.Q4 {
    const words = try a.alloc(u8, n * wt.dims / 2);
    defer a.free(words);
    const scales = try a.alloc(u8, n * wt.dims / 16);
    defer a.free(scales);
    const biases = try a.alloc(u8, scales.len);
    defer a.free(biases);
    for (0..n) |i| try @import("cuda_draft_affine.zig").row(rows[i * wt.dims * 2 ..][0 .. wt.dims * 2], words[i * wt.dims / 2 ..][0 .. wt.dims / 2], scales[i * wt.dims / 16 ..][0 .. wt.dims / 16], biases[i * wt.dims / 16 ..][0 .. wt.dims / 16]);
    return wt.uploadQ4(a, d, words, scales, biases, n, wt.dims / 8);
}
