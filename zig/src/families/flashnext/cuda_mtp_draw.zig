//! Draws from score rows: a sampled draw reads fn_row_topk's top_k candidates (TF_FLASHNEXT_GPU_TOPK), not the row.
const std = @import("std");
const cuda = @import("cuda");
const wt = @import("cuda_prompt_wt.zig");
const mtp = @import("cuda_mtp.zig");
const lanes = @import("lanes");

const vocab = wt.vocab;
const max_rows = mtp.max_rows;
const Sampling = mtp.Sampling;
const Pick = mtp.Pick;

/// A sampled draw with top_k up to this reads candidates; a wider one (or top_k 0) downloads its whole row.
pub const topk_max: usize = 64;

/// On by default (the same draws); TF_FLASHNEXT_GPU_TOPK=0 downloads whole rows.
pub fn on() bool {
    const v = std.c.getenv("TF_FLASHNEXT_GPU_TOPK") orelse return true;
    return !std.mem.eql(u8, std.mem.span(v), "0");
}

/// Whether a draw under `s` can read fn_row_topk's candidates instead of its whole row.
pub fn usable(s: Sampling) bool {
    return s.temperature > 0 and s.top_k > 0 and s.top_k <= topk_max and on();
}

/// drawRow over a row's candidates (its top_k bf16 values and their columns, any order): choose draws from these alone.
pub fn drawTopk(gpa: std.mem.Allocator, vals: []const u16, cols: []const i32, ids: ?[]const u32, position: u64, s: Sampling) !u32 {
    var v: [topk_max]f64 = undefined;
    var keys: [topk_max]u64 = undefined;
    for (vals, cols, 0..) |b, c, i| {
        v[i] = mtp.promote(b);
        keys[i] = if (ids) |m| m[@intCast(c)] else @intCast(c);
    }
    return @intCast(try lanes.sampling.choose(gpa, v[0..vals.len], keys[0..vals.len], position, s));
}

/// The candidates of `rows` rows of `v` scores `ld` apart, k a row, into d.best past its best values, then to d.host.
fn candidates(h: *const mtp.Head, st: cuda.Stream, d: *mtp.Draft, scores: u64, v: usize, ld: usize, rows: usize, k: usize) !struct { vals: []u16, cols: []i32 } {
    const tv = d.best.ptr + max_rows * 16;
    const tc = tv + max_rows * topk_max * 2;
    try h.topk(st, scores, v, ld, rows, k, tv, tc);
    const vals = d.host[0 .. rows * k];
    const cols: []i32 = @alignCast(std.mem.bytesAsSlice(i32, std.mem.sliceAsBytes(d.host[max_rows * topk_max ..][0 .. 2 * rows * k])));
    try download(st, std.mem.sliceAsBytes(vals), tv);
    try download(st, std.mem.sliceAsBytes(cols), tc);
    return .{ .vals = vals, .cols = cols };
}

/// The head's sampled draft from its candidates: drawRow's draw, and the drawn token's probability under `lse`.
pub fn pickHead(h: *const mtp.Head, st: cuda.Stream, gpa: std.mem.Allocator, d: *mtp.Draft, lse: f32, position: u64, s: Sampling) !Pick {
    const n = h.ids.len;
    const c = try candidates(h, st, d, d.dlogits.ptr, n, n, 1, s.top_k);
    const tok = try drawTopk(gpa, c.vals, c.cols, h.ids, position, s);
    for (c.vals, c.cols) |v, col| if (h.ids[@intCast(col)] == tok) return .{ .token = tok, .prob = @exp(@as(f64, mtp.promote(v)) - @as(f64, lse)) };
    return .{ .token = tok, .prob = 0 };
}

/// The window's rows drawn at their keyed positions (fed0 + 1 + r), each within its `masks` row if any.
pub fn draws(h: *const mtp.Head, st: cuda.Stream, gpa: std.mem.Allocator, d: *mtp.Draft, sm: ?Sampling, masks: []const u32, out: []u32) !void {
    const n = d.rows;
    if (!mtp.isSampled(sm) and masks.len == 0) {
        try h.best(st, d.logits.ptr, vocab, vocab, n, d.best.ptr);
        try download(st, std.mem.sliceAsBytes(d.best_host[0 .. 4 * n]), d.best.ptr);
        for (out[0..n], 0..) |*t, r| t.* = @intCast(@as(i32, @bitCast(d.best_host[4 * r + 2])));
        return;
    }
    if (masks.len == 0) if (sm) |s| if (usable(s)) {
        const k: usize = s.top_k;
        const c = try candidates(h, st, d, d.logits.ptr, vocab, vocab, n, k);
        for (out[0..n], 0..) |*t, r| t.* = try drawTopk(gpa, c.vals[r * k ..][0..k], c.cols[r * k ..][0..k], null, d.fed0 + 1 + r, s);
        return;
    };
    const rows = d.host[0 .. n * vocab];
    try download(st, std.mem.sliceAsBytes(rows), d.logits.ptr);
    const words = masks.len / n;
    for (out[0..n], 0..) |*t, r| {
        const row = rows[r * vocab ..][0..vocab];
        if (words > 0) mtp.maskRow(row, masks[r * words ..][0..words]);
        t.* = try mtp.drawRow(gpa, row, null, d.fed0 + 1 + r, sm);
    }
}

fn download(st: cuda.Stream, dst: []u8, src: u64) !void {
    try st.synchronize();
    try st.d.check(st.d.api.cuMemcpyDtoH_v2(dst.ptr, src, dst.len), "cuMemcpyDtoH");
}
