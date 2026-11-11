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
const Scores = @import("core").proposal_rows.Scores;
const full: Scores = .{ .vocab = vocab, .logical = vocab, .stride = vocab };

/// A sampled draw with top_k up to this reads candidates; a wider one (or top_k 0) downloads its whole row.
pub const topk_max: usize = 64;

/// On by default (the same draws); TF_FLASHNEXT_GPU_TOPK=0 downloads whole rows.
pub fn on() bool {
    // the startup check refused every malformed value already, so a plain default stands here
    return @import("sys").environment.read(bool, "TF_FLASHNEXT_GPU_TOPK", true) catch true;
}

/// Whether a draw under `s` can read fn_row_topk's candidates instead of its whole row.
pub fn usable(s: Sampling) bool {
    return s.temperature > 0 and s.top_k > 0 and s.top_k <= topk_max and on();
}

/// drawRow over a row's candidates (its top_k bf16 values and their columns, any order): choose draws from these alone.
pub fn drawTopk(gpa: std.mem.Allocator, vals: []const u16, cols: []const i32, ids: ?[]const u32, position: u64, s: Sampling) !u32 {
    if (vals.len == 0 or vals.len > topk_max or vals.len != cols.len) return error.ProposalScoreShape;
    const logical: u32 = if (ids) |m| @intCast(m.len) else vocab;
    const view: Scores = .{ .vocab = vocab, .logical = logical, .stride = logical, .ids = ids };
    var v: [topk_max]f64 = undefined;
    var keys: [topk_max]u64 = undefined;
    for (vals, cols, 0..) |b, c, i| {
        v[i] = mtp.promote(b);
        if (!std.math.isFinite(v[i]) or c < 0 or c >= logical) return error.ProposalNonfinite;
        keys[i] = try view.global(@intCast(c));
    }
    return @intCast(try lanes.sampling.choose(gpa, v[0..vals.len], keys[0..vals.len], position, s));
}

/// Single and batched heads share validation before either a greedy map or a sampled draw.
pub fn pick(h: *const mtp.Head, st: cuda.Stream, gpa: std.mem.Allocator, d: *mtp.Draft, best: [4]f32, position: u64, sm: ?Sampling, batched: bool) !Pick {
    const tok = h.view().pickBest(best) catch |err| {
        badRow(st, d, best, batched, d.dlogits.ptr, h.ids.len);
        return err;
    };
    if (!mtp.isSampled(sm)) return .{ .token = tok, .prob = @exp(@as(f64, best[0]) - @as(f64, best[1])) };
    if (usable(sm.?)) return pickHead(h, st, gpa, d, best[1], position, sm.?) catch |err| {
        if (err == error.ProposalNonfinite) badRow(st, d, best, batched, d.dlogits.ptr, h.ids.len);
        return err;
    };
    const row = d.host[0..h.ids.len];
    try download(st, std.mem.sliceAsBytes(row), d.dlogits.ptr);
    return mtp.drawPick(gpa, row, h.ids, best[1], position, sm);
}

// Failure-only downloads leave the normal row selection and graph captures unchanged.
fn badRow(st: cuda.Stream, d: *mtp.Draft, best: [4]f32, batched: bool, scores: u64, n: usize) void {
    var input: [wt.wide]u16 = undefined;
    const input_ok: ?bool = if (download(st, std.mem.sliceAsBytes(&input), d.out.ptr)) |_| finite(&input) else |_| null;
    const row = d.host[0..n];
    const row_ok: ?bool = if (download(st, std.mem.sliceAsBytes(row), scores)) |_| finite(row) else |_| null;
    std.debug.print("Flash Next ProposalNonfinite stream={s} len={d} cap={d} top={d} lse={d} cache={s} batched={} input=head_streams input_finite={?} row_finite={?}\n", .{ d.label, d.len, d.cap, best[0], best[1], if (d.pooled_lane) "pool" else "own", batched, input_ok, row_ok });
}

fn finite(row: []const u16) bool {
    for (row) |b| if (!std.math.isFinite(mtp.promote(b))) return false;
    return true;
}

/// The candidates of `rows` rows of `v` scores `ld` apart, k a row, into d.best past its best values, then to d.host.
fn candidates(h: *const mtp.Head, st: cuda.Stream, d: *mtp.Draft, scores: u64, v: usize, ld: usize, rows: usize, k: usize) !struct { vals: []u16, cols: []i32 } {
    const tv = d.best.ptr + max_rows * 16;
    const tc = tv + max_rows * topk_max * 2;
    try h.topk(st, scores, v, ld, rows, k, tv, tc);
    const vals = d.host[0 .. rows * k];
    const cols: []i32 = @alignCast(std.mem.bytesAsSlice(i32, std.mem.sliceAsBytes(d.host[max_rows * topk_max ..][0 .. 2 * rows * k])));
    try st.synchronize();
    try copyReady(st, std.mem.sliceAsBytes(vals), tv);
    try copyReady(st, std.mem.sliceAsBytes(cols), tc);
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

/// A verify failure names its own row; the head's stream activations are not a verify input capture.
fn badVerify(d: *const mtp.Draft, row: usize, best: ?[4]f32, row_ok: ?bool, masked: bool) void {
    const top: ?f32 = if (best) |b| b[0] else null;
    const lse: ?f32 = if (best) |b| b[1] else null;
    std.debug.print("Flash Next ProposalNonfinite stream={s} kind=verify row={d} rows={d} len={d} cap={d} top={?} lse={?} cache={s} input=not_captured scores={s} row_finite={?}\n", .{ d.label, row, d.rows, d.len, d.cap, top, lse, if (d.pooled_lane) "pool" else "own", if (masked) "masked" else "raw", row_ok });
}

// A preceding row or candidate download has already waited for both queued kernels.
fn checkBest(st: cuda.Stream, d: *mtp.Draft, rows: ?[]const u16) !void {
    try copyReady(st, std.mem.sliceAsBytes(d.best_host[0 .. 4 * d.rows]), d.best.ptr);
    for (0..d.rows) |r| _ = full.pickBest(d.best_host[r * 4 ..][0..4].*) catch |err| {
        badVerify(d, r, d.best_host[r * 4 ..][0..4].*, if (rows) |v| finite(v[r * vocab ..][0..vocab]) else null, false);
        return err;
    };
}

/// The window's rows drawn at their keyed positions (fed0 + 1 + r), each within its `masks` row if any.
pub fn draws(h: *const mtp.Head, st: cuda.Stream, gpa: std.mem.Allocator, d: *mtp.Draft, sm: ?Sampling, masks: []const u32, out: []u32) !void {
    const n = d.rows;
    var pending: [max_rows]u32 = undefined;
    const scores = if (d.logit_paged) d.window_scores else d.logits.ptr; // on pieces: the rows its verify wrote
    if (masks.len == 0) try h.best(st, scores, vocab, vocab, n, d.best.ptr);
    if (!mtp.isSampled(sm) and masks.len == 0) {
        try st.synchronize();
        try checkBest(st, d, null);
        for (pending[0..n], 0..) |*t, r| t.* = try full.pickBest(d.best_host[r * 4 ..][0..4].*);
        @memcpy(out[0..n], pending[0..n]);
        return;
    }
    if (masks.len == 0) if (sm) |s| if (usable(s)) {
        const k: usize = s.top_k;
        const c = try candidates(h, st, d, scores, vocab, vocab, n, k);
        try checkBest(st, d, null);
        for (pending[0..n], 0..) |*t, r| t.* = drawTopk(gpa, c.vals[r * k ..][0..k], c.cols[r * k ..][0..k], null, d.fed0 + 1 + r, s) catch |err| {
            if (err == error.ProposalNonfinite) badVerify(d, r, d.best_host[r * 4 ..][0..4].*, null, false);
            return err;
        };
        @memcpy(out[0..n], pending[0..n]);
        return;
    };
    const rows = d.host[0 .. n * vocab];
    try download(st, std.mem.sliceAsBytes(rows), scores);
    if (masks.len == 0) {
        try checkBest(st, d, rows);
    } else {
        const words = masks.len / n;
        for (0..n) |r| {
            const row = rows[r * vocab ..][0..vocab];
            if (words > 0) mtp.maskRow(row, masks[r * words ..][0..words]);
            for (row) |b| {
                const v = mtp.promote(b);
                if (std.math.isNan(v) or v == std.math.inf(f32)) {
                    badVerify(d, r, null, false, true);
                    return error.ProposalNonfinite;
                }
            }
        }
    }
    for (pending[0..n], 0..) |*t, r| {
        t.* = try mtp.drawRow(gpa, rows[r * vocab ..][0..vocab], null, d.fed0 + 1 + r, sm);
    }
    @memcpy(out[0..n], pending[0..n]);
}

fn copyReady(st: cuda.Stream, dst: []u8, src: u64) !void {
    try cuda.memory.download(st.d, src, dst);
}

fn download(st: cuda.Stream, dst: []u8, src: u64) !void {
    try st.synchronize();
    try copyReady(st, dst, src);
}
