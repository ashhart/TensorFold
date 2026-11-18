//! Stateful MTP absorption keeps keys during a priced skip; proposal arithmetic runs only for useful drafts.
const std = @import("std");
const lanes = @import("lanes");
const Normalized = lanes.backend.Features.Normalized;
const mtp = @import("cuda_mtp.zig");
const Mtp = mtp.Mtp;
const Ctx = mtp.Ctx;
const TapRows = mtp.TapRows;
const engine = @import("cuda_engine.zig");
const precision = @import("cuda_draft_precision.zig");
const max_rows = engine.most_verify;
pub const prompt_kv_tile = engine.most_verify * 4;
const most_window = engine.most_window;
const ranks = lanes.shape.ranks;

/// Larger KV-only prompt tiles amortize repeated row-independent MTP launches.
pub fn nextChunkRows(prompt: bool, propose: bool, remaining: usize) usize {
    if (remaining == 0) return 0;
    if (prompt) {
        const proposal_tail: usize = if (propose) max_rows else 0;
        const kv_rows = remaining -| proposal_tail;
        if (kv_rows > 0) return @min(kv_rows, prompt_kv_tile);
    }
    return @min(remaining, max_rows);
}

/// Skip only prompt slices known to end before the final prompt row; unknown positions keep proposals.
pub fn proposalNeeded(prompt: bool, requested: bool, start: u64, rows: usize, prompt_len: usize) bool {
    if (!requested or rows == 0) return false;
    if (!prompt) return true;
    const prompt_end: u64 = @intCast(prompt_len);
    if (start >= prompt_end) return true;
    const remaining: usize = @intCast(prompt_end - start);
    return rows >= remaining;
}

/// The batch kernel always proposes for each row, so prompt items are batchable only at the final row.
pub fn batchProposalNeeded(prompt: bool, requested: bool, start: u64, rows: usize, prompt_len: usize) bool {
    return proposalNeeded(prompt, requested, start, rows, prompt_len) and (!prompt or rows == 1);
}

pub fn batchProposalAllowed(items: []const lanes.drafter.Absorb) bool {
    for (items) |it| {
        const rows: usize = if (it.rows) |rs| rs.len else @intCast(it.features.rows);
        if (!batchProposalNeeded(it.prompt, it.propose, it.start, rows, it.stream.prompt_len)) return false;
    }
    return true;
}

/// Within a multi-chunk prompt absorb, only the final chunk can feed a proposal.
pub fn chunkProposalNeeded(prompt: bool, requested: bool, chunk_rows: usize, remaining_rows: usize) bool {
    if (!requested) return false;
    return !prompt or chunk_rows == remaining_rows;
}

pub const NormalizedRange = struct { start: u64, skip: usize, rows: usize };

/// The overlap between requested prompt rows and an ephemeral normalized view.
pub fn normalizedRange(start: u64, rows: usize, source_start: u64, source_rows: usize) ?NormalizedRange {
    const row_count = std.math.cast(u64, rows) orelse return null;
    const source_row_count = std.math.cast(u64, source_rows) orelse return null;
    const end = std.math.add(u64, start, row_count) catch return null;
    const source_end = std.math.add(u64, source_start, source_row_count) catch return null;
    const first = @max(start, source_start);
    const last = @min(end, source_end);
    if (last <= first) return null;
    return .{ .start = first, .skip = @intCast(first - start), .rows = @intCast(last - first) };
}

pub fn projKv(d: *Mtp, rows: usize) !void {
    try d.e.projManySums(&.{ d.w.k, d.w.v }, d.x, d.xs, &.{ d.kraw, d.vb }, rows, precision.prompt(d.projection_role, rows, precision.mtp_projection_prompt_cutoff), 0);
}

pub fn absorbOne(d: *Mtp, it: lanes.drafter.Absorb) !void {
    const ctx = d.streams.get(it.stream) orelse return error.UnknownStream;
    if (it.features.space != .device or it.features.dtype != .bf16) return error.UnsupportedFeatures;
    const rows: usize = if (it.rows) |rs| rs.len else @intCast(it.features.rows);
    const propose = proposalNeeded(it.prompt, it.propose, it.start, rows, it.stream.prompt_len);
    if (it.features.normalized) |source| {
        if (it.rows != null) return error.UnsupportedNormalizedFeatures;
        try d.absorbNormalizedRole(ctx, source, it.start, rows, it.follow, it.prompt, propose, it.stream.sampling);
    } else if (it.rows) |rs| {
        try d.absorbRole(ctx, it.features.buffer, it.features.offset, rs.len, rs, it.follow, it.prompt, propose, it.stream.sampling);
    } else try d.absorbRole(ctx, it.features.buffer, it.features.offset, it.features.rows, null, it.follow, it.prompt, propose, it.stream.sampling);
}

pub fn absorbRole(d: *Mtp, ctx: *Ctx, tap_ring: u64, start_: u64, n_: usize, rows_: ?[]const u32, follow: []const u32, prompt: bool, propose: bool, sampling: ?lanes.Sampling) !void {
    return absorbRoleSource(d, ctx, tap_ring, null, start_, n_, rows_, follow, prompt, propose, sampling);
}

pub fn absorbNormalizedRole(d: *Mtp, ctx: *Ctx, source: Normalized, start_: u64, n_: usize, follow: []const u32, prompt: bool, propose: bool, sampling: ?lanes.Sampling) !void {
    return absorbRoleSource(d, ctx, null, source, start_, n_, null, follow, prompt, propose, sampling);
}

fn absorbRoleSource(d: *Mtp, ctx: *Ctx, tap_ring: ?u64, normalized: ?Normalized, start_: u64, n_: usize, rows_: ?[]const u32, follow: []const u32, prompt: bool, propose: bool, sampling: ?lanes.Sampling) !void {
    const e = d.e;
    const g = e.g;
    var start = start_;
    var n = n_;
    var fi: usize = 0;
    if (follow.len < n) return error.BadDraftRows;
    if (rows_) |rs| if (rs.len != n or n > most_window) return error.BadDraftRows;
    if (normalized) |source| {
        if (!prompt or rows_ != null or source.row_bytes != @as(u32, @intCast(g.hidden * 2)) or source.buffer == 0) return error.UnsupportedNormalizedFeatures;
        const range = normalizedRange(start, n, source.start, source.rows) orelse return;
        const range_end = std.math.add(u64, range.start, std.math.cast(u64, range.rows) orelse
            return error.NormalizedFeaturesOutOfRange) catch return error.NormalizedFeaturesOutOfRange;
        const requested_end = std.math.add(u64, start, std.math.cast(u64, n) orelse
            return error.NormalizedFeaturesOutOfRange) catch return error.NormalizedFeaturesOutOfRange;
        if (range_end != requested_end) return error.NormalizedFeaturesOutOfRange;
        start = range.start;
        fi += range.skip;
        n = range.rows;
    } else if (tap_ring == null) return error.MissingMtpFeatures;
    const previous = d.projection_role;
    defer d.projection_role = previous;
    d.projection_role = if (prompt) .prompt else .live;
    if (start < ctx.end) ctx.end = start;
    if (start > ctx.end) ctx.begin = start; // a gap (a prompt restored from a cache): the context starts here
    // the target's ring holds this many rows behind its newest forward
    const held = @import("cuda_cache.zig").tap_window;
    if (n > held) {
        const skipped = n - held;
        fi += skipped;
        start += skipped;
        n = held;
        ctx.begin = start;
    }
    if (n == 0) return;
    if (n > 0) try d.putIds(follow[fi..][0..n]);
    ctx.end = start;
    var id_at: usize = 0;
    const tap_ring_rows: c_int = @intCast(e.tapRows());
    while (n > 0) {
        const rows = nextChunkRows(prompt, propose, n);
        const propose_chunk = chunkProposalNeeded(prompt, propose, rows, n);
        var input_h = d.hin;
        if (normalized) |source| {
            input_h = source.buffer + source.byte_offset + (start - source.start) * @as(u64, source.row_bytes);
        } else {
            const ring = tap_ring.?;
            if (rows_) |rs| {
                var tr: TapRows = .{ .n = @intCast(rows), .rows = @splat(0) };
                for (rs[fi..][0..rows], 0..) |r, i| tr.rows[i] = @intCast(r);
                try e.run(d.ring_fns.gather_rows, .{ rows, 1, 1 }, .{ 256, 1, 1 }, .{ ring, tap_ring_rows, @as(c_longlong, @intCast(start)), tr, @as(c_int, @intCast(g.hidden)), d.taps_in });
            } else try e.run(d.ring_fns.gather_taps, .{ rows, 1, 1 }, .{ 256, 1, 1 }, .{ ring, tap_ring_rows, @as(c_longlong, @intCast(start)), @as(c_int, @intCast(g.hidden)), d.taps_in });
            try d.tapNorm(rows);
        }
        try d.layerModeWithInput(ctx, d.ids + id_at * 4, rows, start, .ring, propose_chunk, input_h);
        start += rows;
        n -= rows;
        fi += rows;
        id_at += rows;
        ctx.end = start;
        if (!propose_chunk or n > 0) continue;
        // the newest row: its state kept for the drafts, its tokens drawn at target verify position ctx.end + 1
        try ctx.last.copyFrom(0, d.x + (rows - 1) * g.hidden * 2, g.hidden * 2, e.stream.handle);
        try d.drawSpecs(rows - 1, &.{.{ .sampling = sampling, .position = ctx.end + 1 }}, true);
    }
    if (!propose) return;
    if (d.trees) {
        var tok: [1]u32 = undefined;
        var cands: [1][ranks]u32 = undefined;
        try d.fetchSpecs(&.{.{ .sampling = sampling, .position = ctx.end + 1 }}, &tok, &cands);
        ctx.cands[0] = tok[0];
        ctx.cands = cands[0];
    } else {
        try e.d.check(e.d.api.cuMemcpyDtoDAsync_v2(ctx.held_dev.ptr, d.tok, 4, e.stream.handle), "cuMemcpyDtoDAsync");
        ctx.held_n = 1;
    }
}

test "MTP prompt chunks retain KV without proposing until the final prompt rows" {
    const t = std.testing;
    try t.expectEqual(prompt_kv_tile, nextChunkRows(true, false, 2048));
    try t.expectEqual(@as(usize, 384), nextChunkRows(true, true, 512));
    try t.expectEqual(@as(usize, 60), nextChunkRows(true, true, 188));
    try t.expectEqual(@as(usize, 128), nextChunkRows(true, true, 128));
    try t.expectEqual(@as(usize, 128), nextChunkRows(false, false, 2048));
    try t.expectEqual(@as(usize, 0), nextChunkRows(true, false, 0));
    try t.expect(!proposalNeeded(true, true, 0, 2047, 2048));
    try t.expect(proposalNeeded(true, true, 2047, 1, 2048));
    try t.expect(proposalNeeded(true, true, 1024, 1024, 2048));
    try t.expect(!proposalNeeded(true, false, 2047, 1, 2048));
    try t.expect(proposalNeeded(false, true, 128, 16, 2048));
    try t.expect(!proposalNeeded(true, true, 2048, 0, 2048));
    try t.expect(proposalNeeded(true, true, 70_000, 2, 2));
    try t.expect(proposalNeeded(true, true, 4, 1, 1));
    try t.expect(!batchProposalNeeded(true, true, 1024, 1024, 2048));
    try t.expect(batchProposalNeeded(true, true, 2047, 1, 2048));
    try t.expect(batchProposalNeeded(true, true, 70_000, 1, 2));
    try t.expect(batchProposalNeeded(false, true, 128, 16, 2048));
    try t.expect(!chunkProposalNeeded(true, true, 1024, 2048));
    try t.expect(chunkProposalNeeded(true, true, 1024, 1024));
    try t.expect(chunkProposalNeeded(false, true, 1024, 2048));
    try t.expect(!chunkProposalNeeded(false, false, 1024, 1024));
    const suffix = normalizedRange(4096, 4095, 6143, 2049).?;
    try t.expectEqual(@as(u64, 6143), suffix.start);
    try t.expectEqual(@as(usize, 2047), suffix.skip);
    try t.expectEqual(@as(usize, 2048), suffix.rows);
    try t.expect(normalizedRange(0, 4096, 4096, 0) == null);
    const final_row = normalizedRange(8191, 1, 6143, 2049).?;
    try t.expectEqual(@as(usize, 0), final_row.skip);
    try t.expectEqual(@as(usize, 1), final_row.rows);
}
