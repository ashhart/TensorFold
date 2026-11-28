//! The MTP head on the drafting rank: its forward over (residual streams, next token) rows, and greedy draft chains.
const std = @import("std");
const hip = @import("hip");
const model = @import("model.zig");
const cache = @import("cache.zig");
const forward = @import("forward.zig");
const mixers = @import("mixers.zig");

const Ops = hip.ops.Ops;
const Tensor = hip.ops.Tensor;
const take = forward.take;

pub const Out = forward.Out;

/// The head's caches: one sparse layer at the whole model's heads.
pub fn caches(
    gpa: std.mem.Allocator,
    d: *const hip.Runtime,
    h: *const model.Head,
    cap: usize,
    act: model.Kind,
    eos: i64,
) !cache.Caches {
    return cache.Caches.init(gpa, d, h.dims, &.{false}, cap, act, eos);
}

/// Staging bytes for the two aligned `i32` uploads `run` makes after the model's forward.
pub fn inputBytes(rows: usize) usize {
    const ids = forward.Host.putEnd(0, rows * @sizeOf(i32));
    return forward.Host.putEnd(ids, rows * @sizeOf(i32));
}

/// The upper stage's limit leaves room for the MTP IDs and positions put after the forward.
pub fn promptUpperLimit(len: usize, head_rows: usize) usize {
    return len -| inputBytes(head_rows);
}

/// `segs`' tokens over `streams` from each segment's `pos`; `prompt`: a chunk's rows, caches only, outputs unread.
pub fn run(
    ops: Ops,
    prompt: bool,
    m: *const model.Model,
    host: *forward.Host,
    segs: []const forward.Seg,
    streams: Tensor,
    ones: hip.DeviceBuffer,
) !Out {
    var o = ops;
    o.window = !prompt;
    o.prefill = prompt;
    const w = &m.mtp.?;
    const d = w.dims;
    var rows: usize = 0;
    for (segs) |g| rows += g.tokens.len;
    if (rows * d.streams * m.act.size() > ones.len) return error.HeadRowsPastOnes;
    const ids = try host.gpa.alloc(i32, rows);
    defer host.gpa.free(ids);
    const at_pos = try host.gpa.alloc(i32, rows);
    defer host.gpa.free(at_pos);
    var at: usize = 0;
    for (segs) |g| {
        for (g.tokens, 0..) |t, r| {
            if (t >= d.vocab) return error.InvalidToken; // embed reads the table at id unchecked
            ids[at + r] = @intCast(t);
            at_pos[at + r] = @intCast(g.pos + r);
        }
        g.c.pooled = @min(g.c.pooled, g.pos / d.index_ratio);
        at += g.tokens.len;
    }
    const dev_ids = try host.put(o, std.mem.sliceAsBytes(ids));
    const dev_pos = try host.put(o, std.mem.sliceAsBytes(at_pos));
    const emb = try take(o, m.act, rows * d.hidden);
    try o.embedRows(m.embed, dev_ids, rows, emb);
    const en = try take(o, m.act, rows * d.hidden);
    try o.rms(emb, w.e_norm, en, rows, d.hidden, d.eps);
    const e = try o.project(en, w.fc_e, rows, false);
    const hn = try take(o, m.act, rows * d.wide());
    const waves = try take(o, .f32, rows * hip.Launcher.fn_wide_waves);
    try o.l.fn_rms_wide(
        @ptrFromInt(streams.ptr),
        @ptrFromInt(w.h_norm),
        @ptrFromInt(hn.ptr),
        @ptrFromInt(waves.ptr),
        @backingInt(m.act),
        @intCast(rows),
        @intCast(d.wide()),
        d.eps,
        o.stream,
    );
    const h = try o.project(hn, w.fc_h, rows * d.streams, false);
    const x = try take(o, m.act, rows * d.wide());
    const l = w.layer;
    // the embedding broadcast over the streams: h += round(e * 1) per stream, then the layer's first norm
    try forward.hcNorm(o, m, h, e, .{ .ptr = ones.base(), .kind = m.act }, x, l.ahc.scale, rows);
    const inject = try take(o, m.act, rows * d.streams);
    const mixed = try forward.hcPre(o, m, l.ahc, x, rows, inject);
    if (prompt) {
        // a prompt's rows: only the layer's caches are read later (its first draw runs the last row as a window)
        try mixers.sparseKeys(o, m, d, l.mixer.sparse, segs, 0, mixed, rows, dev_pos);
        for (segs) |g| g.c.pooled = (g.pos + g.tokens.len) / d.index_ratio;
        return .{ .mixed = x, .streams = h };
    }
    const branch = try mixers.sparse(o, m, d, l.mixer.sparse, segs, 0, mixed, rows, dev_pos, host, null);
    try forward.hcNorm(o, m, h, branch, inject, x, l.mhc.scale, rows);
    const mixed2 = try forward.hcPre(o, m, l.mhc, x, rows, inject);
    const y = try hip.moe.run(o, m.act, l.moe, mixed2, rows, prompt);
    try forward.hcNorm(o, m, h, y, inject, x, w.mix.scale, rows);
    for (segs) |g| g.c.pooled = (g.pos + g.tokens.len) / d.index_ratio;
    return .{ .mixed = try forward.hcPre(o, m, w.mix, x, rows, null), .streams = h };
}

/// Each mixed row's greedy draft token and probability; `window` for a shared round's rows, else a prompt's.
pub fn draws(
    ops: Ops,
    m: *const model.Model,
    host: *forward.Host,
    s: hip.Stream,
    mixed: Tensor,
    rows: usize,
    ids: []const u32,
    out: []u32,
    probs: []f32,
    window: bool,
) !void {
    var o = ops;
    o.window = window;
    const w = &m.mtp.?;
    const logits = try o.project(mixed, w.draft, rows, false);
    const arg = try o.arena.of(i32, 2 * rows);
    try o.argmaxRows(logits, rows, w.draft.n, arg);
    try o.tokenProb(logits, rows, w.draft.n, arg, arg + rows * 4);
    const at = std.mem.alignForward(usize, host.used, 16);
    const bytes = host.staging.bytes[at..][0 .. 2 * rows * 4];
    try hip.runtime.check(host.d.api.hipMemcpyDtoHAsync(bytes.ptr, @ptrFromInt(arg), 2 * rows * 4, s.handle));
    try s.synchronize();
    const words = std.mem.bytesAsSlice(u32, @as([]align(4) u8, @alignCast(bytes)));
    for (out[0..rows], probs[0..rows], words[0..rows], words[rows..]) |*t, *p, i, f| {
        t.* = ids[i];
        p.* = @bitCast(f);
    }
}

test "Flash Next prompt staging: upper-half acceptance reserves aligned MTP inputs before offset-zero fallback" {
    const staging_bytes = 64 << 20;
    const half = staging_bytes / 2;
    const rows = 129; // each 516-byte input puts the next one across a 16-byte boundary
    const head = inputBytes(rows);
    try std.testing.expectEqual(@as(usize, 528 + 516), head);
    try std.testing.expect(head < half);
    const limit = promptUpperLimit(staging_bytes, rows);
    try std.testing.expectEqual(staging_bytes - head, limit);

    // Accepted: the forward's S bytes and both head uploads exactly fill the upper half.
    const accepted = half - head;
    try std.testing.expect(half + accepted <= limit);
    try std.testing.expect(accepted + head == half);

    // Supplied accepted this upper stage (`half + S <= len`), then the head puts crossed `len`.
    const fallback = accepted + 16;
    try std.testing.expect(fallback <= half);
    try std.testing.expect(fallback + head > half);
    try std.testing.expect(half + fallback <= staging_bytes);
    try std.testing.expect(half + fallback + head > staging_bytes);
    // Fixed admission falls back to offset zero, where both the stage and head uploads fit.
    try std.testing.expect(half + fallback > limit);
    try std.testing.expect(fallback <= staging_bytes);
    try std.testing.expect(fallback + head <= staging_bytes);
    try std.testing.expect(fallback + head < staging_bytes);

    // Offset zero still accepts exactly full staging, but nothing genuinely past it.
    const full = staging_bytes - head;
    try std.testing.expect(full + head == staging_bytes);
    try std.testing.expect(full + 1 + head > staging_bytes);
    try std.testing.expect(promptUpperLimit(staging_bytes, 0) == staging_bytes);
}
