//! The MTP head on the drafting rank: its forward over (residual streams, next token) rows, and greedy draft chains.
const std = @import("std");
const hip = @import("hip");
const model = @import("model.zig");
const cache = @import("cache.zig");
const forward = @import("forward.zig");
const mixers = @import("mixers.zig");
const lanes = @import("lanes");
const serve = @import("hip_serve");

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

/// The head's layer input: token embeddings and residual `streams` normed, projected, mixed; the first HC read.
const Entry = struct { o: Ops, h: Tensor, x: Tensor, inject: Tensor, mixed: Tensor, dev_pos: u64 };

fn entry(
    ops: Ops,
    m: *const model.Model,
    host: *forward.Host,
    streams: Tensor,
    tokens: []const u32,
    pos: usize,
    ones: hip.DeviceBuffer,
) !Entry {
    const rows = tokens.len;
    // a window's rows (absorbs, chain steps) on the decode tiles; a prompt chunk's on the prefill tiles, as the trunk's
    var o = ops;
    o.window = rows <= cache.max_window;
    o.prefill = !o.window;
    const w = &m.mtp.?;
    const d = w.dims;
    if (rows * d.streams * m.act.size() > ones.len) return error.HeadRowsPastOnes;
    const ids = try host.gpa.alloc(i32, rows);
    defer host.gpa.free(ids);
    const at_pos = try host.gpa.alloc(i32, rows);
    defer host.gpa.free(at_pos);
    for (ids, at_pos, tokens, 0..) |*i, *p, t, r| {
        i.* = @intCast(t);
        p.* = @intCast(pos + r);
    }
    const dev_ids = try host.put(o, std.mem.sliceAsBytes(ids));
    const dev_pos = try host.put(o, std.mem.sliceAsBytes(at_pos));
    const emb = try take(o, m.act, rows * d.hidden);
    try o.embedRows(m.embed, dev_ids, rows, emb);
    const en = try take(o, m.act, rows * d.hidden);
    try o.rms(emb, w.e_norm, en, rows, d.hidden, d.eps);
    const e = try o.project(en, w.fc_e, rows, false);
    const hn = try take(o, m.act, rows * d.wide());
    try o.l.fn_rms_wide(
        @ptrFromInt(streams.ptr),
        @ptrFromInt(w.h_norm),
        @ptrFromInt(hn.ptr),
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
    return .{ .o = o, .h = h, .x = x, .inject = inject, .mixed = mixed, .dev_pos = dev_pos };
}

/// `tokens` (each the token after its row) over residual `streams` from `pos`; `ones` a one a stream a row.
pub fn run(
    ops: Ops,
    m: *const model.Model,
    host: *forward.Host,
    c: *cache.Caches,
    streams: Tensor,
    tokens: []const u32,
    pos: usize,
    ones: hip.DeviceBuffer,
) !Out {
    const rows = tokens.len;
    const w = &m.mtp.?;
    const d = w.dims;
    const l = w.layer;
    const in = try entry(ops, m, host, streams, tokens, pos, ones);
    const o = in.o;
    const seg = [_]forward.Seg{.{ .c = c, .tokens = tokens, .pos = pos }};
    c.pooled = @min(c.pooled, pos / d.index_ratio);
    const branch = try mixers.sparse(o, m, d, l.mixer.sparse, &seg, 0, in.mixed, rows, in.dev_pos, host, null);
    try forward.hcNorm(o, m, in.h, branch, in.inject, in.x, l.mhc.scale, rows);
    const mixed2 = try forward.hcPre(o, m, l.mhc, in.x, rows, in.inject);
    const y = try hip.moe.run(o, m.act, l.moe, mixed2, rows, false);
    try forward.hcNorm(o, m, in.h, y, in.inject, in.x, w.mix.scale, rows);
    c.pooled = (pos + rows) / d.index_ratio;
    return .{ .mixed = try forward.hcPre(o, m, w.mix, in.x, rows, null), .streams = in.h };
}

/// The head's caches over prompt rows whose outputs nothing reads: run's key writes, no attention, HC reads or MoE.
pub fn keys(
    ops: Ops,
    m: *const model.Model,
    host: *forward.Host,
    c: *cache.Caches,
    streams: Tensor,
    tokens: []const u32,
    pos: usize,
    ones: hip.DeviceBuffer,
) !void {
    const w = &m.mtp.?;
    const d = w.dims;
    const in = try entry(ops, m, host, streams, tokens, pos, ones);
    c.pooled = @min(c.pooled, pos / d.index_ratio);
    try mixers.sparseKeys(in.o, m, d, w.layer.mixer.sparse, .{ .c = c, .tokens = tokens, .pos = pos }, 0, in.mixed, in.dev_pos);
    c.pooled = (pos + tokens.len) / d.index_ratio;
}

/// The draft head's greedy token for each of `rows` mixed rows (argmax over the draft vocabulary) and its probability.
pub fn draws(
    o: Ops,
    m: *const model.Model,
    host: *forward.Host,
    s: hip.Stream,
    mixed: Tensor,
    rows: usize,
    ids: []const u32,
    out: []u32,
    probs: []f32,
) !void {
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

/// Candidates a keyed draft downloads at most (top_k + MARGIN); a request asking for more drafts its argmax.
pub const max_keyed = 64;

/// A sampled stream's draft: the target's keyed Gumbel-max over the head's survivors; null if greedy or past max_keyed.
pub fn keyed(
    o: Ops,
    m: *const model.Model,
    host: *forward.Host,
    s: hip.Stream,
    mixed: Tensor,
    ids: []const u32,
    sampling: lanes.Sampling,
    position: u64,
    prob: ?*f32,
) !?u32 {
    if (sampling.temperature <= 0 or sampling.top_k == 0) return null;
    const w = &m.mtp.?;
    const n = w.draft.n;
    const stride: usize = @min(n, @as(usize, sampling.top_k) + serve.sample.MARGIN);
    if (stride > max_keyed) return null;
    const logits = try o.project(mixed, w.draft, 1, false);
    const k: i32 = @intCast(stride);
    const ks = try host.put(o, std.mem.asBytes(&k));
    // argmax, its probability, then the candidates' ids and values: one download
    const dev = try o.arena.take(8 + stride * 6);
    if (prob != null) {
        try o.argmaxRows(logits, 1, n, dev);
        try o.tokenProb(logits, 1, n, dev, dev + 4);
    }
    try o.topkRows(logits, 1, n, ks, stride, dev + 8, dev + 8 + stride * 4);
    const at = std.mem.alignForward(usize, host.used, 16);
    const bytes = host.staging.bytes[at..][0 .. 8 + stride * 6];
    try hip.runtime.check(host.d.api.hipMemcpyDtoHAsync(bytes.ptr, @ptrFromInt(dev), bytes.len, s.handle));
    try s.synchronize();
    const words = std.mem.bytesAsSlice(u32, @as([]align(4) u8, @alignCast(bytes[0 .. 8 + stride * 4])));
    if (prob) |q| q.* = @bitCast(words[1]);
    const vals = std.mem.bytesAsSlice(u16, @as([]align(2) u8, @alignCast(bytes[8 + stride * 4 ..])));
    var values: [max_keyed]f64 = undefined;
    var tokens: [max_keyed]u64 = undefined;
    const dtype: serve.sample.Dtype = if (m.act == .f16) .f16 else .bf16;
    for (0..stride) |j| {
        values[j] = serve.sample.widen(vals, dtype, j);
        tokens[j] = ids[words[2 + j]];
    }
    return @intCast(try lanes.sampling.choose(host.gpa, values[0..stride], tokens[0..stride], position, sampling));
}
