//! Lanes Drafter vtable wrapper over SenseNova-Loop-9B native Metal MTP.
const std = @import("std");
const tf = @import("tensorfold");
const q = tf.qwen27;
const lanes = tf.lanes;
const Mtp = q.sn_mtp.Mtp;
const Stream = lanes.Stream;
const dr = lanes.drafter;

const default_prior = [_]f64{ 0.80, 0.62, 0.48, 0.36 };
const max_batch = 16;

fn cast(ptr: *anyopaque) *Mtp {
    return @ptrCast(@alignCast(ptr));
}

fn tapsFn(ptr: *anyopaque) []const u32 {
    return &cast(ptr).tap_layer;
}

fn factsFn(ptr: *anyopaque) dr.Facts {
    const m = cast(ptr);
    return .{
        .limit_shared_drafts = true,
        .depth = m.depth,
        .step_ms = m.step_ms,
        .prior = default_prior[0..@min(m.depth, default_prior.len)],
        .plain_guard = m.plain_guard,
        .batched = true,
        .stream_bytes = m.slotBytes(),
        .copy_first = true,
    };
}

fn openFn(ptr: *anyopaque, s: *Stream) anyerror!void {
    return cast(ptr).openStream(@intFromPtr(s));
}

fn toAbsorbItem(it: dr.Absorb) !q.sn_mtp.AbsorbItem {
    if (it.features.dtype != .bf16) return error.BadMtpFeatures;
    // Skip intermediate prompt-chunk draft proposals (rows strictly before the prompt's last token).
    const prompt_chunk = it.rows == null and it.features.rows > 0 and it.start + it.features.rows < it.stream.prompt_len;
    return .{
        .stream = @intFromPtr(it.stream),
        .features = .{
            .buffer = it.features.buffer,
            .offset = it.features.offset,
            .rows = it.features.rows,
            .row_bytes = it.features.row_bytes,
            .host = it.features.space == .host,
        },
        .start = it.start,
        .follow = it.follow,
        .rows = it.rows,
        .propose = it.propose and !prompt_chunk,
        .prompt_only = prompt_chunk,
    };
}

fn toHoldItem(it: dr.Hold) q.sn_mtp.HoldItem {
    return .{ .stream = @intFromPtr(it.stream), .pending = it.pending, .position = it.position, .depth = it.depth };
}

fn absorbFn(ptr: *anyopaque, items: []const dr.Absorb) anyerror!void {
    const m = cast(ptr);
    var buf: [max_batch]q.sn_mtp.AbsorbItem = undefined;
    var at: usize = 0;
    while (at < items.len) {
        const take = @min(buf.len, items.len - at);
        for (items[at..][0..take], buf[0..take]) |it, *dst| dst.* = try toAbsorbItem(it);
        try m.absorb(buf[0..take]);
        at += take;
    }
}

fn holdFn(ptr: *anyopaque, items: []const dr.Hold) anyerror!void {
    const m = cast(ptr);
    var buf: [max_batch]q.sn_mtp.HoldItem = undefined;
    var at: usize = 0;
    while (at < items.len) {
        const take = @min(buf.len, items.len - at);
        for (items[at..][0..take], buf[0..take]) |it, *dst| dst.* = toHoldItem(it);
        try m.hold(buf[0..take]);
        at += take;
    }
}

fn draftRoundFn(ptr: *anyopaque, absorbs: []const dr.Absorb, holds: []const dr.Hold) anyerror!void {
    if (absorbs.len > max_batch or holds.len > max_batch) {
        if (absorbs.len > 0) try absorbFn(ptr, absorbs);
        if (holds.len > 0) try holdFn(ptr, holds);
        return;
    }
    const m = cast(ptr);
    var ab: [max_batch]q.sn_mtp.AbsorbItem = undefined;
    var hb: [max_batch]q.sn_mtp.HoldItem = undefined;
    for (absorbs, ab[0..absorbs.len]) |it, *dst| dst.* = try toAbsorbItem(it);
    for (holds, hb[0..holds.len]) |it, *dst| dst.* = toHoldItem(it);
    try m.draftRound(ab[0..absorbs.len], hb[0..holds.len]);
}

fn heldFn(ptr: *anyopaque, streams: []const *Stream, outs: []const []u32) anyerror!void {
    const m = cast(ptr);
    var ids: [max_batch]usize = undefined;
    var at: usize = 0;
    while (at < streams.len) {
        const take = @min(ids.len, streams.len - at);
        for (streams[at..][0..take], ids[0..take]) |s, *id| id.* = @intFromPtr(s);
        try m.held(ids[0..take], outs[at..][0..take]);
        at += take;
    }
}

fn releaseFn(ptr: *anyopaque, s: *Stream) void {
    cast(ptr).releaseStream(@intFromPtr(s));
}

const vtable: dr.Drafter.VTable = .{
    .taps = tapsFn,
    .facts = factsFn,
    .open = openFn,
    .absorb = absorbFn,
    .hold = holdFn,
    .draft_round = draftRoundFn,
    .held = heldFn,
    .release = releaseFn,
};

pub fn drafter(m: *Mtp) dr.Drafter {
    return .{
        .kind = "MTP",
        .ptr = m,
        .vtable = &vtable,
    };
}
