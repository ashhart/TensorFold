//! Engine.verify/verifyMasked/verifyShared: each window's forward reads and the multi-stream shared pass.
const std = @import("std");
const state = @import("cuda_state.zig");
const Dump = @import("cuda_dump.zig").Dump;
const Mask = @import("cuda_forward.zig").Mask;
const Seg = @import("cuda_forward.zig").Seg;
const engine_mod = @import("cuda_engine.zig");
const Engine = engine_mod.Engine;
const Shared = engine_mod.Shared;
const max_streams = engine_mod.max_streams;
const pin_meta = engine_mod.pin_meta;
const pin_ids = engine_mod.pin_ids;
const pin_sampled = engine_mod.pin_sampled;
const pin_shared = engine_mod.pin_shared;
const window_words = engine_mod.window_words;
const masked_ids = engine_mod.masked_ids;
const draw_rules_words = engine_mod.draw_rules_words;
const draw_meta_words = engine_mod.draw_meta_words;
const pinDraw = engine_mod.pinDraw;

/// Engine.forward: a window of `rows` rows at pos, its first `ids.len` inputs from the host (drafts on the device).
pub fn verify(e: *Engine, ids: []const u32, rows: usize, dump: ?*Dump) !void {
    return e.verifyMasked(ids, rows, dump, .{});
}

/// verify with a grammar's bits for each row: its logits masked before its draws, a graph each row count once made.
pub fn verifyMasked(e: *Engine, ids: []const u32, rows: usize, dump: ?*Dump, mask: Mask) !void {
    if (rows < 1 or rows > state.windowRows(e.c) or ids.len > rows) return error.BadWindow;
    if (e.pos + rows > e.cap()) return error.ContextFull;
    try e.copied.synchronize();
    const host = e.pinned.slice(u32);
    host[pin_meta..][0..4].* = e.metaRow();
    @memcpy(host[pin_ids..][0..ids.len], ids);
    try e.ops().upload(e.b.ids, std.mem.sliceAsBytes(host[pin_ids..][0..ids.len]));
    try e.ops().upload(e.b.meta, std.mem.sliceAsBytes(host[pin_meta..][0..4]));
    try e.copied.record(e.stream);
    const m: usize = @intFromBool(e.sampling != null);
    if (dump == null and mask.bits == 0 and e.windows[m][rows] != null) {
        try e.launchGraph(e.windows[m][rows].?, @intCast(1 + m * (state.max_rows + 1) + rows), .{ .window = .{ .rows = rows, .feed = false } });
    } else if (dump == null and e.windows[m][rows] != null and maskGraphs(e, mask)) {
        try e.launchCaptured(@intCast(masked_ids + m * (state.max_rows + 1) + rows), .{ .window = .{ .rows = rows, .feed = false, .mask = mask } });
    } else {
        e.stats.eager += 1;
        const f = e.forward(dump);
        try f.windowSegs(&.{.{ .b = f.b, .row0 = 0, .rows = rows, .sampled = f.sampled, .mask = mask, .slot = f.slot, .untaught = f.untaught }});
        try e.ops().download(std.mem.sliceAsBytes(host[pin_sampled..][0..rows]), e.b.sampled);
    }
    try e.sampled_ready.record(e.stream);
    e.parity ^= 1;
    e.rows = rows;
}

/// Whether masked windows replay graphs: their bits stay where the first one read them (the host rewrites them).
pub fn maskGraphs(e: *Engine, mask: Mask) bool {
    if (e.mask_bits == 0) e.mask_bits = mask.bits;
    return mask.bits == e.mask_bits;
}

/// Several streams' windows in one forward (eager): each stream's rows read its own caches, state and rule.
pub fn verifyShared(e: *Engine, parts: []const Shared) !void {
    if (parts.len < 1 or parts.len > @import("cuda_wide.zig").streams(e.c)) return error.BadWindow;
    var total: usize = 0;
    for (parts) |p| {
        // a stream's rows stride at windowRows in its own buffers (ids, hidden, Mamba rows), however wide the round
        if (p.rows < 1 or p.rows > state.windowRows(e.c) or p.ids.len < 1 or p.ids.len > p.rows) return error.BadWindow;
        total += p.rows;
    }
    if (total > state.sharedRows(e.c)) return error.WindowTooWide;
    try e.copied.synchronize();
    const host = e.pinned.slice(u32)[pin_shared..][0 .. parts.len * window_words];
    var views: [max_streams]state.Buffers = undefined;
    var parts_f: [max_streams]Seg = undefined;
    var row0: usize = 0;
    for (parts, 0..) |p, k| {
        e.bind(p.seq);
        if (e.pos + p.rows > e.cap()) return error.ContextFull;
        const slot = host[k * window_words ..][0..window_words];
        slot[pin_meta..][0..4].* = e.metaRow();
        @memcpy(slot[pin_ids..][0..p.ids.len], p.ids);
        try e.ops().upload(e.b.ids, std.mem.sliceAsBytes(slot[pin_ids..][0..p.ids.len]));
        try e.ops().upload(e.b.meta, std.mem.sliceAsBytes(slot[pin_meta..][0..4]));
        views[k] = e.b; // the bound sequence's own fields beside the shared scratch
        parts_f[k] = .{ .b = &views[k], .row0 = row0, .rows = p.rows, .sampled = e.sampling != null, .mask = p.mask, .slot = p.seq.slot, .untaught = p.seq.untaught };
        row0 += p.rows;
    }
    var sd: [max_streams]@import("cuda_shared_draw.zig").SharedDraw = undefined;
    var draw_row: usize = 0;
    for (parts, 0..) |p, k| {
        sd[k] = .{ .sampling = p.seq.sampling, .pos = p.seq.pos, .rows = p.rows };
        e.draw_at[k] = @intCast(draw_row);
        draw_row += p.rows;
    }
    var batch = if (parts.len > 1) @import("cuda_shared_draw.zig").batch(sd[0..parts.len], e.pinned, e.b.draw_rules, e.b.draw_meta, e.b.draw_tokens, pinDraw(e.c), pinDraw(e.c) + draw_rules_words) else null;
    if (batch) |b| try @import("cuda_shared_draw.zig").upload(b, e.ops());
    try e.copied.record(e.stream);
    var f = e.forward(null);
    f.batch = if (batch) |*b| b else null;
    try f.windowSegs(parts_f[0..parts.len]);
    const slab = e.pinned.slice(u32)[pinDraw(e.c) + draw_rules_words + draw_meta_words ..][0..draw_row];
    if (batch != null) {
        try e.ops().download(std.mem.sliceAsBytes(slab), e.b.draw_tokens);
    } else for (parts, 0..) |p, k| {
        try e.ops().download(std.mem.sliceAsBytes(slab[e.draw_at[k]..][0..p.rows]), views[k].sampled);
    }
    try e.sampled_ready.record(e.stream);
    for (parts) |p| {
        e.bind(p.seq);
        e.parity ^= 1;
        e.rows = p.rows;
    }
}

/// Part k's drawn ids from the last shared window (row r: the token at its position + r + 1).
pub fn sharedTokens(e: *Engine, k: usize, rows: usize) ![]const u32 {
    try e.sampled_ready.synchronize();
    return e.pinned.slice(u32)[pinDraw(e.c) + draw_rules_words + draw_meta_words + e.draw_at[k] ..][0..rows];
}
