//! Multi-segment prefill runs, the CED decoder replay and a segment's commit.

const std = @import("std");
const block_prefill = @import("block_prefill.zig");
const longpf = @import("longpf.zig");
const image_opts = @import("image_opts.zig");
const fwd = @import("forward.zig");
const ced = @import("ced.zig");
const Forward = fwd.Forward;
const pf = @import("forward_prefill.zig");
const pf_glue = @import("forward_prefill_glue.zig");
const pf_k4 = @import("forward_prefill_k4.zig");
const Multi = pf.Multi;
const MultiSeg = pf.MultiSeg;
const Tail = pf.Tail;
const copy = pf_glue.copy;
const decoderLayers = pf.decoderLayers;
const encoderLayers = pf.encoderLayers;
const glueFn = pf_glue.glueFn;
const k4Release = pf_k4.k4Release;
const log = pf.log;
const ready = pf_glue.ready;

/// Several slots' prompt segments in one run, each committed as `segment` would; error.Unsupported: one at a time.
pub fn promptMulti(f: *Forward, segs: []const MultiSeg) !void {
    const st = pf.state(f);
    const sw = f.pf_switch orelse return error.Unsupported;
    if (segs.len == 0 or segs.len > 16) return error.BadWindow;
    var sa = std.heap.ArenaAllocator.init(f.gpa);
    defer sa.deinit();
    const a = sa.allocator();
    const replay = st.mode == .replay;
    const layers = if (replay) try encoderLayers(f, try f.backbone(a)) else try f.backbone(a);
    const spans = try a.alloc(block_prefill.Span, segs.len);
    const row0 = try a.alloc(usize, segs.len);
    const tails = try a.alloc(Tail, segs.len);
    var all: std.ArrayList(u32) = .empty;
    for (segs, spans, row0, tails) |sg, *sp, *r0, *t| {
        try sw.run(sw.ctx, sg.slot);
        if (f.slot.pending != null or f.slot.pos != sg.start or sg.ids.len == 0) return error.BadWindow;
        if (sg.start + sg.ids.len > @as(u64, @intCast(f.opts.limit))) return error.BadWindow;
        sp.* = .{ .slot = sg.slot, .start = @intCast(sg.start), .n = @intCast(sg.ids.len) };
        r0.* = all.items.len;
        t.len = f.slot.tail_len;
        @memcpy(t.ids[0..t.len], f.slot.tail[0..t.len]);
        try all.appendSlice(a, sg.ids);
    }
    if (image_opts.segmentOptions(f, all.items).image_rows != 0) return error.Unsupported;
    const cs = try block_prefill.emitMulti(a, f.cfg, f.widths, f.opts, layers, spans, if (replay) .encoder else .whole);
    try ready(f);
    // the pages of every segment's positions, before any launch (each on its slot's table)
    for (segs) |sg| {
        try sw.run(sw.ctx, sg.slot);
        if (f.kv) |kx| try kx.reserve(sg.start + sg.ids.len);
    }
    var m: Multi = .{ .segs = segs, .row0 = row0, .tails = tails, .ids = all.items };
    f.multi = &m;
    defer f.multi = null;
    try enterRun(f, &m);
    st.layers = layers;
    st.moe = 0;
    st.taps_rows = 0;
    f.runner.glue = .{ .ctx = f, .run = glueFn };
    try longpf.window(&st.long, f.gpa, f.runner, f.kv, segs[0].start, cs);
    if (f.runner.prof) |pr| pr.prefillRows(@intCast(all.items.len), false);
    // each slot keeps its segment (carries, Engram tail, position, stash ids)
    for (segs, row0) |sg, r0| {
        try sw.run(sw.ctx, sg.slot);
        f.ids = sg.ids;
        f.slot.pending = .{ .start = sg.start, .n = @intCast(sg.ids.len) };
        try commitAt(f, @intCast(sg.ids.len), r0);
        if (replay) f.slot.ced.extend(sg.start, sg.ids);
    }
    f.last_rows = 0;
}

/// Whether a multi-segment run takes `segs` (promptMulti's refusals), checked before the leader sends it.
pub fn runnable(f: *Forward, segs: []const MultiSeg) !bool {
    if (f.pf_switch == null or segs.len < 2 or segs.len > 16) return false;
    var sa = std.heap.ArenaAllocator.init(f.gpa);
    defer sa.deinit();
    const a = sa.allocator();
    const replay = pf.state(f).mode == .replay;
    const layers = if (replay) try encoderLayers(f, try f.backbone(a)) else try f.backbone(a);
    const spans = try a.alloc(block_prefill.Span, segs.len);
    var all: std.ArrayList(u32) = .empty;
    for (segs, spans) |sg, *sp| {
        sp.* = .{ .slot = sg.slot, .start = @intCast(sg.start), .n = @intCast(sg.ids.len) };
        try all.appendSlice(a, sg.ids);
    }
    if (image_opts.segmentOptions(f, all.items).image_rows != 0) return false;
    _ = block_prefill.emitMulti(a, f.cfg, f.widths, f.opts, layers, spans, if (replay) .encoder else .whole) catch |e| switch (e) {
        error.Unsupported => return false,
        else => return e,
    };
    return true;
}

pub const Replayed = @import("forward_types.zig").Replayed;

/// Whether a batched decoder replay takes `segs`, checked before the leader sends it (emitMulti's refusals).
pub fn replayRunnable(f: *Forward, segs: []const MultiSeg) !bool {
    if (f.pf_switch == null or pf.state(f).mode != .replay or segs.len < 2 or segs.len > 16) return false;
    var sa = std.heap.ArenaAllocator.init(f.gpa);
    defer sa.deinit();
    const a = sa.allocator();
    const spans = try a.alloc(block_prefill.Span, segs.len);
    var all: std.ArrayList(u32) = .empty;
    for (segs, spans) |sg, *sp| {
        sp.* = .{ .slot = sg.slot, .start = @intCast(sg.start), .n = @intCast(sg.ids.len) };
        try all.appendSlice(a, sg.ids);
    }
    if (image_opts.segmentOptions(f, all.items).image_rows != 0) return false;
    _ = block_prefill.emitMulti(a, f.cfg, f.widths, f.opts, try decoderLayers(f, try f.backbone(a)), spans, .decoder) catch |e| switch (e) {
        error.Unsupported => return false,
        else => return e,
    };
    return true;
}

/// Several slots' CED decoder replays in one run; nothing is committed, each tail as its one-slot replay row for row.
pub fn finishMulti(f: *Forward, slots: []const u32, out: []Replayed) !void {
    const st = pf.state(f);
    const sw = f.pf_switch orelse return error.Unsupported;
    if (st.mode != .replay or slots.len < 2 or slots.len > 16 or out.len < slots.len) return error.BadWindow;
    var sa = std.heap.ArenaAllocator.init(f.gpa);
    defer sa.deinit();
    const a = sa.allocator();
    const layers = try decoderLayers(f, try f.backbone(a));
    const segs = try a.alloc(MultiSeg, slots.len);
    const spans = try a.alloc(block_prefill.Span, slots.len);
    const row0 = try a.alloc(usize, slots.len);
    const tails = try a.alloc(Tail, slots.len);
    var all: std.ArrayList(u32) = .empty;
    for (slots, segs, spans, row0, tails) |slot, *sg, *sp, *r0, *t| {
        try sw.run(sw.ctx, slot);
        if (f.slot.pending != null) return error.BadWindow;
        const end = f.slot.pos;
        const tl = ced.tailOf(end);
        if (tl.m == 0) return error.BadWindow;
        if (!f.slot.ced.covers(tl.r0, end)) {
            log.err("CED replay of [{d}, {d}), but the stash holds [{d}, {d})", .{ tl.r0, end, f.slot.ced.lo(), f.slot.ced.hi });
            return error.StashGap;
        }
        const ids = try a.alloc(u32, tl.m);
        f.slot.ced.tail(tl.r0, end, ids);
        sg.* = .{ .slot = slot, .start = tl.r0, .ids = ids };
        sp.* = .{ .slot = slot, .start = @intCast(tl.r0), .n = @intCast(tl.m) };
        r0.* = all.items.len;
        t.len = f.slot.tail_len;
        @memcpy(t.ids[0..t.len], f.slot.tail[0..t.len]);
        try all.appendSlice(a, ids);
    }
    const cs = try block_prefill.emitMulti(a, f.cfg, f.widths, image_opts.segmentOptions(f, all.items), layers, spans, .decoder);
    try ready(f);
    var m: Multi = .{ .segs = segs, .row0 = row0, .tails = tails, .ids = all.items };
    f.multi = &m;
    defer f.multi = null;
    try enterRun(f, &m);
    st.layers = layers;
    st.moe = 0;
    st.taps_rows = 0;
    f.runner.glue = .{ .ctx = f, .run = glueFn };
    try longpf.window(&st.long, f.gpa, f.runner, f.kv, segs[0].start, cs);
    if (f.runner.prof) |pr| pr.prefillRows(@intCast(m.ids.len), true);
    for (segs, row0, out[0..slots.len]) |sg, r0, *o| {
        try sw.run(sw.ctx, sg.slot);
        f.slot.pending = null;
        o.* = .{ .slot = sg.slot, .at = sg.start, .rows = @intCast(sg.ids.len), .row0 = @intCast(r0) };
    }
    f.last_rows = 0;
    try k4Release(f); // TF_DSV41_PF_4K: these prompts are prefilled
    try longpf.release(&st.long, f.runner); // TF_DSV41_STREAM_LAZY: their stream scratch too
}

/// The glue's view of the whole run (Forward.multi): the first segment's slot, every row's ids, the run's rows pending.
pub fn enterRun(f: *Forward, m: *const Multi) !void {
    const sw = f.pf_switch.?;
    try sw.run(sw.ctx, m.segs[0].slot);
    f.ids = m.ids;
    f.slot.pending = .{ .start = m.segs[0].start, .n = @intCast(m.ids.len) };
}

/// CED's decoder replay: layers D .. over the prefilled tail [R0, end) from the stash; nothing is committed.
pub fn finish(f: *Forward) !void {
    const st = pf.state(f);
    if (st.mode != .replay) return;
    if (f.slot.pending != null) return error.BadWindow;
    const end = f.slot.pos;
    const t = ced.tailOf(end);
    if (t.m == 0) return;
    if (!f.slot.ced.covers(t.r0, end)) {
        log.err("CED replay of [{d}, {d}), but the stash holds [{d}, {d})", .{ t.r0, end, f.slot.ced.lo(), f.slot.ced.hi });
        return error.StashGap;
    }
    try ready(f);
    var sa = std.heap.ArenaAllocator.init(f.gpa);
    defer sa.deinit();
    const a = sa.allocator();
    const layers = try decoderLayers(f, try f.backbone(a));
    const ids = st.tail_ids[0..t.m];
    f.slot.ced.tail(t.r0, end, ids);
    // the tail's image positions route with gate.bias_vl (replay._image); no decoder layer reads Engram
    const cs = try block_prefill.emitReplay(a, f.cfg, f.widths, image_opts.segmentOptions(f, ids), layers, t.m, @intCast(t.r0), .decoder);
    f.ids = ids;
    f.slot.pending = .{ .start = t.r0, .n = t.m };
    defer f.slot.pending = null;
    st.layers = layers;
    st.moe = 0;
    f.runner.glue = .{ .ctx = f, .run = glueFn };
    try longpf.window(&st.long, f.gpa, f.runner, f.kv, t.r0, cs);
    if (f.runner.prof) |pr| pr.prefillRows(@intCast(t.m), true);
    f.last_rows = 0;
    // TF_DSV41_PF_4K: the prompt is prefilled, its workspace goes back (TF_DSV41_STREAM_LAZY: and its stream scratch)
    try k4Release(f);
    try longpf.release(&st.long, f.runner);
    // the drafter's context rows: the tail's taps (replay.finish's, which slots.finish_prompt hands to `replayed`)
    st.taps_at = t.r0;
    st.taps_rows = t.m;
}

/// Keep of all the pending segment's rows: the ratio-2 carries, the Engram tail, the position.
pub fn commit(f: *Forward, n: u32) !void {
    return commitAt(f, n, 0);
}

/// `commit` of a segment whose rows are rows [row0, row0 + n) of the run's projections.
pub fn commitAt(f: *Forward, n: u32, row0: usize) !void {
    return commitRoles(f, n, row0, "", n);
}

/// `commitAt` with carries from rows of "w.<pre>L<i>.proj" (TBO's B: "b~") ending at row `n`; the slot advances `adv`.
pub fn commitRoles(f: *Forward, n: u32, row0: usize, comptime pre: []const u8, adv: u32) !void {
    const p = f.slot.pending orelse return error.BadWindow;
    if (f.sess) |x| if (x.reserve) |reserve| try reserve(x.ptr, f.ids[0..adv]);
    const r = f.runner;
    var nb: [64]u8 = undefined;
    var la = std.heap.ArenaAllocator.init(f.gpa);
    defer la.deinit();
    const row: usize = 4 * 2 * @as(usize, f.cfg.head_dim); // fp32 [kv | gate]
    for (try f.backbone(la.allocator())) |L| if (f.cfg.isKvSource(L) and f.cfg.compressRatio(L) == 2) {
        // a role the 4K segment shares with the forward (K4.keep) is under its bare name
        const src = r.addressOf(try std.fmt.bufPrint(&nb, "w." ++ pre ++ "L{d}.proj", .{L})) orelse r.addressOf(try std.fmt.bufPrint(&nb, "w.L{d}.proj", .{L})) orelse return error.Unbound;
        const dst = r.addressOf(try std.fmt.bufPrint(&nb, "s.L{d}.carry", .{L})) orelse return error.Unbound;
        try copy(r, dst, src + (row0 + n - 1) * row, row);
    };
    if (f.engram) |h| tailOf(&f.slot, f.ids[0..adv], h.max_ngram - 1);
    f.slot.pos = p.start + adv;
    f.slot.pending = null;
    if (f.sess) |x| try x.commit(x.ptr, f.ids[0..adv]);
}

/// Engram's tail after committing `ids`: (tail + ids)[-keep:].
pub fn tailOf(s: *fwd.Slot, ids: []const u32, keep: usize) void {
    if (ids.len >= keep) {
        @memcpy(s.tail[0..keep], ids[ids.len - keep ..]);
        s.tail_len = keep;
        return;
    }
    const old = @min(s.tail_len, keep - ids.len); // the old tail's last ones that stay
    std.mem.copyForwards(u32, s.tail[0..old], s.tail[s.tail_len - old .. s.tail_len]);
    @memcpy(s.tail[old .. old + ids.len], ids);
    s.tail_len = old + ids.len;
}
