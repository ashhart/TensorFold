//! The 27B lane backend's slots: a stream's prompt into its slot, slot 0's hand-over and a lone stream's serial loop.
const std = @import("std");
const mtl = @import("metal");
const api = @import("engine_api");
const tf = @import("tensorfold");
const q = tf.qwen27;
const lanes = tf.lanes;
const lb = @import("qwen27_lanes.zig");
const images = @import("qwen27_images.zig");
const Metal = lb.Metal;
const Ring = q.tap_ring.Ring;
const Context = q.dflash.runtime_model.Context;
const max_rows = lb.max_rows;

/// A stream's prompt into `slot`, from its kept state: chunks cut as a fresh pass cuts, decoded spans as decoded.
pub fn prefill(b: *Metal, s: *lanes.Stream, slot: u32) !void {
    const ids = s.prompt();
    if (ids.len == 0 or ids.len + s.max_new + max_rows > b.runner.capacity) return error.PromptTooLong;
    if (s.yields()) return error.Yielded; // a background pass steps aside before it touches the slot
    try recover(b, s);
    b.runner.rope[slot] = try images.layout(b.vision, s);
    const d = b.drafter(slot);
    if (d) |x| try x.reset() else try b.runner.reset(slot);
    try clearContext(b, slot);
    s.cached = 0;
    var at: usize = 0;
    if (s.reuse.saved) |saved| { // a kept state of this prompt's prefix: the pass starts there
        const st: *const q.snapshot.State = @ptrCast(@alignCast(saved));
        if (st.at < ids.len) {
            if (restore(b, slot, st)) {
                at = st.at;
                s.cached = st.at;
            } else |_| {
                s.reuse_failed = true;
                if (d) |x| try x.reset() else try b.runner.reset(slot);
                try clearContext(b, slot);
            }
        } else s.reuse_failed = true;
    }
    const from = at;
    s.fill_at.store(@intCast(at), .monotonic); // what the restore kept: 0 when it failed
    b.runner.images = if (b.vision) |v| try images.encode(v, b.runner.model.queue, s, slot) else null;
    defer b.runner.images = null;
    while (at < ids.len) {
        if (s.isCancelled()) return error.Cancelled;
        if (s.yields()) { // real work waits: the cache keeps what this background pass prefilled
            if (at > from) if (s.reuse.hook) |k| k.at(k.ptr, s, @intCast(at));
            return error.Yielded;
        }
        var end = @min(ids.len, at + b.chunk);
        for (s.chunks) |cut| if (cut > at and cut < end) {
            end = cut;
        };
        for (s.reuse.marks) |mark| if (mark > at and mark < end) {
            end = mark;
        };
        var decoded = false;
        for (s.decode_spans) |span| {
            for (span) |cut| if (cut > at and cut < end) {
                end = cut;
            };
            decoded = decoded or (at >= span[0] and at < span[1]);
        }
        const part = ids[at..end];
        const last = end == ids.len;
        if (d) |x| {
            if (decoded) try x.decodeChunk(part, last) else try x.promptChunk(part, last);
        } else {
            const session = q.session.Session{ .runner = b.runner, .slot = slot };
            if (decoded) try session.decodeChunk(part, last) else try session.promptChunk(part, last);
            if (b.ringOf(slot)) |g| try ringChunk(b, g, part.len);
        }
        at = end;
        s.fill_at.store(@intCast(at), .monotonic); // a prompt chunk waits: `at` is landed progress
        if (std.mem.indexOfScalar(u32, s.reuse.marks, @intCast(at)) != null) if (s.reuse.hook) |k| k.at(k.ptr, s, @intCast(at));
    }
}

/// After a GPU failure whose streams have all left, the runner and drafter take rounds again; `s` takes its slot now.
fn recover(b: *Metal, s: *const lanes.Stream) !void {
    const drafter = if (b.draft) |d| d.model.backend.failed else false;
    if (!b.runner.failed and !drafter) return;
    for (b.owners[0..b.runner.slots]) |o| if (o != null and o != s) return error.RoundGpuFailure;
    b.dropRound();
    b.dropTree();
    b.runner.recover();
    if (b.draft) |d| try d.model.recover();
    for (1..b.runner.slots) |slot| try clearContext(b, @intCast(slot));
    std.log.info("qwen27: rounds run again after a GPU failure", .{});
}

/// Empty `slot`'s own drafter state past slot 0 (its taps ring and session), as a fresh prompt starts it.
fn clearContext(b: *Metal, slot: u32) !void {
    if (slot == 0) return;
    const d = b.draft orelse return;
    if (b.contexts[slot]) |x| try x.reset(d.model.graph.config);
}

/// A prompt chunk's `rows` taps into a slot's ring.
pub fn ringChunk(b: *Metal, g: Ring, rows: usize) !void {
    try @import("qwen27_prompt_taps.zig").absorb(TapGather{ .b = b, .ring = g }, rows);
}

const TapGather = struct {
    b: *Metal,
    ring: Ring,
    pub fn gather(ctx: TapGather, first: usize, rows: usize) !void {
        const b = ctx.b;
        const r = b.runner;
        if (rows > b.kept.length() / 4 or rows * ctx.ring.stride > b.gathered_taps.?.length()) return error.ShortTapScratch;
        const list = b.kept.slice(u32, rows);
        for (list, 0..) |*x, i| x.* = @intCast(first + i);
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        const cb = r.model.queue.commandBuffer();
        const e = cb.compute(.serial);
        try r.taps.accepted(r.model.glue, e, list, .{ .buffer = b.kept }, .{ .buffer = b.gathered_taps.? });
        e.end();
        cb.commit();
        cb.wait();
        if (q.decode_round.gpuFailed(cb, "a prompt chunk's taps")) return error.DraftTapGpuFailure;
        try ctx.ring.absorb(b.gathered_taps.?, 0, @intCast(rows));
    }
};

/// `slot` becomes a kept state: the drafter (slot 0) rebuilds its context from the restored taps.
pub fn restore(b: *Metal, slot: u32, st: *const q.snapshot.State) !void {
    if (b.drafter(slot)) |d| try d.model.reset();
    try clearContext(b, slot);
    try q.snapshot.restoreSlot(b.gpa, b.runner, slot, b.ringOf(slot), st);
}

/// A slot's K/V rows and drafter context, taken on its first use by `s`; error.LaneMemory when they do not fit yet.
pub fn ready(b: *Metal, slot: u32, s: *const lanes.Stream) !void {
    const kv = slot >= b.runner.ready;
    const context_missing = b.draft != null and slot > 0 and b.contexts[slot] == null;
    if (!kv and !context_missing) return;
    const drafting = slot < b.draft_slots;
    const context_bytes: usize = if (b.draft) |d| contextBytes(d.model.graph.config, drafting) else 0;
    const need = (if (kv) b.runner.slotBytes() else 0) + (if (context_missing) context_bytes else 0);
    if (b.grow) |g| if (!g.room(g.ctx, need, s)) return error.LaneMemory;
    var added: std.ArrayList(mtl.Buffer) = .empty;
    defer added.deinit(b.gpa);
    try added.appendSlice(b.gpa, try b.runner.ensure(slot));
    if (context_missing) {
        const x = try Context.init(b.gpa, b.draft.?.model, drafting);
        b.contexts[slot] = x;
        try x.buffers(&added, b.gpa);
    }
    if (b.grow) |g| g.added(g.ctx, added.items);
}

/// A drafter context's bytes: the taps ring, and K/V caches when the slot drafts.
fn contextBytes(c: q.dflash.config.Config, drafting: bool) usize {
    const ring = @as(usize, c.window) * c.tapWidth() * 2;
    return ring + if (drafting) @as(usize, c.layers) * 2 * (@as(usize, c.window) + 16) * c.kvWidth() * 2 else 0;
}

/// A free slot for a new stream, slot 0 (the drafter's) first.
pub fn free(b: *const Metal) ?u32 {
    for (b.owners[0..b.runner.slots], 0..) |o, i| if (o == null) return @intCast(i);
    return null;
}

/// The drafter absorbs nothing more for a stream that is gone; its slot is free once its state is kept.
pub fn unbind(b: *Metal, s: *lanes.Stream) void {
    const slot = b.slotOf(s) orelse return;
    b.settle() catch {};
    b.dropRound();
    b.dropTreeOf(slot);
    if (b.release_hook) |hook| if (s.finished and s.reason != .cancelled and s.reason != .@"error") hook.kept(hook.ctx, s, slot);
    b.owners[slot] = null;
    b.runner.rope[slot] = .{};
    if (slot == 0) promote(b);
}

/// Slot 0 freed while other streams decode: the lowest one moves there and drafts from its next round on.
pub fn promote(b: *Metal) void {
    const d = b.draft orelse return;
    if (b.owners[0] != null or b.round != null) return;
    for (1..b.runner.slots) |k| {
        const s = b.owners[k] orelse continue;
        if (s.finished) continue;
        const from: u32 = @intCast(k);
        d.model.reset() catch return;
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        q.snapshot.moveSlot(b.gpa, b.runner, from, b.ringOf(from), 0, d.model.ring()) catch |err| {
            std.log.warn("qwen27: a stream stays in slot {d}, undrafted ({s})", .{ k, @errorName(err) });
            return;
        };
        b.owners[0] = s;
        b.owners[k] = null;
        b.runner.rope[0] = b.runner.rope[k];
        b.runner.rope[k] = .{};
        clearContext(b, from) catch {};
        return;
    }
}

/// Slot 0 emptied and prefilled with `prompt`, through the drafter when there is one (the window timing's context).
pub fn refill(b: *Metal, prompt: []const u32) !void {
    b.runner.rope[0] = .{};
    if (b.draft) |d| try d.reset() else try b.runner.reset(0);
    if (b.draft) |d| try d.generation.prefill(prompt, 128) else try (q.session.Session{ .runner = b.runner }).prefill(prompt, 128);
}

/// A lone stream's copies: what followed the latest earlier match of its context's end.
const Copies = struct {
    lookup: lanes.SuffixLookup,
    history: []const u32 = &.{},
    fn call(ptr: *anyopaque, path: []const u32, max: u32) anyerror![]const u32 {
        const c: *Copies = @ptrCast(@alignCast(ptr));
        return c.lookup.proposeAfter(c.history, path[1..], max); // path[0], the pending token, already ends the history
    }
};

/// Copied rows a lone round may add past its drafted tree, at most (TF_Q27_COPY_ROWS overrides; 0: none).
fn copyRows() u32 {
    const v = std.c.getenv("TF_Q27_COPY_ROWS") orelse return 48;
    return std.fmt.parseInt(u32, std.mem.span(v), 10) catch 48;
}

/// A lone greedy stream on the DFlash2 serial loop in slot 0 until it ends (false) or the lane core adopts it (true).
pub fn lone(b: *Metal, s: *lanes.Stream, hooks: api.LoneHooks) !bool {
    const d = b.draft orelse return error.DraftNotAttached;
    if (s.sampling != null) return error.GreedyDraftsOnly;
    if (b.owners[0] != null) return error.StreamBusy;
    b.owners[0] = s;
    var handed = false;
    defer if (!handed) unbind(b, s);
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    try d.generation.settle();
    defer d.generation.settle() catch {}; // the last round's keep ends before unbind keeps the slot's state
    try prefill(b, s, 0);
    if (s.isCancelled()) return error.Cancelled;
    const g = &d.generation;
    const cap = copyRows();
    var extra: u32 = @min(cap, 16); // grows while copies land whole, shrinks when most miss
    var copies = Copies{ .lookup = try lanes.SuffixLookup.init(b.gpa, .{ .min_match = 5 }) };
    defer copies.lookup.deinit();
    _ = try s.commit(b.gpa, &.{try (q.session.Session{ .runner = b.runner }).greedy()}, &.{});
    hooks.committed(hooks.ctx);
    while (!s.finished) {
        if (hooks.yield(hooks.ctx)) {
            s.cache_len = b.runner.offsets[0];
            s.pending = s.context.items[s.context.items.len - 1];
            handed = true;
            return true;
        }
        const room: u32 = @intCast(s.max_new - s.emitted().len);
        copies.history = s.context.items;
        const result = try g.runRound(room, b.nodes, s.eos, true, if (extra > 0) .{ .ptr = &copies, .call = Copies.call } else null, extra);
        defer b.gpa.free(result.tokens);
        if (result.extended > 0) {
            try copies.lookup.observe(result.extended, result.extension_landed);
            if (result.extension_landed == result.extended) extra = @min(cap, extra * 2) else if (2 * result.extension_landed < result.extended) extra = @max(@min(cap, 16), extra / 2);
        }
        if (result.tokens.len == 0) break;
        s.rounds += @intCast(result.rounds);
        s.drafted += @intCast(result.verified - result.rounds);
        s.accepted += @intCast(result.matched);
        if (result.min_rows > 0) s.min_rows = if (s.min_rows == 0) result.min_rows else @min(s.min_rows, result.min_rows);
        _ = try s.commit(b.gpa, result.tokens, &.{});
        hooks.committed(hooks.ctx);
    }
    if (!s.finished) {
        s.finished = true;
        s.reason = .stop;
    }
    return false;
}
