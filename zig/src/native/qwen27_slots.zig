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
    if (b.ringOf(slot)) |g| g.end.* = 0;
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
                if (b.ringOf(slot)) |g| g.end.* = 0;
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
    std.log.info("qwen27: rounds run again after a GPU failure", .{});
}

/// A prompt chunk's `rows` taps into a slot's ring.
pub fn ringChunk(b: *Metal, g: Ring, rows: usize) !void {
    const r = b.runner;
    const list = b.kept.slice(u32, rows);
    for (list, 0..) |*x, i| x.* = @intCast(i);
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const cb = r.model.queue.commandBuffer();
    const e = cb.compute(.serial);
    try r.taps.accepted(r.model.glue, e, list, .{ .buffer = b.kept }, .{ .buffer = b.gathered_taps.? });
    e.end();
    cb.commit();
    cb.wait();
    if (q.decode_round.gpuFailed(cb, "a prompt chunk's taps")) return error.DraftTapGpuFailure;
    try g.absorb(b.gathered_taps.?, 0, @intCast(rows));
}

/// `slot` becomes a kept state: the drafter (slot 0) rebuilds its context from the restored taps.
pub fn restore(b: *Metal, slot: u32, st: *const q.snapshot.State) !void {
    if (b.drafter(slot)) |d| try d.model.reset();
    try q.snapshot.restoreSlot(b.gpa, b.runner, slot, b.ringOf(slot), st);
}

/// A slot's K/V rows and taps ring, taken on its first use by `s`; error.LaneMemory when they do not fit yet.
pub fn ready(b: *Metal, slot: u32, s: *const lanes.Stream) !void {
    const kv = slot >= b.runner.ready;
    const ring_missing = b.draft != null and slot > 0 and b.rings[slot] == null;
    if (!kv and !ring_missing) return;
    const ring_bytes: usize = if (b.draft) |d| @as(usize, d.model.ring().window) * d.model.ring().stride else 0;
    const need = (if (kv) b.runner.slotBytes() else 0) + (if (ring_missing) ring_bytes else 0);
    if (b.grow) |g| if (!g.room(g.ctx, need, s)) return error.LaneMemory;
    var added: std.ArrayList(mtl.Buffer) = .empty;
    defer added.deinit(b.gpa);
    try added.appendSlice(b.gpa, try b.runner.ensure(slot));
    if (ring_missing) {
        const buf = try b.runner.model.device.buffer(ring_bytes, lb.opts);
        b.rings[slot] = .{ .buf = buf };
        try added.append(b.gpa, buf);
    }
    if (b.grow) |g| g.added(g.ctx, added.items);
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
    if (slot == 0) b.dropTree();
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
        b.rings[k].?.end = 0;
        return;
    }
}

/// Slot 0 emptied and prefilled with `prompt`, through the drafter when there is one (the window timing's context).
pub fn refill(b: *Metal, prompt: []const u32) !void {
    b.runner.rope[0] = .{};
    if (b.draft) |d| try d.reset() else try b.runner.reset(0);
    if (b.draft) |d| try d.generation.prefill(prompt, 128) else try (q.session.Session{ .runner = b.runner }).prefill(prompt, 128);
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
    try prefill(b, s, 0);
    if (s.isCancelled()) return error.Cancelled;
    const g = &d.generation;
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
        const result = try g.runRound(room, b.nodes, s.eos, true);
        defer b.gpa.free(result.tokens);
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
