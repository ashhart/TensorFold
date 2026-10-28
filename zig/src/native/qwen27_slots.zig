//! The 27B lane backend's slots: a stream's prompt into its slot, slot 0's hand-over and a lone stream's serial loop.
const std = @import("std");
const mtl = @import("metal");
const api = @import("engine_api");
const tf = @import("tensorfold");
const q = tf.qwen27;
const lanes = tf.lanes;
const lb = @import("qwen27_lanes.zig");
const images = @import("qwen27_images.zig");
const resume_rules = @import("qwen27_resume.zig");
const Metal = lb.Metal;
const Ring = q.tap_ring.Ring;
const Context = q.dflash.runtime_model.Context;
const max_rows = lb.max_rows;

/// A prompt pass under way in `slot`: where its next chunk starts, where its restore ended, and its encoded images.
pub const Fill = struct { slot: u32, at: usize, from: usize, images: ?q.image_rows.Images, checkpoint: u32 = 0, rewind: u32 = 0 }; // rewind: slot 0's point

/// A stream's prompt into `slot`, from its kept state: chunks cut as a fresh pass cuts, decoded spans as decoded.
pub fn prefill(b: *Metal, s: *lanes.Stream, slot: u32) !void {
    var f = try opening(b, s, slot);
    defer closing(b, s);
    while (!try chunk(b, s, &f)) {}
}

/// A prompt pass begins in `slot`: its kept state restored and its images encoded (one image prompt fills at a time).
pub fn opening(b: *Metal, s: *lanes.Stream, slot: u32) !Fill {
    const ids = s.prompt();
    if (ids.len == 0 or ids.len + s.max_new + max_rows > b.runner.capacity) return error.PromptTooLong;
    if (s.yields()) return error.Yielded; // a background pass steps aside before it touches the slot
    // as does one over the conversation slot 0 holds when the cache cannot keep its end: it would leave nothing there
    if (slot == 0 and s.yield_check != null and s.max_new == 0 and !keepsEnd(s) and resumable(b, s, 0) != .none) return error.Yielded;
    if (b.vision != null and s.images.len > 0 and b.image_fill != null and b.image_fill != s) return error.LaneMemory;
    try recover(b, s);
    b.runner.rope[slot] = try images.layout(b.vision, s);
    const d = b.drafter(slot);
    const kept_at: u32 = if (s.reuse.saved) |saved| @as(*const q.snapshot.State, @ptrCast(@alignCast(saved))).at else 0;
    const resumed = if (slot == 0) try takeStanding(b, s, kept_at) else 0; // slot 0's own state: nothing copied
    if (resumed == 0) {
        if (d) |x| try x.reset() else try b.runner.reset(slot);
        try clearContext(b, slot);
    }
    s.cached = resumed;
    var at: usize = resumed;
    if (resumed == 0) if (s.reuse.saved) |saved| { // a kept state of this prompt's prefix: the pass starts there
        const st: *q.snapshot.State = @ptrCast(@alignCast(saved));
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
    };
    s.fill_at.store(@intCast(at), .monotonic); // what the restore kept: 0 when it failed
    const encoded = if (b.vision) |v| try images.encode(v, b.runner.model.queue, s, slot) else null;
    if (encoded != null) b.image_fill = s; // the vision rows are this pass's until it closes
    return .{ .slot = slot, .at = at, .from = at, .images = encoded, .checkpoint = resume_rules.cachePoint(slot == 0 and slotResume(b) and plain(s) and s.max_new > 0 and s.reuse.hook != null, at, b.chunk, ids.len, s.history_len, s.reuse.marks, s.chunks, s.decode_spans), .rewind = rewindPoint(b, s, slot, at) };
}

/// The pass is over (done, failed or cancelled): its images' rows are free for the next image prompt.
pub fn closing(b: *Metal, s: *const lanes.Stream) void {
    if (b.image_fill == s) b.image_fill = null;
}

/// With TF_Q27_PROFILE on, the target's passes since `before` count whole toward `s`'s reply (`gpu_seconds`).
pub fn charge(b: *const Metal, s: *lanes.Stream, before: f64) void {
    if (b.runner.trace != null) s.chargeGpu(b.runner.gpu_seconds - before);
}

/// The pass's next chunk, its drafter's absorb before its marks; true once the prompt is in.
pub fn chunk(b: *Metal, s: *lanes.Stream, f: *Fill) !bool {
    const paused = tf.gpu_profile.pause(b.runner.trace); // prompt passes run unprofiled: TF_Q27_PROFILE's rounds decode
    defer paused.end();
    const ids = s.prompt();
    const slot = f.slot;
    const d = b.drafter(slot);
    b.runner.images = f.images;
    defer b.runner.images = null;
    b.runner.settleSlot(slot); // a mark's kept state read the slot: its blit lands before this chunk writes it
    if (s.isCancelled()) return error.Cancelled;
    if (s.yields()) { // real work waits: the cache keeps what this background pass prefilled
        if (f.at > f.from) if (s.reuse.hook) |k| k.at(k.ptr, s, @intCast(f.at));
        if (slot == 0) leave(b, s); // and slot 0 stands at it for the next prompt
        return error.Yielded;
    }
    const at = f.at;
    var end = resume_rules.chunkEnd(at, @import("qwen27_prompt_rows.zig").choose(b, s), ids.len, s.chunks, s.decode_spans);
    if (f.checkpoint > at and f.checkpoint < end) end = f.checkpoint;
    var decoded = false;
    for (s.decode_spans) |span| decoded = decoded or (at >= span[0] and at < span[1]);
    var mark_bank: ?q.gdn_pool_gpu.PromptMarks = null;
    defer {
        if (mark_bank) |*m| {
            b.runner.gdn.marks = null;
            m.deinit();
        }
    }
    const shape = b.runner.gdn.budget.shape;
    if (s.reuse.hook != null and d == null and !decoded and !b.runner.capture_taps and b.runner.images == null and shape.c.dk == 128 and shape.c.dv % 4 == 0) {
        const picked = try q.prompt_marks.select(@intCast(at), @intCast(end), s.reuse.marks);
        if (picked.count > 0) if (b.grow) |g| {
            const budget = b.runner.gdn.budget;
            const size = try q.prompt_marks.layout(budget.layers, shape.state, shape.conv);
            if (g.room(g.ctx, size.total, s)) {
                if (b.runner.gdn.marks != null) return error.RoundNotReady;
                mark_bank = q.gdn_pool_gpu.PromptMarks.init(b.runner.model.device, budget, b.runner.gdn.kernels, slot, @intCast(at), picked.end, picked.rows, b.runner.residency) catch |err| switch (err) {
                    error.NoBuffer, error.UnsupportedGdnPipeline => null,
                    else => return err,
                };
                if (mark_bank) |*m| {
                    end = picked.end;
                    b.runner.gdn.marks = m;
                }
            }
        };
    }
    if (mark_bank == null) for (s.reuse.marks) |mark| if (mark > at and mark < end) {
        end = mark;
    };
    if (mark_bank == null and f.rewind > at and f.rewind < end) end = f.rewind;
    const part = ids[at..end];
    const last = end == ids.len;
    const before = b.runner.gpu_seconds;
    if (d) |x| {
        if (decoded) try x.decodeChunk(part, last) else try x.promptChunk(part, last);
    } else {
        const session = q.session.Session{ .runner = b.runner, .slot = slot };
        if (decoded) try session.decodeChunk(part, last) else try session.promptChunk(part, last);
        if (b.ringOf(slot)) |g| try ringChunk(b, g, part.len);
    }
    charge(b, s, before);
    f.at = end;
    s.fill_at.store(@intCast(end), .monotonic); // a prompt chunk waits: `at` is landed progress
    if (end == f.rewind) saveRewind(b, f.rewind);
    if (s.isCancelled()) return error.Cancelled;
    if (mark_bank) |*m| {
        try m.complete(slot, @intCast(end));
        if (s.reuse.hook) |k| for (m.position.rows) |row| {
            if (s.isCancelled()) return error.Cancelled;
            if (row != 0) k.at(k.ptr, s, m.position.start + row);
        };
    }
    if (end == f.checkpoint or std.mem.indexOfScalar(u32, s.reuse.marks, @intCast(end)) != null) if (s.reuse.hook) |k| k.at(k.ptr, s, @intCast(end));
    if (end == ids.len) b.runner.settleSlot(slot); // the stream's first round writes the slot next
    return end == ids.len;
}

/// `s` in its slot, or a free one taken: its memory ready and its held tree dropped.
pub fn bind(b: *Metal, s: *lanes.Stream) !u32 {
    const slot = b.slotOf(s) orelse (free(b) orelse return error.NoFreeSlot);
    try ready(b, slot, s);
    b.dropTreeOf(slot);
    b.owners[slot] = s;
    return slot;
}

/// One chunk of `s`'s prompt pass a call (the lane host's prefill_step), the others' rounds kept first.
pub fn prefillStep(ptr: *anyopaque, s: *lanes.Stream) anyerror!bool {
    const b: *Metal = @ptrCast(@alignCast(ptr));
    b.settle() catch {};
    b.dropRound();
    promoteDue(b);
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    if (!b.fills.contains(s)) {
        const slot = try bind(b, s);
        errdefer b.owners[slot] = null;
        const f = try opening(b, s, slot);
        errdefer closing(b, s);
        try b.fills.put(b.gpa, s, f);
    }
    const f = b.fills.getPtr(s).?;
    const slot = f.slot;
    const done = chunk(b, s, f) catch |err| {
        endFill(b, s);
        b.owners[slot] = null;
        return err;
    };
    if (done) endFill(b, s);
    return done;
}

/// The host cancels between chunks: the last landed chunk's state is kept while the slot still holds it.
pub fn cancelPrefill(ptr: *anyopaque, s: *lanes.Stream) anyerror!void {
    const b: *Metal = @ptrCast(@alignCast(ptr));
    if (b.fills.get(s)) |f| s.keepInterruptedPrefill(@intCast(f.at));
}

/// A stepped pass is over, or its stream left mid-pass.
pub fn endFill(b: *Metal, s: *lanes.Stream) void {
    if (b.fills.remove(s)) closing(b, s);
}

/// Slot 0 resumes a later prompt from where its last stream left it unless TF_QWEN27_SLOT_RESUME=0 (read at start).
pub fn slotResumeEnv() bool {
    const v = std.c.getenv("TF_QWEN27_SLOT_RESUME") orelse return true;
    return !std.mem.eql(u8, std.mem.span(v), "0");
}

/// Whether slot 0 resumes in place: on, and without --vision (an image prompt's rows go by its images, not its ids).
fn slotResume(b: *const Metal) bool {
    return b.slot_resume and b.vision == null;
}

/// A stream the prompt cache would look up: no images, no prompt logprobs (they start at 0), taught rows.
fn plain(s: *const lanes.Stream) bool {
    return s.images.len == 0 and s.prompt_logprobs == null and !s.untaught and s.embedding == null;
}

/// Whether the cache keeps the state at the end of `s`'s prompt (a background pass's: the cache planned that mark).
fn keepsEnd(s: *const lanes.Stream) bool {
    return std.mem.indexOfScalar(u32, s.reuse.marks, @intCast(s.prompt().len)) != null;
}

/// Slot 0 and its drafter stand at `at`, with no failure or round pending.
fn standsAt(b: *const Metal, at: usize) bool {
    if (b.runner.failed or b.runner.active != null or b.runner.offsets[0] != at) return false;
    const d = b.draft orelse return true;
    return !d.model.backend.failed and d.model.committed_end == at;
}

/// How `s` resumes from what slot 0 stands at, past the cache's restore point `kept_at` (qwen27_resume.zig).
fn resumable(b: *const Metal, s: *const lanes.Stream, kept_at: u32) resume_rules.Resume {
    if (!slotResume(b) or !plain(s) or !standsAt(b, b.standing.tokens.len)) return .none;
    return b.standing.plan(s.prompt(), s.decode_spans, kept_at);
}

/// Slot 0's position for `s` when it resumes there (0: not); the record is used up, as the pass changes the slot.
fn takeStanding(b: *Metal, s: *const lanes.Stream, kept_at: u32) !u32 {
    defer b.standing.deinit(b.gpa);
    switch (resumable(b, s, kept_at)) {
        .none => return 0,
        .here => |n| {
            const d = b.draft orelse return n;
            try d.model.reset(); // a new session over the kept taps, as after a restore
            d.model.committed_end = n;
            return n;
        },
        .back => |h| {
            const buf = b.rewind orelse return 0;
            if (b.draft) |d| try d.model.reset();
            q.snapshot.rewindSlot(b.gpa, b.runner, 0, b.ringOf(0), h, buf) catch return 0; // the pass then starts at 0
            return h;
        },
    }
}

/// A stream leaves slot 0: what it stands at stays for the next prompt that extends it, if a fresh pass stands there.
fn leave(b: *Metal, s: *const lanes.Stream) void {
    const at = b.runner.offsets[0];
    const ended = s.finished and s.reason != .cancelled and s.reason != .@"error";
    if (!slotResume(b) or !plain(s) or !standsAt(b, at)) return b.standing.deinit(b.gpa);
    b.standing.leave(b.gpa, .{ .at = at, .prompt_len = @intCast(s.prompt_len), .context_len = @intCast(s.context.items.len), .fill_at = s.fill_at.load(.monotonic), .ended = ended }, s.context.items, s.decode_spans);
}

/// Where slot 0's pass keeps its rewind point: its rendered history's end, which a next turn shares (0: none).
fn rewindPoint(b: *const Metal, s: *const lanes.Stream, slot: u32, at: usize) u32 {
    if (slot != 0 or b.rewind == null or !slotResume(b) or !plain(s)) return 0;
    return if (s.history_len > at and s.history_len < s.prompt().len) s.history_len else 0;
}

/// Slot 0 stands at its prompt's history end: its DeltaNet states and taps go to the rewind buffer (none if it fails).
fn saveRewind(b: *Metal, at: u32) void {
    b.standing.rewind_at = 0;
    q.snapshot.saveRewind(b.gpa, b.runner, 0, b.ringOf(0), at, b.rewind.?) catch return;
    b.standing.rewind_at = at;
}

/// The rewind buffer, where the prompt cache cannot keep a full-context state (`cache` bytes) and `room` holds it.
pub fn rewindInit(b: *Metal, room: u64, cache: u64) ?mtl.Buffer {
    if (!slotResume(b) or b.runner.ready == 0) return null;
    const need = q.snapshot.rewindBytes(b.runner, b.ringOf(0));
    if (cache >= q.snapshot.slotBytes(b.runner, b.ringOf(0), b.runner.capacity)) return null; // the cache keeps them
    if (need > room) {
        std.log.info("qwen27: no rewind point for slot 0 ({d} MiB, {d} MiB left under the load limit): a next turn that drops the last reply's reasoning reads its prompt again", .{ need >> 20, room >> 20 });
        return null;
    }
    b.rewind = b.runner.model.device.buffer(need, mtl.ResourceOptions.private | mtl.ResourceOptions.untracked) catch return null;
    std.log.info("qwen27: slot 0 keeps a rewind point of {d} MiB at each prompt's history end", .{need >> 20});
    return b.rewind;
}

/// After a GPU failure whose streams have all left, the runner and drafter take rounds again; `s` takes its slot now.
fn recover(b: *Metal, s: *const lanes.Stream) !void {
    const drafter = if (b.draft) |d| d.model.backend.failed else false;
    if (!b.runner.failed and !drafter) return;
    for (b.owners[0..b.runner.slots]) |o| if (o != null and o != s) return error.RoundGpuFailure;
    b.standing.deinit(b.gpa); // the failure may have left slot 0 anywhere
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
pub fn restore(b: *Metal, slot: u32, st: *q.snapshot.State) !void {
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
    if (slot == 0) leave(b, s);
    b.owners[slot] = null;
    b.runner.rope[slot] = .{};
    if (slot == 0) b.promote_due = b.draft != null; // the next backend call moves a stream in, after this reply is out
    var live: usize = 0;
    for (b.owners[0..b.runner.slots]) |owner| live += @intFromBool(owner != null);
    if (live <= 1) b.runner.trimBefore();
}

/// A promote unbind left for later, once no round is pending: a stream still in slot 0 or none to move ends it.
pub fn promoteDue(b: *Metal) void {
    if (!b.promote_due or b.round != null) return;
    b.promote_due = false;
    promote(b);
}

/// Slot 0 freed while other streams decode: the lowest one moves there and drafts from its next round on.
pub fn promote(b: *Metal) void {
    const d = b.draft orelse return;
    if (b.owners[0] != null or b.round != null) return;
    for (1..b.runner.slots) |k| {
        const s = b.owners[k] orelse continue;
        if (s.finished or b.fills.contains(s)) continue; // a prompt pass under way stays in its slot until it is in
        const from: u32 = @intCast(k);
        b.standing.deinit(b.gpa); // slot 0 takes another stream's state
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
    b.standing.deinit(b.gpa);
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
        const before = b.runner.gpu_seconds;
        const result = try g.runRound(room, b.nodes, s.eos, true, if (extra > 0) .{ .ptr = &copies, .call = Copies.call } else null, extra);
        defer b.gpa.free(result.tokens);
        charge(b, s, before);
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
