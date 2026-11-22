//! Nemotron CUDA lanes' prompt pass: whole (prefill) or a chunk a call between other streams' rounds (prefill_step).
const std = @import("std");
const cuda = @import("cuda");
const lanes = @import("lanes");
const engine = @import("cuda_engine.zig");
const state = @import("cuda_state.zig");
const pass = @import("cuda_prompt.zig");
const prompt_rows = @import("cuda_prompt_rows.zig");
const lesson_route = @import("living").lesson_route;
const heat = @import("heat");
const Cuda = @import("cuda_lanes.zig").Cuda;
const LogRow = lanes.LogprobRow;
const row_words = lanes.logprob.words;
/// Host-mapped words for every row a shared window holds, then the first token's row (cuda_lanes.zig's row_slots).
const row_slots = engine.max_streams * state.max_rows + 1;

/// Stepped passes between their chunks, by stream: each one's serial pass (cuda_prompt.zig's Pass).
pub const Fills = std.AutoHashMapUnmanaged(*const lanes.Stream, pass.Pass);

/// prefill: the stream's sequence, sampling and kept prefix, then its whole pass.
pub fn whole(self: *Cuda, s: *lanes.Stream) !void {
    if (s.isCancelled()) return error.Cancelled; // cancelled while it waited: no sequence, restore or tower
    const from = try open(self, s);
    try run(self, s, from);
}

/// A new sequence for the stream and its sampling, its kept prefix restored: the row its pass starts at.
fn open(self: *Cuda, s: *lanes.Stream) !usize {
    const e = self.e;
    const ids = s.prompt();
    if (ids.len == 0 or ids.len + s.max_new + state.max_rows > e.max_len) return error.PromptTooLong;
    const gop = try self.lanes.getOrPut(self.gpa, s);
    if (gop.found_existing) self.drop(gop.value_ptr.*);
    // the engine's own sequence while it is free: its rounds replay the captured graphs
    const own = self.own_free and e.serial != null and !s.untaught and !self.paged_only; // the own graphs carry the pocket
    const seq = if (own) &e.own else if (self.pages) |p| p.open(ids.len, s.max_new, s.untaught) else e.newSeq();
    gop.value_ptr.* = .{
        .own = own,
        .paged = !own and self.pages != null,
        .stats = e.stats,
        .seq = seq catch |err| {
            self.lanes.removeByPtr(gop.key_ptr); // nothing held: a LaneMemory request waits for a stream to end
            return err;
        },
    };
    if (own) self.own_free = false;
    gop.value_ptr.seq.untaught = s.untaught;
    e.bind(gop.value_ptr.seq);
    // a decision row only where lessons route on it: else the prompt's chunks stay as plain serving cuts them
    e.decide = if (s.decide != null and e.lessons() != null) s.decide.? else lesson_route.none;
    try e.setSampling(s.sampling);
    s.cached = 0;
    const from = if (s.prompt_logprobs != null) 0 else self.restoreKept(s, ids.len);
    s.fill_at.store(@intCast(from), .monotonic); // what was restored: 0 when the kept state failed or was skipped
    return from;
}

/// The pass's kept marks, as the lane host's prompt cache asks for them.
fn keepOf(s: *lanes.Stream) ?engine.Keep {
    return if (s.reuse.marks.len == 0) null else .{ .at = s.reuse.marks, .ctx = s, .call = Cuda.keptAt };
}

/// A pass's rows read back from its start: prompt logprobs and taps begin empty at `from`.
fn reset(self: *Cuda, s: *lanes.Stream, from: usize) void {
    s.prompt_rows.clearRetainingCapacity();
    const t = &self.lanes.getPtr(s).?.taps;
    t.prompt.clearRetainingCapacity();
    t.from = from;
    t.verified = null;
}

/// The engine's readers of this call's chunks (prompt logprobs, taps); `clear` takes them off again.
fn hooks(self: *Cuda, s: *lanes.Stream, sink: *Cuda.PromptSink, tap: *Cuda.TapSink) void {
    const e = self.e;
    if (s.prompt_logprobs) |k| {
        // a logits buffer of its own takes batch_rows a head pass; without the memory, the window's rows
        if (self.prompt_logits == null) self.prompt_logits = cuda.DeviceBuffer.alloc(e.ctx.d, prompt_rows.batch_rows * e.c.vocab * 2) catch null;
        const rows: usize = if (self.prompt_logits != null) prompt_rows.batch_rows else state.windowRows(e.c);
        const words = self.prompt_out.slice(u32);
        const picks_at = rows * row_words; // after the rows
        e.prompt_rows = .{ .k = k, .rows = rows, .logits = if (self.prompt_logits) |b| b.ptr else e.b.logits, .picks = words[picks_at..][0..rows], .picks_dev = self.prompt_dev + picks_at * 4, .out = words[0..picks_at], .out_dev = self.prompt_dev, .stride = row_words, .ctx = sink, .call = Cuda.PromptSink.take };
    }
    if (self.tapped != null) e.taps = .{ .ctx = tap, .call = Cuda.TapSink.take };
}

fn clear(self: *Cuda) void {
    self.e.prompt_rows = null;
    self.e.taps = null;
}

/// The whole pass from `from`: the images through the tower, every chunk, then the first token.
fn run(self: *Cuda, s: *lanes.Stream, from: usize) !void {
    const e = self.e;
    if (s.images.len > 0) try (e.vision orelse return error.NoVision).begin(e.stream, s.images, from);
    defer if (e.vision) |v| v.end();
    reset(self, s, from);
    var sink: Cuda.PromptSink = .{ .gpa = self.gpa, .s = s };
    var tap: Cuda.TapSink = .{ .self = self, .taps = &self.lanes.getPtr(s).?.taps };
    hooks(self, s, &sink, &tap);
    defer clear(self);
    const first = try Cuda.servedPrefill(e, self.head, s, from, keepOf(s));
    try finish(self, s, first);
}

/// The pass's end: the head's first draft reads the prompt's last row, the first token's handle and logprob row.
fn finish(self: *Cuda, s: *lanes.Stream, first: u32) !void {
    const e = self.e;
    const ids = s.prompt();
    // the head's first draft reads the prompt's last row (its hidden row waits where a window's would)
    const last = ids.len - 1 - e.last_chunk;
    try e.ops().copy(e.b.hidden, e.b.p_hidden + last * @as(u64, e.c.hidden) * 2, e.c.hidden * 2);
    _ = self.take(first);
    s.fill_at.store(@intCast(ids.len), .monotonic); // the whole prompt, as a whole pass or a last chunk ends it
    if (s.logprobs) |k| {
        const at = (row_slots - 1) * row_words;
        try e.ops().logprobRows(e.b.p_logits, e.c.vocab, e.b.p_sampled, k, false, self.rows_dev + at * 4, 1);
        try e.stream.synchronize();
        self.lanes.getPtr(s).?.first = LogRow.fromWords(self.rows_out.slice(u32)[at..][0..row_words], first, k);
    }
}

/// prefill_step: a chunk a call, other streams' rounds between; image prompts and segmented passes run whole at once.
pub fn step(ptr: *anyopaque, s: *lanes.Stream, rows: u32) anyerror!bool {
    return stepOnce(ptr, s, rows) catch |err| {
        // any failure but a cancel ends the stream here, as direct callers expect (open.zig); the host finds it gone
        if (err != error.Cancelled) @as(*Cuda, @ptrCast(@alignCast(ptr))).release(s);
        return err;
    };
}

fn stepOnce(ptr: *anyopaque, s: *lanes.Stream, _: u32) anyerror!bool {
    const self: *Cuda = @ptrCast(@alignCast(ptr));
    const e = self.e;
    var sink: Cuda.PromptSink = .{ .gpa = self.gpa, .s = s };
    var tap: Cuda.TapSink = undefined;
    if (self.fills.getPtr(s) == null) {
        if (s.isCancelled()) return error.Cancelled; // cancelled while it waited: no sequence, restore or tower
        const from = try open(self, s);
        const cancel: engine.Cancel = .{ .ptr = s, .check = Cuda.cancelled, .yields = s.yield_check != null };
        tap = .{ .self = self, .taps = &self.lanes.getPtr(s).?.taps };
        hooks(self, s, &sink, &tap);
        const alone = s.images.len > 0 or pass.segmented(e, s.prompt(), from, null, cancel, keepOf(s), lanes.cutPoints(s));
        clear(self);
        if (alone) {
            try run(self, s, from);
            return true;
        }
        try pass.begin(e, s.prompt(), from, self.head);
        reset(self, s, from);
        const p = try pass.Pass.begin(e, s.prompt(), from, null, self.head, cancel, keepOf(s), lanes.cutPoints(s));
        try self.fills.put(self.gpa, s, p);
    } else e.bind(self.lanes.getPtr(s).?.seq); // its sequence, sampling, decision row and head, as its last chunk left them
    const p = self.fills.getPtr(s).?;
    tap = .{ .self = self, .taps = &self.lanes.getPtr(s).?.taps };
    hooks(self, s, &sink, &tap);
    defer clear(self);
    const waited = e.heat_gate.waited_s;
    const done = p.next(e) catch |err| {
        if (err == error.Cancelled) s.keepInterruptedPrefill(@intCast(e.pos));
        _ = self.fills.remove(s);
        return err;
    };
    if (e.heat_gate.waited_s > waited) heat.note(e.heat_gate.waited_s - waited);
    if (!done) return unfinished(e, s, p.s);
    const from = p.from;
    _ = self.fills.remove(s);
    const first = try pass.firstToken(e);
    if (e.last_chunk == from) lanes.prefill_widths.completed(e.gpa, s, s.prompt().len - from);
    try finish(self, s, first);
    return true;
}

/// A chunk that left its pass unfinished: its rows reported and, with settle_step, its queued GPU work waited for.
pub fn unfinished(e: *engine.Engine, s: *lanes.Stream, at: usize) !bool {
    s.fill_at.store(@intCast(at), .monotonic);
    if (s.settle_step) try e.stream.synchronize(); // the lane host bills this call's time against decode's share
    return false;
}

/// cancel_prefill: a pass cancelled between chunks keeps its prefix where its last chunk ended, on its own sequence.
pub fn keepCancelled(ptr: *anyopaque, s: *lanes.Stream) anyerror!void {
    const self: *Cuda = @ptrCast(@alignCast(ptr));
    const p = self.fills.getPtr(s) orelse return; // no pass between chunks: nothing filled past what was kept
    self.e.bind(self.lanes.getPtr(s).?.seq); // the prompt cache's save reads the bound sequence (cuda_reuse.save)
    s.keepInterruptedPrefill(@intCast(p.s));
}

/// A stream released between its chunks: its pass is dropped (the lane host keeps any kept prefix).
pub fn forget(self: *Cuda, s: *const lanes.Stream) void {
    _ = self.fills.remove(s);
}
