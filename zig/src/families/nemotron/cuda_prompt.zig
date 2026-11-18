//! The Nemotron CUDA engine's prompt pass: a prompt's chunks, serial or in staggered segments, and its first token.

const std = @import("std");
const engine = @import("cuda_engine.zig");
const Engine = engine.Engine;
const Cancel = engine.Cancel;
const Keep = engine.Keep;
const Mask = engine.Mask;
const Dump = @import("cuda_dump.zig").Dump;
const Head = @import("cuda_mtp.zig").Head;
const state = @import("cuda_state.zig");
const segs = @import("cuda_segments.zig");
const marks = @import("cuda_marks.zig");
const cuts = @import("cuda_prompt_cuts.zig");
const prompt_rows = @import("cuda_prompt_rows.zig");
const reuse = @import("cuda_reuse.zig");
const lesson_route = @import("living").lesson_route;
const heat = @import("heat");
const prompt_tail = @import("core").prompt_tail;
const pin_sampled = engine.pin_sampled;
const pin_prompt = engine.pin_prompt;

/// End at the nearest grid step, request start, store mark or prompt end.
pub fn chunkEnd(at: usize, len: usize, step: usize, chunks: []const u32, store_marks: []const u32) usize {
    var end = cuts.end(at, len, step, chunks);
    end = cuts.end(at, end, end - at, store_marks); // step `end - at`: only a nearer mark can shorten it
    return end;
}

/// decode.prefill: a fresh state, the prompt in passes of up to 4,096 rows or in segments (the head too); the token.
pub fn prefill(e: *Engine, prompt: []const u32, dump: ?*Dump, head: ?*Head) !u32 {
    return e.prefillWith(prompt, dump, head, null);
}

/// prefill, stopping with error.Cancelled at the next chunk boundary once `cancel` says so.
pub fn prefillWith(e: *Engine, prompt: []const u32, dump: ?*Dump, head: ?*Head, cancel: ?Cancel) !u32 {
    return e.prefillFrom(prompt, 0, dump, head, cancel, null, &.{});
}

/// prefill from `from` (0 resets; else a state restored at that row); `chunks` are the request's message starts.
pub fn prefillFrom(e: *Engine, prompt: []const u32, from: usize, dump: ?*Dump, head: ?*Head, cancel: ?Cancel, keep: ?Keep, chunks: []const u32) !u32 {
    try begin(e, prompt, from, head);
    const waited = e.heat_gate.waited_s;
    if (segmented(e, prompt, from, dump, cancel, keep, chunks)) {
        try segs.prefill(e, try e.segmentSet(), prompt, head, e.segments, cancel);
    } else try e.serialChunks(prompt, from, dump, head, cancel, keep, chunks);
    if (e.heat_gate.waited_s > waited) heat.note(e.heat_gate.waited_s - waited);
    return firstToken(e);
}

/// A pass from `from` on the bound sequence: checked, and a fresh state (the head's too) when it starts at row 0.
pub fn begin(e: *Engine, prompt: []const u32, from: usize, head: ?*Head) !void {
    if (prompt.len == 0) return error.EmptyPrompt;
    if (from >= prompt.len) return error.BadReuse;
    if (prompt.len + state.max_rows > e.cap()) return error.ContextFull;
    if (from == 0) {
        if (e.usage) |u| u.trace.restore(0, .{});
        try e.reset();
        if (head) |h| try h.reset();
    } else if (e.pos != from) return error.BadReuse;
}

/// Whether prefillFrom runs the pass as staggered segments (named request cuts keep the serial chunks).
pub fn segmented(e: *const Engine, prompt: []const u32, from: usize, dump: ?*Dump, cancel: ?Cancel, keep: ?Keep, chunks: []const u32) bool {
    return from == 0 and keep == null and chunks.len == 0 and e.vision == null and e.audio == null and e.segments > 1 and dump == null and e.prompt_rows == null and e.taps == null and e.usage == null and prompt.len > state.prefill_rows and e.decide == lesson_route.none and !Cancel.background(cancel);
}

/// The pass's first token, which its last chunk drew on the GPU.
pub fn firstToken(e: *Engine) !u32 {
    const host = e.pinned.slice(u32)[pin_sampled..][0..1];
    try e.ops().download(std.mem.sliceAsBytes(host), e.b.p_sampled);
    try e.stream.synchronize();
    if (e.trace) e.traceLessons();
    return host[0];
}

/// The prompt's next token at `position` drawn again under a grammar's bits (structured output's first token).
pub fn redrawFirst(e: *Engine, position: u64, mask: Mask) !u32 {
    const o = e.ops();
    try o.grammarMask(e.b.p_logits, e.c.vocab, mask.bits, mask.words, 1);
    try o.fill32(e.b.p_meta, @intCast(position - 1), 4); // the prompt's last row: the draw keys position meta + 1
    try e.forward(null).sample(e.b.p_logits, 1, e.b.p_meta, e.b.p_sampled, e.sampling != null, e.b.rule);
    const host = e.pinned.slice(u32)[pin_sampled..][0..1];
    try o.download(std.mem.sliceAsBytes(host), e.b.p_sampled);
    try e.stream.synchronize();
    return host[0];
}

/// Heat bands, if set, before the chunk; a cancel ends the wait. Unset bands return without reading a zone.
pub fn beforeChunk(e: *Engine, cancel: ?Cancel) !void {
    try e.heat_gate.beforePromptChunk(e.io, if (cancel) |c| .{ .ptr = c.ptr, .check = c.check } else null);
}

/// The prompt's chunks one after another on the engine's stream, starting at `from`.
pub fn serialChunks(e: *Engine, prompt: []const u32, from: usize, dump: ?*Dump, head: ?*Head, cancel: ?Cancel, keep: ?Keep, chunks: []const u32) !void {
    var pass = try Pass.begin(e, prompt, from, dump, head, cancel, keep, chunks);
    while (!try pass.next(e)) {}
}

/// A serial pass's state between its chunks: the whole pass runs `next` in a loop, a stepped one a call a round.
pub const Pass = struct {
    prompt: []const u32,
    from: usize,
    s: usize, // the next chunk's first row
    dump: ?*Dump,
    head: ?*Head,
    cancel: ?Cancel,
    keep: ?Keep,
    chunks: []const u32,
    lessons: bool, // a lesson adapter latches as the pass runs
    wide: bool, // passes of up to pass_rows (rows equal at any cut)
    background: bool,

    /// The pass from `from` on the bound sequence (begin() done): no lesson acts until it decides.
    pub fn begin(e: *Engine, prompt: []const u32, from: usize, dump: ?*Dump, head: ?*Head, cancel: ?Cancel, keep: ?Keep, chunks: []const u32) !Pass {
        const shared = e.lessons();
        if (shared) |ad| try e.trainOps().latch(ad.score, ad.need, ad.latch, e.slot, false);
        const wide = dump == null and shared == null and e.decide == lesson_route.none;
        return .{ .prompt = prompt, .from = from, .s = from, .dump = dump, .head = head, .cancel = cancel, .keep = keep, .chunks = chunks, .lessons = shared != null, .wide = wide, .background = Cancel.background(cancel) };
    }

    /// One chunk on the bound sequence; true once the prompt is all in.
    pub fn next(p: *Pass, e: *Engine) !bool {
        const prompt = p.prompt;
        const s = p.s;
        if (s >= prompt.len) return true;
        const keep = p.keep;
        if (p.background) try e.stream.synchronize(); // one chunk in flight: a request that arrives waits for that one alone
        if (Cancel.now(p.cancel)) {
            if (p.background and s > p.from) if (keep) |k| k.call(k.ctx, @intCast(s)); // what it prefilled stays for the next turn
            return error.Cancelled;
        }
        const shared = e.lessons();
        var f = e.forward(p.dump);
        f.usage = e.usage;
        try e.beforeChunk(p.cancel);
        const step: usize = if (p.wide and p.background) state.yield_rows else if (p.wide and inside(keep, s, s + state.pass_rows) < marks.max) state.pass_rows else state.prefill_rows; // 2,048 rows where a wide pass would crowd the scan's marks
        const forced: []const u32 = if (keep) |k| (if (k.cut) k.at else &.{}) else &.{};
        var end = chunkEnd(s, prompt.len, step, p.chunks, forced);
        // the decision row ends its chunk: the rows after it read the lessons it latched
        if (e.decide != lesson_route.none and e.decide >= s and e.decide + 1 < end) end = @intCast(e.decide + 1);
        const chunk = prompt[s..end];
        e.last_chunk = s;
        if (e.usage) |u| try u.trace.start(s, chunk.len);
        if (keep) |k| arm(e, k, s, end);
        defer e.armed.n = 0;
        f.captures = e.armed.marks();
        const lessons = p.lessons or e.decide != lesson_route.none;
        const readers = p.head != null or p.dump != null or e.taps != null or e.prompt_rows != null or lessons;
        f.unread = prompt_tail.unread(.{ .logits = end == prompt.len, .readers = readers });
        f.route = .{};
        f.route.add(0, chunk.len, e.pos, e.decide, e.slot);
        try e.copied.synchronize();
        const host = e.promptHost(chunk.len);
        @memcpy(host, chunk);
        // a speech prompt's <|audio|> placeholders lie past the token table: their rows are the clip's (cuda_audio.zig)
        if (e.audio != null) for (host) |*t| if (t.* >= e.c.vocab) {
            t.* = 0;
        };
        try e.ops().upload(e.b.p_ids, std.mem.sliceAsBytes(host));
        try e.copied.record(e.stream);
        try f.chunk(@intCast(chunk.len), e.pos);
        if (shared) |ad| if (f.route.decision(0) != null) try e.trainOps().latch(ad.score, ad.need, ad.latch, e.slot, true);
        e.pos += @intCast(chunk.len);
        const known = @min(chunk.len, prompt.len - 1 - s);
        if (e.prompt_rows) |pr| try prompt_rows.run(e, pr, prompt[s + 1 ..][0..known]);
        if (p.head) |h| if (known > 0) try h.absorb(e.b.p_hidden, prompt[s + 1 ..][0..known], s + 1);
        if (e.taps) |t| try t.call(t.ctx, e.b.p_hidden, s, chunk.len);
        try reuse.finish(.{ .e = e, .head = p.head });
        p.s = end;
        if (keep) |k| if (end <= std.math.maxInt(u32) and
            (std.mem.indexOfScalar(u32, k.at, @intCast(end)) != null or std.mem.indexOfScalar(u32, p.chunks, @intCast(end)) != null)) k.call(k.ctx, @intCast(end));
        return end >= prompt.len;
    }
};

/// The keep marks strictly inside rows [s, end).
pub fn inside(keep: ?Keep, s: usize, end: usize) usize {
    const k = keep orelse return 0;
    var n: usize = 0;
    for (k.at) |m| n += @intFromBool(m > s and m < end);
    return n;
}

/// The keeps of rows strictly inside the chunk [s, end), made before it runs: the chunk writes their Mamba states.
fn arm(e: *Engine, k: Keep, s: usize, end: usize) void {
    e.armed.open = .{ s, end };
    defer e.armed.open = null;
    for (k.at) |m| if (m > s and m < end) k.call(k.ctx, m);
}

/// Whole chunks a prompt call runs as staggered segments from here on; their streams and scratch are made now.
pub fn setSegments(e: *Engine, n: usize) !void {
    if (n < 1 or n > segs.MAX) return error.BadSegments;
    e.segments = n;
    if (n > 1) _ = try e.segmentSet();
}

/// Pinned words for `n` prompt ids on their way up (free once `copied` has completed).
pub fn promptHost(e: *Engine, n: usize) []u32 {
    return e.pinned.slice(u32)[pin_prompt..][0..n];
}

/// The segments' streams and scratch for calls of e.segments, remade when a call needs more of them.
pub fn segmentSet(e: *Engine) !*segs.Segments {
    if (e.seg) |*s| {
        if (s.n >= e.segments) return s;
        try e.stream.synchronize();
        s.deinit();
        e.seg = null;
    }
    e.seg = try segs.Segments.init(e, e.segments);
    return &e.seg.?;
}

test {
    _ = @import("cuda_request_starts_test.zig");
}
