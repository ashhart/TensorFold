//! Inkling 2 Mini lane execution and snapshot adapter.
const std = @import("std");
const cuda = @import("cuda");
const lanes = @import("lanes");
const engine = @import("loop_engine.zig");
const ops = @import("ops.zig");

const Model = engine.Model;
const Kernels = engine.Kernels;
const prompt_rows = engine.prompt_rows;
const window_rows = engine.window_rows;
const round_rows = engine.round_rows;
const token_ring = 1024;
const LogRow = lanes.LogprobRow;
const row_words = lanes.logprob.words;

const Lane = struct {
    len: usize = 0,
    fill: ?usize = null,
    base: usize = 0,
    rows: std.ArrayList(u32) = .empty,
    last: u64,
    rule: u64,
    sampled: bool = false,
    first: ?LogRow = null, // the first token's logprob row, with `logprobs`
};

/// The bytes a slot's lane buffers ask for: a vocabulary row of last logits and a sampling rule.
pub fn laneAllocs(vocab: u32) [2]usize {
    return .{ @as(usize, vocab) * 4, @sizeOf(cuda.sample.Rule) };
}

/// The device bytes each served stream holds: its model slot (engine.slotAllocs) and its lane buffers.
pub fn streamBytes(m: *const Model) u64 {
    const slot = engine.slotAllocs(&m.layout, m.w);
    const lane = laneAllocs(m.w.c.text.unpadded_vocab);
    return slot[0] + slot[1] + slot[2] + slot[3] + lane[0] + lane[1];
}

pub const Backend = struct {
    gpa: std.mem.Allocator,
    d: *const cuda.Driver,
    m: *Model,
    k: *const Kernels,
    s: cuda.Stream,
    lanes_: [engine.max_slots]?Lane = @splat(null),
    streams: [engine.max_slots]?*lanes.Stream = @splat(null), // the stream each model slot holds
    tokens: cuda.DeviceBuffer,
    logits16: cuda.DeviceBuffer,
    meta: cuda.DeviceBuffer,
    words: cuda.DeviceBuffer, // logprob words, a round's rows of row_words
    slot: [engine.max_slots][2]cuda.DeviceBuffer = undefined, // each slot's last prompt logits and sampling rule
    next: u64 = 0,
    held: usize = 0, // slots with buffers

    const be = lanes.backend;

    pub fn init(gpa: std.mem.Allocator, d: *const cuda.Driver, m: *Model, k: *const Kernels, s: cuda.Stream) !Backend {
        const v = m.w.c.text.unpadded_vocab;
        var tokens = try cuda.DeviceBuffer.alloc(d, (token_ring + round_rows) * 4);
        errdefer tokens.free();
        var logits16 = try cuda.DeviceBuffer.alloc(d, window_rows * v * 2);
        errdefer logits16.free();
        var meta = try cuda.DeviceBuffer.alloc(d, 16);
        errdefer meta.free();
        var words = try cuda.DeviceBuffer.alloc(d, round_rows * row_words * 4);
        errdefer words.free();
        var b: Backend = .{ .gpa = gpa, .d = d, .m = m, .k = k, .s = s, .tokens = tokens, .logits16 = logits16, .meta = meta, .words = words };
        try b.slotBuffers(m.n_slots);
        return b;
    }

    /// Each model slot's own last-logits and rule buffers, up to `n`.
    pub fn slotBuffers(b: *Backend, n: usize) !void {
        const sizes = laneAllocs(b.m.w.c.text.unpadded_vocab);
        for (0..n) |i| {
            if (i < b.held) continue;
            var last = try cuda.DeviceBuffer.alloc(b.d, sizes[0]);
            errdefer last.free();
            b.slot[i] = .{ last, try cuda.DeviceBuffer.alloc(b.d, sizes[1]) };
            b.held = i + 1;
        }
    }

    /// The server's stream count: one model slot and its buffers per stream.
    pub fn reserve(b: *Backend, streams: u32) !void {
        if (streams > engine.max_slots) return error.Inkling2TooManyStreams;
        try b.m.addSlots(b.s, streams);
        try b.slotBuffers(streams);
    }

    pub fn deinit(b: *Backend) void {
        for (&b.lanes_) |*x| if (x.*) |*l| l.rows.deinit(b.gpa);
        for (b.slot[0..b.held]) |*pair| for (pair) |*x| x.free();
        b.words.free();
        b.meta.free();
        b.logits16.free();
        b.tokens.free();
        b.* = undefined;
    }

    pub fn backend(b: *Backend) be.Backend {
        return .{ .ptr = b, .vtable = &vtable };
    }

    pub fn facts(_: *const Backend) lanes.Model {
        return .{ .exact_width = window_rows, .gpu_tokens = false, .mtp = false, .speculate = false, .hidden_rows = true, .plain_share = true, .batch_rows = round_rows, .max_streams = engine.max_slots, .streams_exact = true };
    }

    pub fn admits(b: *const Backend, prompt: usize, max_new: usize) bool {
        return b.freeSlot() != null and prompt <= b.m.context and max_new <= b.m.context - prompt;
    }

    fn freeSlot(b: *const Backend) ?usize {
        for (b.streams[0..@min(b.m.n_slots, b.held)], 0..) |x, i| if (x == null) return i;
        return null;
    }

    fn slotOf(b: *const Backend, s: *const lanes.Stream) ?usize {
        for (b.streams, 0..) |x, i| if (x == s) return i;
        return null;
    }

    /// The device buffer holding `s`'s last prompt logits.
    pub fn lastOf(b: *const Backend, s: *const lanes.Stream) !u64 {
        const i = b.slotOf(s) orelse return error.UnknownStream;
        return b.lanes_[i].?.last;
    }

    fn self(ptr: *anyopaque) *Backend {
        return @ptrCast(@alignCast(ptr));
    }

    fn upload(b: *Backend, dst: u64, bytes: []const u8) !void {
        try cuda.copy_source.enqueue(b.d, dst, bytes, b.s);
    }

    /// `s`'s lane, its model slot selected for the calls that follow.
    fn laneOf(b: *Backend, s: *const lanes.Stream) !*Lane {
        const i = b.slotOf(s) orelse return error.UnknownStream;
        if (i >= b.m.n_slots) return error.UnknownStream;
        b.m.slot = i;
        return if (b.lanes_[i]) |*l| l else error.UnknownStream;
    }

    fn resumeSaved(b: *Backend, s: *lanes.Stream) usize {
        const saved = s.reuse.saved orelse return 0;
        const k: *const Model.Saved = @ptrCast(@alignCast(saved));
        if (k.at != s.reuse.at or k.at == 0 or k.at >= s.prompt().len) {
            s.reuse_failed = true;
            return 0;
        }
        b.m.restore(b.s, k) catch {
            s.reuse_failed = true;
            return 0;
        };
        s.cached = k.at;
        return k.at;
    }

    fn opening(b: *Backend, s: *lanes.Stream) !*Lane {
        const prompt = s.prompt();
        if (prompt.len == 0) return error.EmptyPrompt;
        if (prompt.len > b.m.context or s.max_new > b.m.context - prompt.len) return error.PastContext;
        if (b.slotOf(s) == null) {
            const i = b.freeSlot() orelse return error.NoFreeSlot;
            b.lanes_[i] = .{ .last = b.slot[i][0].ptr, .rule = b.slot[i][1].ptr };
            b.streams[i] = s;
        }
        const l = try b.laneOf(s);
        l.sampled = false;
        l.first = null;
        if (cuda.sample.ruleFor(s.sampling)) |r| {
            try b.upload(l.rule, std.mem.asBytes(&r));
            l.sampled = true;
        }
        l.len = 0;
        try b.m.reset(b.s);
        const at = b.resumeSaved(s);
        l.len = at;
        l.fill = at;
        s.fill_at.store(@intCast(at), .monotonic);
        return l;
    }

    fn chunk(b: *Backend, s: *lanes.Stream, l: *Lane) !bool {
        const prompt = s.prompt();
        const at = l.fill.?;
        if (s.isCancelled()) return error.Cancelled;
        var end = @min(prompt.len, (at / prompt_rows + 1) * prompt_rows);
        for (s.reuse.marks) |mark| {
            if (mark > at and mark < end) end = mark;
        }
        const done = end == prompt.len;
        try b.m.forward(b.k, b.s, prompt[at..end], at, end - at - 1, @intFromBool(done), false);
        if (done) try b.d.check(b.d.api.cuMemcpyDtoDAsync_v2(l.last, b.m.s.logits, b.m.w.c.text.unpadded_vocab * 4, b.s.handle), "cuMemcpyDtoDAsync");
        l.len = end;
        l.fill = if (done) null else end;
        s.fill_at.store(@intCast(end), .monotonic);
        if (s.reuse.hook) |hook| if (std.mem.indexOfScalar(u32, s.reuse.marks, @intCast(end)) != null)
            hook.at(hook.ptr, s, @intCast(end));
        return done;
    }

    fn handle(b: *Backend) usize {
        const h = b.next;
        b.next += 1;
        return @intCast(h % token_ring);
    }

    fn drawRows(b: *Backend, l: *const Lane, logits: u64, rows: usize, position0: u64, at: usize) !void {
        const v = b.m.w.c.text.unpadded_vocab;
        if (!l.sampled) return ops.top1(&b.k.glue, b.s, logits, rows, v, v, 0, b.m.s.t + at * 4, b.tokens.ptr + at * 4);
        try ops.tobf(&b.k.glue, b.s, logits, rows, v, v, b.logits16.ptr, v, 0, v);
        const metadata = [4]i32{ @intCast(position0 - 1), 0, 0, 0 };
        try b.upload(b.meta.ptr, std.mem.asBytes(&metadata));
        try b.k.draws.draw(b.s, b.logits16.ptr, v, l.rule, b.meta.ptr, 0, b.tokens.ptr + at * 4, rows, null, null);
    }

    /// Logprob words at word row `word_row` for `rows` fp32 rows drawn at tokens `at`: the bf16 rows a draw reads.
    fn wordRows(b: *Backend, l: *const Lane, logits: u64, rows: usize, at: usize, k: u8, word_row: usize) !void {
        const v = b.m.w.c.text.unpadded_vocab;
        if (!l.sampled) try ops.tobf(&b.k.glue, b.s, logits, rows, v, v, b.logits16.ptr, v, 0, v);
        try b.k.draws.logprobRows(b.s, b.logits16.ptr, v, b.tokens.ptr + at * 4, k, false, b.words.ptr + word_row * row_words * 4, rows);
    }

    fn read(ptr: *anyopaque, h: u64) anyerror!u32 {
        const b = self(ptr);
        if (h >= b.next or b.next - h > token_ring) return error.NoSuchToken;
        var out: [1]u32 = undefined;
        try b.s.synchronize();
        try b.tokens.download(@intCast((h % token_ring) * 4), std.mem.sliceAsBytes(&out));
        return out[0];
    }

    const vtable: be.Backend.VTable = .{
        .prefill = prefill,
        .prefill_step = prefillStep,
        .first = first,
        .first_masked = firstMasked,
        .queue = queue,
        .read = read,
        .verify = verify,
        .keep = keep,
        .draft = draft,
        .release = release,
        .first_row = firstRow,
        .reports_fill = true,
    };

    fn prefill(ptr: *anyopaque, s: *lanes.Stream) anyerror!void {
        const b = self(ptr);
        const l = try b.opening(s);
        errdefer release(ptr, s);
        while (!try b.chunk(s, l)) {}
    }

    fn prefillStep(ptr: *anyopaque, s: *lanes.Stream, _: u32) anyerror!bool {
        const b = self(ptr);
        const going = if (b.slotOf(s)) |i| b.lanes_[i].?.fill != null else false;
        const l = if (going) try b.laneOf(s) else try b.opening(s);
        errdefer release(ptr, s);
        return b.chunk(s, l);
    }

    fn first(ptr: *anyopaque, s: *lanes.Stream, position: u64) anyerror!u64 {
        const b = self(ptr);
        const l = try b.laneOf(s);
        if (position != l.len or l.fill != null) return error.PositionMismatch;
        const h = b.next;
        const at = b.handle();
        try b.drawRows(l, l.last, 1, position, at);
        if (s.logprobs) |k| {
            try b.wordRows(l, l.last, 1, at, k, 0);
            var w: [row_words]u32 = undefined;
            try b.s.synchronize();
            try b.words.download(0, std.mem.sliceAsBytes(&w));
            l.first = LogRow.fromWords(&w, try read(b, h), k);
        }
        return h;
    }

    fn firstRow(ptr: *anyopaque, s: *lanes.Stream) anyerror!LogRow {
        const l = try self(ptr).laneOf(s);
        return l.first orelse error.NoFirstRow;
    }

    fn firstMasked(_: *anyopaque, _: *lanes.Stream, _: u64, _: []const u32) anyerror!u64 {
        return error.StructuresUnsupported;
    }

    fn feed(b: *Backend, value: be.Feed) !u32 {
        return switch (value) {
            .value => |v| v,
            .handle => |h| read(b, h),
        };
    }

    fn queue(ptr: *anyopaque, s: *lanes.Stream, value: be.Feed, position: u64) anyerror!u64 {
        const b = self(ptr);
        const l = try b.laneOf(s);
        const token = try b.feed(value);
        if (position != l.len + 1) return error.PositionMismatch;
        try b.m.forward(b.k, b.s, &.{token}, l.len, 0, 1, false);
        l.len += 1;
        const h = b.next;
        try b.drawRows(l, b.m.s.logits, 1, position, b.handle());
        return h;
    }

    /// Every window in one forward (loop_round.zig): each stream's rows on its own slot, its bits its solo window's.
    fn verify(ptr: *anyopaque, windows: []const be.Window, out: []be.Verified) anyerror!void {
        const b = self(ptr);
        if (windows.len == 0 or windows.len != out.len or windows.len > engine.max_slots) return error.BadWindows;
        var chains: [engine.max_slots]engine.Model.Chain = undefined;
        var total: usize = 0;
        for (windows, 0..) |w, i| {
            if (w.held != 0) return error.NoHeldDrafts;
            if (w.parents != null) return error.ChainsOnly;
            if (w.masks.len != 0) return error.StructuresUnsupported;
            const l = try b.laneOf(w.stream);
            const n = w.rows();
            if (n == 0 or n > window_rows) return error.WindowTooWide;
            l.base = l.len;
            l.rows.clearRetainingCapacity();
            try l.rows.append(b.gpa, w.pending);
            try l.rows.appendSlice(b.gpa, w.tokens);
            for (w.positions, 0..) |position, r| if (position != l.base + r + 1) return error.PositionMismatch;
            chains[i] = .{ .slot = b.m.slot, .p0 = l.base, .tokens = l.rows.items };
            total += n;
        }
        if (total > round_rows) return error.RoundTooWide;
        try b.m.forwardChains(b.k, b.s, chains[0..windows.len], 0, total, true);
        const v = b.m.w.c.text.unpadded_vocab;
        var at: usize = 0;
        var any_rows = false;
        for (windows, out) |w, o| {
            const l = try b.laneOf(w.stream);
            const n = l.rows.items.len;
            try b.drawRows(l, b.m.s.logits + at * v * 4, n, w.positions[0], token_ring + at);
            if (o.rows.len > 0) {
                if (o.rows.len != n) return error.LogprobRowsMismatch;
                try b.wordRows(l, b.m.s.logits + at * v * 4, n, token_ring + at, w.stream.logprobs orelse return error.LogprobRowsMismatch, at);
                any_rows = true;
            }
            at += n;
        }
        var got: [round_rows]u32 = undefined;
        var words: [round_rows * row_words]u32 = undefined;
        try b.s.synchronize();
        try b.tokens.download(token_ring * 4, std.mem.sliceAsBytes(got[0..total]));
        if (any_rows) try b.words.download(0, std.mem.sliceAsBytes(words[0 .. total * row_words]));
        at = 0;
        for (windows, out) |w, *o| {
            const l = try b.laneOf(w.stream);
            const n = l.rows.items.len;
            @memcpy(o.sampled[0..n], got[at..][0..n]);
            @memcpy(o.drafts[0 .. n - 1], l.rows.items[1..n]);
            if (o.rows.len > 0) for (o.rows, 0..) |*r, i| {
                r.* = LogRow.fromWords(words[(at + i) * row_words ..][0..row_words], got[at + i], w.stream.logprobs.?);
            };
            l.len = l.base + n;
            at += n;
        }
    }

    fn keep(ptr: *anyopaque, windows: []const be.Window, paths: []const []const u32) anyerror!void {
        const b = self(ptr);
        if (windows.len != paths.len) return error.BadWindows;
        for (windows, paths) |w, path| {
            const l = try b.laneOf(w.stream);
            if (path.len == 0 or path.len > l.rows.items.len) return error.EmptyPath;
            for (path, 0..) |row, i| if (row != i) return error.ChainsOnly;
            try b.m.rewind(b.k, b.s, path.len);
            l.len = l.base + path.len;
        }
    }

    fn draft(_: *anyopaque, requests: []const be.DraftRequest) anyerror!void {
        for (requests) |r| if (r.depth != 0 or r.lanes != null) return error.NoDraftHead;
    }

    pub fn saveState(b: *Backend, owner: ?*anyopaque, at: u32) !*Model.Saved {
        const s: *const lanes.Stream = @ptrCast(@alignCast(owner orelse return error.NoOwner));
        const l = try b.laneOf(s);
        if ((l.fill orelse l.len) != at) return error.NotAtMark;
        return b.m.save(b.s, at);
    }

    pub fn restoreState(b: *Backend, owner: ?*anyopaque, saved: *const Model.Saved) !void {
        const s: *const lanes.Stream = @ptrCast(@alignCast(owner orelse return error.NoOwner));
        const l = try b.laneOf(s);
        try b.m.restore(b.s, saved);
        l.len = saved.at;
    }

    pub fn dropState(b: *Backend, saved: *Model.Saved) void {
        b.m.drop(b.s, saved);
    }

    fn release(ptr: *anyopaque, s: *lanes.Stream) void {
        const b = self(ptr);
        const i = b.slotOf(s) orelse return;
        if (b.lanes_[i]) |*l| l.rows.deinit(b.gpa);
        b.lanes_[i] = null;
        b.streams[i] = null;
    }
};
