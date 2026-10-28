//! MiniMax-M3.1-Flash's CUDA lane backend: a slot per stream, verify windows (shared across streams), chunked prompts.

const std = @import("std");
const lanes = @import("lanes");
const cuda = @import("cuda");
const engine = @import("cuda_engine.zig");
const follow = @import("cuda_follow.zig");

const be = lanes.backend;

/// First tokens a handle names.
const ring = 1024;
pub const max_rows = engine.window_rows;
/// A DSpark block's drafts at most.
const max_held = 7;

/// A window's ms by rows, EXL3 pack on two GB10s at TP=2: mean of positions 1k and 8k, never below a narrower one's.
const gb10_ms = [_]f64{ 54.5, 64.4, 74.1, 84.1, 94.6, 102.6, 108.5, 114.2, 120.8, 124.9, 130.1, 136.2, 140.6, 145.2, 149.7, 152.7, 166.2, 173.4, 175.9, 182.2, 188.6, 195.5, 199.6, 203.1, 207.6, 211.8, 213.4, 215.3, 223.2, 224.9, 226.0, 244.8 };
const gb10_window_costs = costTable(max_rows);
const gb10_shared_costs = costTable(engine.shared_rows);

fn costTable(comptime n: usize) [n]lanes.config.Cost {
    var c: [n]lanes.config.Cost = undefined;
    for (&c, gb10_ms[0..n], 0..) |*x, ms, i| x.* = .{ .width = i + 1, .ms = ms };
    return c;
}

/// A stream's place in the backend: its sequence's slot, the rows its last verify left uncommitted, its first token.
const Lane = struct {
    slot: u32,
    pending_rows: ?usize = null,
    held: [max_held]u32 = undefined, // the drafter's block for the next window
    held_n: usize = 0,
    first: ?u64 = null,
    fill: ?usize = null, // a prompt pass going a chunk at a time: where its next chunk starts
};

pub const Lanes = struct {
    x: follow.Leader,
    lanes: std.AutoHashMapUnmanaged(*const lanes.Stream, Lane) = .empty,
    drawn: [ring]u32 = undefined,
    next: u64 = 0,
    shared: u64 = 0, // forwards that held several streams' windows
    bits: ?cuda.HostBuffer = null, // structured output's grammar words (shared_rows rows), device-read, at first use
    bits_dev: u64 = 0,

    pub fn deinit(self: *Lanes) void {
        if (self.bits) |*b| b.free();
        self.lanes.deinit(self.x.e.gpa);
    }

    /// A window's grammar rows (rows of the vocabulary's words) at row `row` of the bits, device-read; 0 for none.
    fn putBits(self: *Lanes, row: usize, rows: usize, masks: []const u32) !u64 {
        if (masks.len == 0) return 0;
        const words = (self.x.e.c.vocab + 31) / 32;
        if (masks.len != rows * words or row + rows > engine.shared_rows) return error.InvalidGrammarMasks;
        if (self.bits == null) {
            var buf = try cuda.HostBuffer.allocMapped(self.x.e.d, engine.shared_rows * words * 4);
            errdefer buf.free();
            self.bits_dev = try buf.device();
            self.bits = buf;
        }
        @memcpy(self.bits.?.slice(u32)[row * words ..][0..masks.len], masks);
        return self.bits_dev + row * words * 4;
    }

    pub fn backend(self: *Lanes) be.Backend {
        return .{ .ptr = self, .vtable = &vt };
    }

    const vt: be.Backend.VTable = .{
        .prefill = prefillFn,
        .prefill_step = prefillStepFn,
        .cancel_prefill = cancelPrefillFn,
        .first = firstFn,
        .queue = queueFn,
        .read = readFn,
        .verify = verifyFn,
        .keep = keepFn,
        .draft = draftFn,
        .held = heldFn,
        .first_masked = firstMaskedFn,
        .release = releaseFn,
    };

    /// Round loop facts: copies up to max_rows, DSpark's blocks when loaded, streams' windows in one shared forward.
    pub fn facts(self: *const Lanes) lanes.Model {
        const head = self.x.e.drafter != null;
        return .{
            .exact_width = max_rows,
            .gpu_tokens = false,
            .mtp = head,
            .speculate = head, // a block drafts from the rows a verify kept (their taps)
            .speculate_early = false,
            .drafts = if (head) max_held else 0,
            .draft_streams = head,
            .mtp_step_ms = if (head) 10.5 / @as(f64, max_held) else 0, // a block's 7 drafts in one pass (10.5 ms timed)
            .window_costs = if (head) &gb10_window_costs else &.{},
            .shared_costs = if (head) &gb10_shared_costs else &.{},
            .hidden_rows = true,
            .batch_rows = engine.shared_rows,
            .max_streams = engine.max_shared,
        };
    }

    fn of(ptr: *anyopaque) *Lanes {
        return @ptrCast(@alignCast(ptr));
    }

    fn take(self: *Lanes, token: u32) u64 {
        const h = self.next;
        self.drawn[h % ring] = token;
        self.next += 1;
        return h;
    }

    /// The lowest slot no stream holds.
    fn freeSlot(self: *const Lanes) u32 {
        var slot: u32 = 0;
        while (true) : (slot += 1) {
            var it = self.lanes.valueIterator();
            const taken = while (it.next()) |l| {
                if (l.slot == slot) break true;
            } else false;
            if (!taken) return slot;
        }
    }

    fn claim(self: *Lanes, s: *const lanes.Stream) !*Lane {
        if (self.lanes.getPtr(s)) |existing| return existing;
        const slot = self.freeSlot();
        const gop = try self.lanes.getOrPut(self.x.e.gpa, s);
        gop.value_ptr.* = .{ .slot = slot };
        return gop.value_ptr;
    }

    /// The stream's sequence bound, its last verify committed (`kept` rows, else all of them).
    fn bind(self: *Lanes, s: *const lanes.Stream, kept: ?usize) !*Lane {
        const l = self.lanes.getPtr(s) orelse return error.UnknownStream;
        try self.x.bind(l.slot);
        if (l.pending_rows) |rows| try self.x.keep(kept orelse rows);
        l.pending_rows = null;
        return l;
    }

    fn cancelled(ptr: ?*anyopaque) bool {
        return @as(*lanes.Stream, @ptrCast(@alignCast(ptr.?))).isCancelled();
    }

    /// A prompt pass begins: the stream's slot bound and its sequence fresh; where it resumes (a kept state).
    fn opening(self: *Lanes, s: *lanes.Stream) !?engine.Resume {
        const ids = s.prompt();
        if (ids.len + s.max_new + max_rows >= self.x.e.capacity) return error.PromptTooLong;
        if (s.images.len > 0 and self.x.e.images == null) return error.NoVision;
        const l = try self.claim(s);
        l.pending_rows = null;
        l.held_n = 0;
        l.fill = null;
        l.first = null;
        errdefer _ = self.lanes.remove(s);
        try self.x.bind(l.slot);
        try self.x.begin(s.sampling);
        if (s.images.len > 0) try self.x.images(s.images); // every image's rows, before the chunks that hold them
        s.cached = 0;
        return if (s.prompt_logprobs != null) null else self.resumeFrom(s, ids.len);
    }

    /// The stream's kept state, where its prompt pass starts (null: none, off the chunk grid, or no longer held).
    fn resumeFrom(self: *Lanes, s: *lanes.Stream, len: usize) ?engine.Resume {
        const saved = s.reuse.saved orelse return null;
        const k: *const engine.Kept = @ptrCast(@alignCast(saved));
        if (k.at != s.reuse.at or k.at == 0 or k.at >= len or k.at % engine.prompt_rows != 0 or !self.x.e.holds(k.id)) {
            s.reuse_failed = true;
            s.reuse_error = if (k.at == 0 or k.at >= len) error.SnapshotNotPrefix else if (k.at != s.reuse.at or k.at % engine.prompt_rows != 0) error.SnapshotOutOfStep else error.SnapshotStale;
            return null;
        }
        s.cached = k.at;
        return .{ .id = k.id, .at = k.at };
    }

    /// The prompt pass stands at `at` (a chunk start): the store keeps the state there when it marked it.
    fn mark(s: *lanes.Stream, at: usize) void {
        if (at == 0) return;
        const hook = s.reuse.hook orelse return;
        if (std.mem.indexOfScalar(u32, s.reuse.marks, @intCast(at)) == null) return;
        hook.at(hook.ptr, s, @intCast(at));
    }

    fn prefillFn(ptr: *anyopaque, s: *lanes.Stream) anyerror!void {
        const self = of(ptr);
        const from = try self.opening(s);
        var at: usize = if (from) |r| r.at else 0;
        var resume_from = from;
        while (true) : (at += engine.prompt_rows) {
            if (resume_from == null) mark(s, at);
            if (try self.x.chunk(s.prompt(), at, resume_from, .{ .ptr = s, .check = cancelled })) |first| {
                self.lanes.getPtr(s).?.first = self.take(first);
                return;
            }
            resume_from = null;
        }
    }

    /// One chunk of the stream's prompt pass (other rounds run between); true once the first token is drawn.
    fn prefillStepFn(ptr: *anyopaque, s: *lanes.Stream, _: u32) anyerror!bool {
        const self = of(ptr);
        const going: ?usize = if (self.lanes.getPtr(s)) |l| l.fill else null;
        const from = if (going == null) try self.opening(s) else null;
        const l = self.lanes.getPtr(s).?;
        if (going != null) try self.x.bind(l.slot);
        const at = going orelse if (from) |r| r.at else 0;
        errdefer l.fill = null;
        if (from == null) mark(s, at);
        if (try self.x.chunk(s.prompt(), at, from, .{ .ptr = s, .check = cancelled })) |first| {
            l.first = self.take(first);
            l.fill = null;
            return true;
        }
        l.fill = at + engine.prompt_rows;
        return false;
    }

    fn cancelPrefillFn(ptr: *anyopaque, s: *lanes.Stream) !void {
        const self = of(ptr);
        const l = self.lanes.getPtr(s) orelse return;
        l.fill = null;
    }

    fn firstFn(ptr: *anyopaque, s: *lanes.Stream, position: u64) anyerror!u64 {
        const self = of(ptr);
        if (position != s.prompt_len) return error.PositionMismatch;
        const l = self.lanes.getPtr(s) orelse return error.UnknownStream;
        return l.first orelse error.NoFirstToken;
    }

    /// The first token drawn again among `mask`'s allowed tokens (prefill drew it unmasked); a fresh handle to it.
    fn firstMaskedFn(ptr: *anyopaque, s: *lanes.Stream, position: u64, mask: []const u32) anyerror!u64 {
        const self = of(ptr);
        if (position != s.prompt_len) return error.PositionMismatch;
        const l = self.lanes.getPtr(s) orelse return error.UnknownStream;
        const bits = try self.putBits(0, 1, mask);
        if (bits == 0) return error.InvalidGrammarMasks;
        try self.x.bind(l.slot);
        return self.take(try self.x.redraw(@intCast(position), bits));
    }

    fn queueFn(ptr: *anyopaque, s: *lanes.Stream, feed: be.Feed, position: u64) anyerror!u64 {
        _ = ptr;
        _ = s;
        _ = feed;
        _ = position;
        return error.NotPipelined;
    }

    fn readFn(ptr: *anyopaque, handle: u64) anyerror!u32 {
        const self = of(ptr);
        if (handle >= self.next or self.next - handle > ring) return error.NoSuchToken;
        return self.drawn[handle % ring];
    }

    /// Each window in its own forward, or a round's windows in one (Engine.verifyShared).
    fn verifyFn(ptr: *anyopaque, windows: []const be.Window, out: []be.Verified) anyerror!void {
        const self = of(ptr);
        if (windows.len > 1) return self.verifyShared(windows, out);
        for (windows, out) |w, *o| {
            if (w.parents != null) return error.TreesNotBuilt;
            const l = try self.bind(w.stream, null);
            if (w.held > l.held_n) return error.NoHeldDrafts;
            const rows = w.rows();
            if (rows > max_rows) return error.WindowTooWide;
            const pos = self.x.e.pos();
            for (w.positions, 0..) |p, r| if (p != pos + 1 + r) return error.PositionMismatch;
            var ids: [max_rows]u32 = undefined;
            ids[0] = w.pending;
            const drafts = if (w.held > 0) l.held[0..w.held] else w.tokens;
            @memcpy(ids[1..][0..drafts.len], drafts);
            var sampled: [max_rows]u32 = undefined;
            const bits = try self.putBits(0, rows, w.masks); // before the step goes out: a refusal after it strands the ranks
            self.x.e.structured = bits != 0;
            defer self.x.e.structured = false;
            try self.x.verify(ids[0..rows], sampled[0..rows], bits);
            @memcpy(o.sampled[0..rows], sampled[0..rows]);
            @memcpy(o.drafts[0..drafts.len], drafts);
            l.pending_rows = rows;
        }
    }

    fn verifyShared(self: *Lanes, windows: []const be.Window, out: []be.Verified) !void {
        if (windows.len > engine.max_shared) return error.WindowTooWide;
        var parts: [engine.max_shared]engine.Part = undefined;
        var ids: [engine.max_shared][max_rows]u32 = undefined;
        var sampled: [engine.max_shared][max_rows]u32 = undefined;
        var lane: [engine.max_shared]*Lane = undefined;
        var row: usize = 0; // the windows' rows so far: each one's grammar rows start there in the bits
        var structured = false;
        for (windows, 0..) |w, k| {
            if (w.parents != null) return error.TreesNotBuilt;
            const l = try self.bind(w.stream, null);
            if (w.held > l.held_n) return error.NoHeldDrafts;
            const rows = w.rows();
            if (rows > max_rows) return error.WindowTooWide;
            const pos = self.x.e.pos();
            for (w.positions, 0..) |p, r| if (p != pos + 1 + r) return error.PositionMismatch;
            ids[k][0] = w.pending;
            const drafts = if (w.held > 0) l.held[0..w.held] else w.tokens;
            @memcpy(ids[k][1..][0..drafts.len], drafts);
            const bits = try self.putBits(row, rows, w.masks);
            structured = structured or bits != 0;
            parts[k] = .{ .slot = l.slot, .ids = ids[k][0..rows], .out = sampled[k][0..rows], .bits = bits };
            lane[k] = l;
            row += rows;
        }
        self.x.e.structured = structured; // one window under a grammar: the whole step draws from whole rows
        defer self.x.e.structured = false;
        try self.x.verifyShared(parts[0..windows.len]);
        self.shared += 1;
        for (out[0..windows.len], parts[0..windows.len], lane[0..windows.len]) |*o, p, l| {
            const rows = p.ids.len;
            @memcpy(o.sampled[0..rows], p.out);
            @memcpy(o.drafts[0 .. rows - 1], p.ids[1..rows]);
            l.pending_rows = rows;
        }
    }

    fn keepFn(ptr: *anyopaque, windows: []const be.Window, paths: []const []const u32) anyerror!void {
        const self = of(ptr);
        for (windows, paths) |w, path| {
            if (path.len == 0) return error.EmptyPath;
            for (path, 0..) |r, i| if (r != i) return error.TreesNotBuilt;
            _ = try self.bind(w.stream, path.len);
        }
    }

    /// DSpark: every request's kept rows committed, then one block round for them all (the ranks' block step).
    fn draftFn(ptr: *anyopaque, requests: []const be.DraftRequest) anyerror!void {
        const self = of(ptr);
        if (self.x.e.drafter == null) return error.NoDraftHead;
        if (requests.len > engine.max_shared) return error.TooManyStreams;
        var reqs: [engine.max_shared]engine.Req = undefined;
        var held: [engine.max_shared]*Lane = undefined;
        for (requests, 0..) |r, i| {
            const l = try self.bind(r.stream, if (r.rows) |rows| rows.len else null);
            if (r.position != self.x.e.pos() + 1) return error.PositionMismatch;
            if (r.lanes != null) return error.TreesNotBuilt;
            var n: usize = 0;
            const anchor = if (r.rows) |rows| a: {
                for (rows, 0..) |row, j| if (row != j) return error.TreesNotBuilt;
                n = rows.len;
                break :a r.follow[rows.len - 1];
            } else self.value(r.first orelse return error.NoFirstToken);
            const depth = @min(r.depth, max_held);
            reqs[i] = .{ .slot = l.slot, .n = @intCast(n), .anchor = anchor, .out = l.held[0..depth] };
            held[i] = l;
        }
        try self.x.block(reqs[0..requests.len]);
        for (requests, held[0..requests.len]) |r, l| l.held_n = @min(r.depth, max_held);
    }

    fn value(self: *const Lanes, feed: be.Feed) u32 {
        return switch (feed) {
            .handle => |h| self.drawn[h % ring],
            .value => |v| v,
        };
    }

    /// The drafter's block for the stream's next window, as held.
    fn heldFn(ptr: *anyopaque, s: *lanes.Stream, out: []u32) anyerror!void {
        const l = of(ptr).lanes.getPtr(s) orelse return error.UnknownStream;
        if (out.len > l.held_n) return error.NoHeldDrafts;
        @memcpy(out, l.held[0..out.len]);
    }

    /// The stream left the rounds: its sequence is freed on every rank (slot 0 stays).
    fn releaseFn(ptr: *anyopaque, s: *lanes.Stream) void {
        const self = of(ptr);
        const kv = self.lanes.fetchRemove(s) orelse return;
        self.x.release(kv.value.slot) catch |err| std.log.err("releasing slot {d}: {s}", .{ kv.value.slot, @errorName(err) });
        if (self.lanes.count() == 0) self.x.idle() catch |err| std.log.err("the other ranks' idle step: {s}", .{@errorName(err)});
    }
};
