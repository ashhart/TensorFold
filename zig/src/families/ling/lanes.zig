//! Ling-3.1-flash's CUDA lane backend (as GLM's cuda_lanes.zig, one process): a slot per stream, shared verify.
const std = @import("std");
const lanes = @import("lanes");
const engine = @import("engine.zig");

const be = lanes.backend;
const Engine = engine.Engine;

/// First tokens a handle names.
const ring = 1024;

/// The widest verify window: the pending token and 15 drafts (copies or the MTP layer's chain).
pub const max_rows = engine.window_rows;

/// A stream's place: its sequence's slot, the rows its last verify left uncommitted, where its prompt pass stands.
const Lane = struct {
    slot: u32,
    pending_rows: ?usize = null,
    held: [engine.max_drafts]u32 = undefined, // the MTP layer's drafts for the next window
    held_n: usize = 0,
    fill: ?usize = null, // a prompt pass going a chunk at a time: where its next chunk starts
};

pub const Lanes = struct {
    e: *Engine,
    whole: bool = false, // prompt passes run whole: no other stream's round between chunks
    alone: bool = false, // each window in its own forward
    shared: u64 = 0, // forwards that held several streams' windows (checks read it)
    lanes: std.AutoHashMapUnmanaged(*const lanes.Stream, Lane) = .empty,
    drawn: [ring]u32 = undefined,
    next: u64 = 0,

    pub fn deinit(self: *Lanes) void {
        self.lanes.deinit(self.e.gpa);
    }

    pub fn backend(self: *Lanes) be.Backend {
        return .{ .ptr = self, .vtable = if (self.whole) &whole_vt else &steps_vt };
    }

    const whole_vt: be.Backend.VTable = .{
        .prefill = prefillFn,
        .first = firstFn,
        .queue = queueFn,
        .read = readFn,
        .verify = verifyFn,
        .keep = keepFn,
        .draft = draftFn,
        .release = releaseFn,
    };
    const steps_vt: be.Backend.VTable = vt: {
        var v = whole_vt;
        v.prefill_step = prefillStepFn;
        break :vt v;
    };

    /// Round loop facts: exact windows of max_rows (copies, MTP chains), shared windows unless `alone`.
    pub fn facts(self: *const Lanes) lanes.Model {
        const sharing = !self.alone;
        const drafting = self.e.mtp != null;
        return .{
            .exact_width = max_rows,
            .gpu_tokens = false,
            .mtp = drafting,
            .speculate = drafting,
            .drafts = engine.max_drafts,
            .speculate_early = false,
            .hidden_rows = sharing,
            .batch_rows = if (sharing) engine.shared_rows else 0,
            .max_streams = if (sharing) engine.max_shared else 1,
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

    /// The stream's lane; a new one takes the lowest free slot, chosen before the map holds its (unset) entry.
    pub fn claim(self: *Lanes, gpa: std.mem.Allocator, s: *const lanes.Stream) !*Lane {
        if (self.lanes.getPtr(s)) |l| return l;
        const slot = self.freeSlot();
        try self.lanes.put(gpa, s, .{ .slot = slot });
        return self.lanes.getPtr(s).?;
    }

    /// The stream's sequence bound, its last verify committed (`kept` rows, else all of them).
    fn bind(self: *Lanes, s: *const lanes.Stream, kept: ?usize) !*Lane {
        const l = self.lanes.getPtr(s) orelse return error.UnknownStream;
        try self.e.bind(l.slot);
        if (l.pending_rows) |rows| try self.e.keep(kept orelse rows);
        l.pending_rows = null;
        return l;
    }

    fn cancelled(ptr: ?*anyopaque) bool {
        return @as(*lanes.Stream, @ptrCast(@alignCast(ptr.?))).isCancelled();
    }

    fn cancelOf(s: *lanes.Stream) engine.Cancel {
        return .{ .ptr = s, .check = cancelled };
    }

    fn prefillFn(ptr: *anyopaque, s: *lanes.Stream) anyerror!void {
        const self = of(ptr);
        const from = try self.opening(s);
        _ = self.take(try self.e.prefill(s.prompt(), from, cancelOf(s), marksOf(s)));
    }

    /// One chunk of the stream's prompt pass (fillOne runs other rounds between); true once the first token is drawn.
    fn prefillStepFn(ptr: *anyopaque, s: *lanes.Stream) anyerror!bool {
        const self = of(ptr);
        const going: ?usize = if (self.lanes.getPtr(s)) |l| l.fill else null;
        const at = going orelse try self.e.prefillStart(s.prompt().len, try self.opening(s));
        const l = self.lanes.getPtr(s).?;
        try self.e.bind(l.slot);
        errdefer l.fill = null;
        const step = try self.e.prefillChunk(s.prompt(), at, cancelOf(s), marksOf(s));
        if (step.first) |first| {
            _ = self.take(first);
            l.fill = null;
            return true;
        }
        l.fill = step.at;
        return false;
    }

    /// A prompt pass begins: the stream's slot bound and its sequence fresh; where the pass resumes (a kept state).
    fn opening(self: *Lanes, s: *lanes.Stream) !?engine.Resume {
        const ids = s.prompt();
        if (ids.len + s.max_new + max_rows >= self.e.capacity) return error.PromptTooLong;
        const l = try self.claim(self.e.gpa, s);
        l.pending_rows = null;
        l.held_n = 0;
        l.fill = null;
        try self.e.bind(l.slot);
        try self.e.begin(s.sampling);
        s.cached = 0;
        return self.resumeFrom(s, ids.len);
    }

    fn marksOf(s: *lanes.Stream) ?engine.Marks {
        return if (s.reuse.marks.len == 0) null else .{ .at = s.reuse.marks, .ctx = s, .call = keptAt };
    }

    /// The stream's kept state, where its prompt pass starts (null: none, or no longer held).
    fn resumeFrom(self: *Lanes, s: *lanes.Stream, len: usize) ?engine.Resume {
        const saved = s.reuse.saved orelse return null;
        const k: *const engine.Kept = @ptrCast(@alignCast(saved));
        if (k.at != s.reuse.at or k.at == 0 or k.at >= len or !self.e.holds(k.id)) {
            s.reuse_failed = true;
            return null;
        }
        s.cached = k.at;
        return .{ .id = k.id, .at = k.at };
    }

    /// The prompt pass stands at a mark: the lane host keeps the state there (native.snap_save).
    fn keptAt(ctx: *anyopaque, at: u32) void {
        const s: *lanes.Stream = @ptrCast(@alignCast(ctx));
        if (s.reuse.hook) |k| k.at(k.ptr, s, at);
    }

    fn firstFn(ptr: *anyopaque, s: *lanes.Stream, position: u64) anyerror!u64 {
        const self = of(ptr);
        if (position != s.prompt_len) return error.PositionMismatch;
        return self.next - 1;
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

    /// A window's ids (pending token, then its drafts) checked against the bound sequence's position.
    fn windowIds(self: *Lanes, l: *const Lane, w: be.Window, ids: *[max_rows]u32) ![]const u32 {
        if (w.parents != null) return error.TreesNotBuilt;
        const rows = w.rows();
        if (rows > max_rows) return error.WindowTooWide;
        if (w.held > l.held_n) return error.NoHeldDrafts;
        const pos = self.e.pos();
        for (w.positions, 0..) |p, r| if (p != pos + 1 + r) return error.PositionMismatch;
        const drafts = if (w.held > 0) l.held[0..w.held] else w.tokens;
        ids[0] = w.pending;
        @memcpy(ids[1..][0..drafts.len], drafts);
        return ids[0..rows];
    }

    /// Each window in its own forward on its stream's sequence (the core sends one when no forward is shared).
    fn verifyFn(ptr: *anyopaque, windows: []const be.Window, out: []be.Verified) anyerror!void {
        const self = of(ptr);
        if (windows.len > 1) return self.verifyShared(windows, out);
        for (windows, out) |w, *o| {
            const l = try self.bind(w.stream, null);
            var ids: [max_rows]u32 = undefined;
            const got = try self.windowIds(l, w, &ids);
            try self.e.verify(got, o.sampled[0..got.len]);
            @memcpy(o.drafts[0 .. got.len - 1], got[1..]);
            l.pending_rows = got.len;
        }
    }

    /// A round's windows in one forward (Engine.verifyShared), each stream's rows on its own sequence.
    fn verifyShared(self: *Lanes, windows: []const be.Window, out: []be.Verified) !void {
        if (windows.len > engine.max_shared) return error.WindowTooWide;
        var parts: [engine.max_shared]engine.Part = undefined;
        var ids: [engine.max_shared][max_rows]u32 = undefined;
        var lane: [engine.max_shared]*Lane = undefined;
        for (windows, out[0..windows.len], 0..) |w, o, k| {
            lane[k] = try self.bind(w.stream, null);
            const got = try self.windowIds(lane[k], w, &ids[k]);
            parts[k] = .{ .slot = lane[k].slot, .ids = got, .out = o.sampled[0..got.len] };
        }
        try self.e.verifyShared(parts[0..windows.len]);
        self.shared += 1;
        for (out[0..windows.len], parts[0..windows.len], lane[0..windows.len]) |*o, p, l| {
            @memcpy(o.drafts[0 .. p.ids.len - 1], p.ids[1..]);
            l.pending_rows = p.ids.len;
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

    /// The MTP layer absorbs each stream's kept rows (its last verify committed first) and chains its drafts.
    fn draftFn(ptr: *anyopaque, requests: []const be.DraftRequest) anyerror!void {
        const self = of(ptr);
        if (self.e.mtp == null) return error.NoDraftHead;
        for (requests) |r| {
            const l = try self.bind(r.stream, if (r.rows) |rows| rows.len else null);
            if (r.position != self.e.pos() + 1) return error.PositionMismatch;
            if (r.lanes != null) return error.TreesNotBuilt;
            const depth = @min(r.depth, engine.max_drafts);
            if (r.rows) |rows| {
                for (rows, 0..) |row, i| if (row != i) return error.TreesNotBuilt;
                try self.e.draft(false, r.follow[0..rows.len], depth, &l.held);
            } else {
                const first = switch (r.first orelse return error.NoFirstToken) {
                    .handle => |h| self.drawn[h % ring],
                    .value => |v| v,
                };
                try self.e.draft(true, &.{first}, depth, &l.held);
            }
            l.held_n = depth;
        }
    }

    /// The stream left the rounds: its sequence freed (slot 0 stays for the next stream).
    fn releaseFn(ptr: *anyopaque, s: *lanes.Stream) void {
        const self = of(ptr);
        const kv = self.lanes.fetchRemove(s) orelse return;
        self.e.release(kv.value.slot);
    }
};

/// A stand-in stream key: claim hashes the address and never reads through it.
fn key(i: usize) *const lanes.Stream {
    return @ptrFromInt(i * 4096);
}

/// Slots claimed over map buckets still holding slot 0 from cleared entries (a release build leaves a new value as is).
pub fn slotCheck(gpa: std.mem.Allocator) !void {
    var ll: Lanes = .{ .e = undefined };
    defer ll.lanes.deinit(gpa);
    for (1..9) |i| try ll.lanes.put(gpa, key(i), .{ .slot = 0 });
    ll.lanes.clearRetainingCapacity();
    const want = [_]struct { k: usize, slot: u32 }{ .{ .k = 1, .slot = 0 }, .{ .k = 2, .slot = 1 }, .{ .k = 1, .slot = 0 }, .{ .k = 3, .slot = 2 } };
    for (want) |w| if ((try ll.claim(gpa, key(w.k))).slot != w.slot) return error.SlotTaken;
    if (!ll.lanes.remove(key(1))) return error.SlotTaken;
    if ((try ll.claim(gpa, key(4))).slot != 0 or (try ll.claim(gpa, key(5))).slot != 3) return error.SlotTaken;
}

test "a new stream takes the lowest free slot, a known one keeps its own, a released one's goes to the next" {
    try slotCheck(std.testing.allocator);
}
