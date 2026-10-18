//! GLM-5.3-Flash's CUDA lane backend (as nemotron/cuda_lanes.zig): a slot per stream, shared verify, chunked prompts.

const std = @import("std");
const lanes = @import("lanes");
const engine = @import("cuda_engine.zig");
const follow = @import("cuda_follow.zig");
const fw = @import("cuda_forward.zig");
const dflash = @import("dflash.zig");
const prompt_lp = @import("cuda_prompt_logprobs.zig");

const be = lanes.backend;

/// First tokens a handle names.
const ring = 1024;

/// The widest verify window: the pending token and a copy (the lane core proposes them), as the decode buffers hold.
pub const max_rows = fw.window_rows;
const mtp_rows = 1 + engine.max_drafts;
const max_held = dflash.BLOCK - 1; // a DFlash2 block's drafts (the MTP head's chains stop at max_drafts)

/// With DFlash2: a window's ms by rows on two GB10s (EXL3, TF_GLM_PROF: 86 to 107 ms at 5 to 8), shared as glm_cuda's.
const gb10_window_costs = costTable(max_rows, 52.0, 6.8);
const gb10_shared_costs = costTable(engine.shared_rows, 76.0, 4.1);

fn costTable(comptime n: usize, comptime one: f64, comptime row: f64) [n]lanes.config.Cost {
    var c: [n]lanes.config.Cost = undefined;
    for (&c, 0..) |*x, i| x.* = .{ .width = i + 1, .ms = one + row * @as(f64, @floatFromInt(i)) };
    return c;
}

/// A stream's place in the backend: its sequence's slot, the rows its last verify left uncommitted, its held drafts.
const Lane = struct {
    slot: u32,
    pending_rows: ?usize = null,
    held: [max_held]u32 = undefined,
    chances: [max_held]f64 = undefined, // DFlash2's: each held draft's chance from the selector
    held_n: usize = 0,
    first: ?u64 = null,
    fill: ?usize = null, // a prompt pass going a chunk at a time: where its next chunk starts
};

pub const Lanes = struct {
    x: follow.Leader,
    whole: bool = false, // prompt passes run whole (TF_GLM_PROMPT_STEPS=0): no other stream's round between chunks
    alone: bool = false, // each window in its own forward (TF_GLM_SHARED=0)
    shared: u64 = 0, // forwards that held several streams' windows (checks read it)
    lanes: std.AutoHashMapUnmanaged(*const lanes.Stream, Lane) = .empty,
    drawn: [ring]u32 = undefined,
    next: u64 = 0,

    pub fn deinit(self: *Lanes) void {
        self.lanes.deinit(self.x.e.gpa);
    }

    pub fn backend(self: *Lanes) be.Backend {
        return .{ .ptr = self, .vtable = if (self.whole) &whole_vt else &steps_vt };
    }

    const whole_vt: be.Backend.VTable = .{
        .prefill = prefillFn,
        .prefill_many = prefillManyFn,
        .first = firstFn,
        .queue = queueFn,
        .read = readFn,
        .verify = verifyFn,
        .keep = keepFn,
        .draft = draftFn,
        .probabilities = chancesFn,
        .release = releaseFn,
        .prompt_rows = true,
    };
    const steps_vt: be.Backend.VTable = vt: {
        var v = whole_vt;
        v.prefill_step = prefillStepFn;
        v.many_rows = engine.prompt_rows;
        break :vt v;
    };

    /// Round loop facts: copies up to max_rows, head chains to mtp_rows, shared windows unless TF_GLM_SHARED=0.
    pub fn facts(self: *const Lanes) lanes.Model {
        const drafting = self.x.e.mtp;
        const block = self.x.e.drafter != null;
        const sharing = drafting and !self.alone;
        return .{
            .exact_width = if (drafting) max_rows else 1,
            .first_copy_rows = mtp_rows, // a first copy as wide as a head round; it grows as its tokens land
            .gpu_tokens = false,
            .mtp = drafting,
            .speculate = drafting,
            .speculate_early = false,
            .drafts = if (block) max_held else engine.max_drafts,
            .draft_streams = block, // one block pass drafts every stream of the round
            .mtp_step_ms = if (block) 1.6 / @as(f64, max_held) else 0, // a block's 7 drafts in one pass (about 1.6 ms)
            .draft_probabilities = block, // the selector's chances size each window against the costs below
            .window_costs = if (block) &gb10_window_costs else &.{},
            .shared_costs = if (block) &gb10_shared_costs else &.{},
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

    fn value(self: *const Lanes, feed: be.Feed) u32 {
        return switch (feed) {
            .handle => |h| self.drawn[h % ring],
            .value => |v| v,
        };
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

    pub fn claim(self: *Lanes, s: *const lanes.Stream) !*Lane {
        return self.claimUsing(s, freeSlot);
    }

    pub fn claimUsing(self: *Lanes, s: *const lanes.Stream, comptime choose: anytype) !*Lane {
        if (self.lanes.getPtr(s)) |existing| return existing;
        const slot = choose(self);
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

    // -- the vtable ---------------------------------------------------------------------------------------------

    fn prefillFn(ptr: *anyopaque, s: *lanes.Stream) anyerror!void {
        const self = of(ptr);
        const from = try self.opening(s);
        try self.images(s, from);
        var sink: Sink = .{ .gpa = self.x.e.gpa, .s = s };
        try sink.prepare();
        self.x.e.prompt_logprobs = sink.want();
        defer self.x.e.prompt_logprobs = null;
        const first = try self.x.prefill(s.prompt(), from, .{ .ptr = s, .check = cancelled }, marksOf(s));
        self.lanes.getPtr(s).?.first = self.take(first);
    }

    fn prefillManyFn(ptr: *anyopaque, ss: []const *lanes.Stream) anyerror!bool {
        const self = of(ptr);
        if (!self.x.e.m.tiled or ss.len > engine.max_together) return false;
        for (ss) |s| if (!canBatch(s)) return false;
        var opened: usize = 0;
        errdefer for (ss[0..opened]) |s| releaseFn(ptr, s);
        var parts: [engine.max_together]engine.Prompt = undefined;
        var firsts: [engine.max_together]u32 = undefined;
        var lo: usize = 0;
        var rows: usize = 0;
        for (ss, 0..) |s, i| {
            if (rows + s.prompt_len > engine.prompt_rows) { // this pass is full: run it, the next starts here
                try self.many(ss[lo..i], parts[lo..i], firsts[lo..i]);
                lo = i;
                rows = 0;
            }
            if (try self.opening(s) != null) return error.BadPrompts; // no kept state (checked above)
            opened += 1;
            parts[i] = .{ .slot = self.lanes.getPtr(s).?.slot, .ids = s.prompt() };
            rows += s.prompt_len;
        }
        try self.many(ss[lo..], parts[lo..ss.len], firsts[lo..ss.len]);
        return true;
    }

    fn many(self: *Lanes, ss: []const *lanes.Stream, parts: []const engine.Prompt, firsts: []u32) !void {
        try self.x.prefillMany(parts, firsts);
        for (ss, firsts) |s, t| self.lanes.getPtr(s).?.first = self.take(t);
    }

    /// One chunk of the stream's prompt pass (fillOne runs other rounds between); true once the first token is drawn.
    fn prefillStepFn(ptr: *anyopaque, s: *lanes.Stream) anyerror!bool {
        const self = of(ptr);
        const going: ?usize = if (self.lanes.getPtr(s)) |l| l.fill else null;
        const from = if (going == null) try self.opening(s) else null;
        if (going == null) try self.images(s, from);
        const l = self.lanes.getPtr(s).?;
        if (going != null) try self.x.bind(l.slot);
        const at = going orelse if (from) |r| r.at else 0;
        errdefer l.fill = null;
        var sink: Sink = .{ .gpa = self.x.e.gpa, .s = s };
        try sink.prepare();
        self.x.e.prompt_logprobs = sink.want();
        defer self.x.e.prompt_logprobs = null;
        if (try self.x.chunk(s.prompt(), at, from, .{ .ptr = s, .check = cancelled }, marksOf(s))) |first| {
            l.first = self.take(first);
            l.fill = null;
            return true;
        }
        l.fill = at + engine.prompt_rows;
        return false;
    }

    /// A prompt pass begins: the stream's slot bound and its sequence fresh; where the pass resumes (a kept state).
    fn opening(self: *Lanes, s: *lanes.Stream) !?engine.Resume {
        const ids = s.prompt();
        if (ids.len + s.max_new + max_rows >= self.x.e.capacity) return error.PromptTooLong;
        const l = try self.claim(s);
        l.pending_rows = null;
        l.held_n = 0;
        l.fill = null;
        l.first = null;
        try self.x.bind(l.slot);
        if (self.x.e.images) |v| v.clear(); // no earlier request's rows splice into this one
        try self.x.begin(s.sampling);
        s.cached = 0;
        s.prompt_rows.clearRetainingCapacity();
        return if (s.prompt_logprobs != null) null else self.resumeFrom(s, ids.len); // prompt rows need every row
    }

    /// A prompt pass with images: rank 0's tower encodes those past `from`, then every rank holds their rows.
    fn images(self: *Lanes, s: *lanes.Stream, from: ?engine.Resume) !void {
        if (s.images.len == 0) return;
        const v = self.x.e.images orelse return error.NoVision;
        try v.begin(self.x.e.slot, s.images, if (from) |r| r.at else 0); // a refusal here leaves every rank as it was
        try self.x.images(v);
    }
    const Sink = @import("cuda_prompt_sink.zig").Sink;

    fn marksOf(s: *lanes.Stream) ?engine.Marks {
        return if (s.reuse.marks.len == 0) null else .{ .at = s.reuse.marks, .ctx = s, .call = keptAt };
    }

    /// The stream's kept state, where its prompt pass starts (null: none, off the chunk grid, or no longer held).
    fn resumeFrom(self: *Lanes, s: *lanes.Stream, len: usize) ?engine.Resume {
        const saved = s.reuse.saved orelse return null;
        const k: *const engine.Kept = @ptrCast(@alignCast(saved));
        if (k.at != s.reuse.at or k.at == 0 or k.at >= len or k.at % fw.prefill_rows != 0 or !self.x.e.holds(k.id)) {
            s.reuse_failed = true;
            s.reuse_error = if (k.at == 0 or k.at >= len) error.SnapshotNotPrefix else if (k.at != s.reuse.at or k.at % fw.prefill_rows != 0) error.SnapshotOutOfStep else error.SnapshotStale;
            return null;
        }
        s.cached = k.at;
        return .{ .id = k.id, .at = k.at };
    }

    /// The prompt pass stands at a mark: the lane host keeps the state there (cuda_native's snap_save).
    fn keptAt(ctx: *anyopaque, at: u32) void {
        const s: *lanes.Stream = @ptrCast(@alignCast(ctx));
        if (s.reuse.hook) |k| k.at(k.ptr, s, at);
    }

    fn firstFn(ptr: *anyopaque, s: *lanes.Stream, position: u64) anyerror!u64 {
        const self = of(ptr);
        if (position != s.prompt_len) return error.PositionMismatch;
        const l = self.lanes.getPtr(s) orelse return error.UnknownStream;
        return l.first orelse error.NoFirstToken;
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

    /// Each window in its own forward on its stream's sequence (the core sends one when no forward is shared).
    fn verifyFn(ptr: *anyopaque, windows: []const be.Window, out: []be.Verified) anyerror!void {
        const self = of(ptr);
        if (windows.len > 1) return self.verifyShared(windows, out);
        for (windows, out) |w, *o| {
            if (w.parents != null) return error.TreesNotBuilt;
            const l = try self.bind(w.stream, null);
            const rows = w.rows();
            if (rows > max_rows) return error.WindowTooWide;
            if (w.held > l.held_n) return error.NoHeldDrafts;
            const pos = self.x.e.pos();
            for (w.positions, 0..) |p, r| if (p != pos + 1 + r) return error.PositionMismatch;
            var ids: [max_rows]u32 = undefined;
            ids[0] = w.pending;
            const drafts = if (w.held > 0) l.held[0..w.held] else w.tokens;
            @memcpy(ids[1..][0..drafts.len], drafts);
            var sampled: [max_rows]u32 = undefined;
            try self.x.verify(ids[0..rows], sampled[0..rows]);
            @memcpy(o.sampled[0..rows], sampled[0..rows]);
            @memcpy(o.drafts[0..drafts.len], drafts);
            l.pending_rows = rows;
        }
    }

    /// A round's windows in one forward (Engine.verifyShared), each stream's rows on its own sequence.
    fn verifyShared(self: *Lanes, windows: []const be.Window, out: []be.Verified) !void {
        if (windows.len > engine.max_shared) return error.WindowTooWide;
        var parts: [engine.max_shared]engine.Part = undefined;
        var ids: [engine.max_shared][max_rows]u32 = undefined;
        var sampled: [engine.max_shared][max_rows]u32 = undefined;
        var lane: [engine.max_shared]*Lane = undefined;
        for (windows, 0..) |w, k| {
            if (w.parents != null) return error.TreesNotBuilt;
            const l = try self.bind(w.stream, null);
            const rows = w.rows();
            if (rows > max_rows) return error.WindowTooWide;
            if (w.held > l.held_n) return error.NoHeldDrafts;
            const pos = self.x.e.pos();
            for (w.positions, 0..) |p, r| if (p != pos + 1 + r) return error.PositionMismatch;
            ids[k][0] = w.pending;
            const drafts = if (w.held > 0) l.held[0..w.held] else w.tokens;
            @memcpy(ids[k][1..][0..drafts.len], drafts);
            parts[k] = .{ .slot = l.slot, .ids = ids[k][0..rows], .out = sampled[k][0..rows] };
            lane[k] = l;
        }
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

    fn draftFn(ptr: *anyopaque, requests: []const be.DraftRequest) anyerror!void {
        const self = of(ptr);
        if (self.x.e.drafter != null) return self.draftBlock(requests);
        if (!self.x.e.mtp) return error.NoDraftHead;
        for (requests) |r| {
            const l = try self.bind(r.stream, if (r.rows) |rows| rows.len else null);
            if (r.position != self.x.e.pos() + 1) return error.PositionMismatch;
            if (r.lanes != null) return error.TreesNotBuilt;
            const depth = @min(r.depth, engine.max_drafts);
            if (r.rows) |rows| {
                for (rows, 0..) |row, i| if (row != i) return error.TreesNotBuilt;
                try self.x.draft(false, r.follow[0..rows.len], depth, &l.held);
            } else {
                const first = self.value(r.first orelse return error.NoFirstToken);
                try self.x.draft(true, &.{first}, depth, &l.held);
            }
            l.held_n = depth;
        }
    }

    /// DFlash2: every request's kept rows committed, then one block round for them all (cuda_follow's block step).
    fn draftBlock(self: *Lanes, requests: []const be.DraftRequest) !void {
        if (requests.len > dflash.MAX_S) return error.TooManyStreams;
        var reqs: [dflash.MAX_S]dflash.Req = undefined;
        var held: [dflash.MAX_S]*Lane = undefined;
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
            reqs[i] = .{ .slot = l.slot, .pos = 0, .n = n, .anchor = anchor, .out = &l.held, .probs = &l.chances };
            held[i] = l;
        }
        try self.x.block(reqs[0..requests.len]);
        const cut = confidence();
        for (requests, held[0..requests.len]) |r, l| {
            l.held_n = @min(r.depth, max_held);
            var chain: f64 = 1; // glm_cuda's rule: past the first, a chain stops where its product falls under the cut
            for (l.chances[0..l.held_n], 0..) |*c, j| {
                chain *= c.*;
                if (j > 0 and chain < cut) c.* = 0; // held with chance 0: the window sizing leaves it out
                if (c.* == 0) chain = 0;
            }
        }
    }

    /// TF_GLM_DRAFT_CONFIDENCE: where a DFlash2 chain's chance product stops it (glm_cuda's 0.7).
    fn confidence() f64 {
        const v = std.c.getenv("TF_GLM_DRAFT_CONFIDENCE") orelse return 0.7;
        return std.fmt.parseFloat(f64, std.mem.span(v)) catch 0.7;
    }

    /// DFlash2's chance for each held draft (the MTP head gives none: the lane core's per-depth rates stand).
    fn chancesFn(ptr: *anyopaque, s: *lanes.Stream, out: []f64) anyerror!bool {
        const self = of(ptr);
        if (self.x.e.drafter == null) return false;
        const l = self.lanes.getPtr(s) orelse return false;
        if (l.held_n == 0) return false;
        const n = @min(out.len, l.held_n);
        var chain: f64 = 1; // a draft lands only if every one before it does: the selector's chances chained
        for (out[0..n], l.chances[0..n]) |*c, p| {
            chain *= p;
            c.* = chain;
        }
        for (out[n..]) |*c| c.* = 0;
        return true;
    }

    /// The stream left the rounds: its sequence is freed on every rank (slot 0 stays for the next stream).
    fn releaseFn(ptr: *anyopaque, s: *lanes.Stream) void {
        const self = of(ptr);
        const kv = self.lanes.fetchRemove(s) orelse return;
        self.x.release(kv.value.slot) catch |err| std.log.err("releasing slot {d}: {s}", .{ kv.value.slot, @errorName(err) });
    }
};

comptime {
    std.debug.assert(max_rows <= fw.window_rows);
}

test "a lane's slot is chosen before its map entry exists" {
    var storage: [8192]u8 = @splat(0);
    var fixed = std.heap.FixedBufferAllocator.init(&storage);
    var e: engine.Engine = undefined;
    e.gpa = fixed.allocator();
    var self: Lanes = .{ .x = .{ .e = &e, .link = undefined } };
    defer self.lanes.deinit(e.gpa);
    var streams: [3]lanes.Stream = undefined;
    const Probe = struct {
        var bad = false;
        fn choose(l: *const Lanes) u32 {
            bad = l.lanes.count() != 0;
            return 0;
        }
    };
    _ = try self.claimUsing(&streams[0], Probe.choose);
    try std.testing.expect(!Probe.bad);
    try std.testing.expectEqual(@as(u32, 0), (try self.claim(&streams[0])).slot);
    try std.testing.expectEqual(@as(u32, 1), (try self.claim(&streams[1])).slot);
    _ = self.lanes.remove(&streams[0]);
    try std.testing.expectEqual(@as(u32, 0), (try self.claim(&streams[2])).slot);
    try std.testing.expectEqual(@as(u32, 1), (try self.claim(&streams[1])).slot);
}

test "a failed lane insertion leaves no entry, and a retry takes slot zero" {
    var fail = std.testing.FailingAllocator.init(std.testing.allocator, .{ .fail_index = 0 });
    var e: engine.Engine = undefined;
    e.gpa = fail.allocator();
    var self: Lanes = .{ .x = .{ .e = &e, .link = undefined } };
    defer self.lanes.deinit(e.gpa);
    var stream: lanes.Stream = undefined;
    try std.testing.expectError(error.OutOfMemory, self.claim(&stream));
    try std.testing.expectEqual(@as(usize, 0), self.lanes.count());
    fail.fail_index = std.math.maxInt(usize);
    try std.testing.expectEqual(@as(u32, 0), (try self.claim(&stream)).slot);
}

/// Eligibility for the maintained fresh-prompt batching path.
pub fn canBatch(s: anytype) bool {
    return s.prompt_logprobs == null and s.images.len == 0 and s.reuse.saved == null and s.prompt_len <= engine.prompt_rows;
}
