//! Nemotron CUDA streams own their sequences and share window rounds.

const std = @import("std");
const cuda = @import("cuda");
const lanes = @import("lanes");
const core = @import("core");
const engine = @import("cuda_engine.zig");
const Engine = engine.Engine;
const Head = @import("cuda_mtp.zig").Head;
const state = @import("cuda_state.zig");
const config = @import("config.zig");
const costs = @import("cuda_costs.zig");

const be = lanes.backend;

/// First tokens a handle names (the prompt's draw is on the host once prefill returns).
const ring = 1024;

/// A stream's sequence; `own` is the engine's, whose buffers the graphs were captured on (one stream at a time).
const Lane = struct { seq: *state.Seq, own: bool = false, pending_rows: ?usize = null };

pub const Cuda = struct {
    gpa: std.mem.Allocator,
    e: *Engine,
    head: ?*Head,
    lanes: std.AutoHashMapUnmanaged(*const lanes.Stream, Lane) = .empty,
    own_free: bool = true, // no stream holds the engine's own sequence
    drawn: [ring]u32 = undefined,
    next: u64 = 0,
    pinned: cuda.HostBuffer, // each window's held drafts read back (state.max_rows words a stream)
    costs: [state.max_rows]lanes.config.Cost = undefined,
    cost_count: usize = 0,
    mtp_ms: f64 = 0,
    measured: ?core.draft_depth.Costs = null, // a lone stream's depth rule prices its rounds by these

    pub fn init(gpa: std.mem.Allocator, e: *Engine, head: ?*Head) !Cuda {
        return .{ .gpa = gpa, .e = e, .head = head, .pinned = try cuda.HostBuffer.alloc(e.ctx.d, engine.max_streams * state.max_rows * 4) };
    }

    pub fn deinit(self: *Cuda) void {
        var it = self.lanes.valueIterator();
        while (it.next()) |l| if (!l.own) self.e.freeSeq(l.seq);
        self.lanes.deinit(self.gpa);
        self.pinned.free();
    }

    pub fn backend(self: *Cuda) be.Backend {
        return .{ .ptr = self, .vtable = &.{
            .prefill = prefillFn,
            .first = firstFn,
            .queue = queueFn,
            .read = readFn,
            .verify = verifyFn,
            .keep = keepFn,
            .draft = draftFn,
            .release = releaseFn,
        } };
    }

    /// The facts the round loop reads at setup: windows up to 16 rows, one stream a forward, this GPU's costs.
    pub fn facts(self: *const Cuda) lanes.Model {
        const drafting = self.head != null;
        return .{
            .exact_width = if (drafting) state.max_rows else 1,
            .gpu_tokens = false,
            .mtp = drafting,
            .speculate = drafting,
            .speculate_early = false,
            .draft_prior = &config.draft_prior,
            .drafts = @import("cuda_mtp.zig").max_chain,
            .window_costs = self.costs[0..self.cost_count],
            .mtp_step_ms = self.mtp_ms,
            .hidden_rows = drafting, // several streams' windows share a forward (Engine.verifyShared)
            .batch_rows = state.max_rows,
            .max_streams = if (drafting) engine.max_streams else 1,
        };
    }

    /// costs.measure on the engine's own sequence: windows of 1 to 16 rows and a head level, in ms.
    pub fn measure(self: *Cuda, io: std.Io, model_dir: []const u8) !void {
        const h = self.head orelse return;
        self.e.bind(&self.e.own);
        const c = try costs.measure(self.gpa, io, self.e, h, model_dir);
        self.cost_count = 0;
        for (1..c.rows + 1) |w| {
            self.costs[self.cost_count] = .{ .width = @intCast(w), .ms = c.verify[w] };
            self.cost_count += 1;
        }
        self.mtp_ms = c.level;
        self.measured = c;
    }

    /// A lone driver's hand-over: the head absorbs the last `kept` rows; the stream stands as a lane round leaves it.
    pub fn handOver(self: *Cuda, s: *lanes.Stream, kept: usize) !void {
        const l = self.lanes.getPtr(s) orelse return error.UnknownStream;
        l.pending_rows = null;
        if (self.head) |h| {
            try h.begin(kept);
            try h.launch(0);
        }
        s.cache_len = self.e.pos;
        s.pending = s.context.items[s.context.items.len - 1];
    }

    /// Bind the lane and commit its kept rows, accounting for shared-round drafting.
    fn bindLane(self: *Cuda, s: *const lanes.Stream, kept: ?usize) !*Lane {
        const l = self.lanes.getPtr(s) orelse return error.UnknownStream;
        self.e.bind(l.seq);
        if (l.pending_rows) |rows| try self.e.commit(kept orelse rows);
        l.pending_rows = null;
        return l;
    }

    fn take(self: *Cuda, token: u32) u64 {
        const h = self.next;
        self.drawn[h % ring] = token;
        self.next += 1;
        return h;
    }

    fn value(self: *const Cuda, feed: be.Feed) u32 {
        return switch (feed) {
            .handle => |h| self.drawn[h % ring],
            .value => |v| v,
        };
    }

    fn of(ptr: *anyopaque) *Cuda {
        return @ptrCast(@alignCast(ptr));
    }

    fn cancelled(ptr: *anyopaque) bool {
        return @as(*lanes.Stream, @ptrCast(@alignCast(ptr))).isCancelled();
    }

    // -- the vtable ---------------------------------------------------------------------------------------------

    /// A new sequence for the stream, its sampling, then its prompt in chunks; the head absorbs every row but the last.
    fn prefillFn(ptr: *anyopaque, s: *lanes.Stream) anyerror!void {
        const self = of(ptr);
        const e = self.e;
        const ids = s.prompt();
        if (ids.len == 0 or ids.len + s.max_new + state.max_rows > e.max_len) return error.PromptTooLong;
        const gop = try self.lanes.getOrPut(self.gpa, s);
        if (gop.found_existing) self.drop(gop.value_ptr.*);
        // the engine's own sequence while it is free: its rounds replay the captured graphs
        const own = self.own_free and e.serial != null;
        gop.value_ptr.* = .{ .own = own, .seq = if (own) &e.own else e.newSeq() catch |err| {
            self.lanes.removeByPtr(gop.key_ptr);
            return err;
        } };
        if (own) self.own_free = false;
        e.bind(gop.value_ptr.seq);
        try e.setSampling(s.sampling);
        const first = try e.prefillWith(ids, null, self.head, .{ .ptr = s, .check = cancelled });
        // the head's first draft reads the prompt's last row (its hidden row waits where a window's would)
        const last = (ids.len - 1) % state.prefill_rows;
        try e.ops().copy(e.b.hidden, e.b.p_hidden + last * @as(u64, e.c.hidden) * 2, e.c.hidden * 2);
        _ = self.take(first);
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

    /// A stream window contains its pending token and position-keyed device or host drafts.
    fn verifyFn(ptr: *anyopaque, windows: []const be.Window, out: []be.Verified) anyerror!void {
        const self = of(ptr);
        if (windows.len > 1) return self.verifyShared(windows, out);
        const w = windows[0];
        if (w.parents != null) return error.TreesNotBuilt;
        const l = try self.bindLane(w.stream, null);
        const e = self.e;
        const rows = w.rows();
        if (rows > state.max_rows) return error.WindowTooWide;
        for (w.positions, 0..) |p, r| if (p != e.pos + 1 + r) return error.PositionMismatch;
        var ids: [state.max_rows]u32 = undefined;
        ids[0] = w.pending;
        @memcpy(ids[1..][0..w.tokens.len], w.tokens);
        try e.verify(ids[0 .. 1 + w.tokens.len], rows, null);
        const held = self.pinned.slice(u32)[0 .. rows - 1];
        if (w.held > 0) try e.ops().download(std.mem.sliceAsBytes(held), e.b.ids + 4);
        try e.stream.synchronize();
        @memcpy(out[0].sampled, try e.tokens());
        @memcpy(out[0].drafts, if (w.held > 0) held else w.tokens);
        l.pending_rows = rows;
    }

    /// Several windows in one forward (Engine.verifyShared), each stream's rows on its own sequence.
    fn verifyShared(self: *Cuda, windows: []const be.Window, out: []be.Verified) !void {
        const e = self.e;
        if (windows.len > engine.max_streams) return error.WindowTooWide;
        var parts: [engine.max_streams]engine.Shared = undefined;
        var ids: [engine.max_streams][state.max_rows]u32 = undefined;
        for (windows, 0..) |w, k| {
            if (w.parents != null) return error.TreesNotBuilt;
            const l = try self.bindLane(w.stream, null);
            for (w.positions, 0..) |p, r| if (p != e.pos + 1 + r) return error.PositionMismatch;
            ids[k][0] = w.pending;
            @memcpy(ids[k][1..][0..w.tokens.len], w.tokens);
            parts[k] = .{ .seq = l.seq, .ids = ids[k][0 .. 1 + w.tokens.len], .rows = w.rows() };
        }
        try e.verifyShared(parts[0..windows.len]);
        const held = self.pinned.slice(u32);
        for (windows, 0..) |w, k| if (w.held > 0) {
            e.bind(parts[k].seq);
            try e.ops().download(std.mem.sliceAsBytes(held[k * state.max_rows ..][0 .. w.rows() - 1]), e.b.ids + 4);
        };
        try e.stream.synchronize();
        for (windows, out, 0..) |w, *o, k| {
            @memcpy(o.sampled, try e.sharedTokens(k, w.rows()));
            @memcpy(o.drafts, if (w.held > 0) held[k * state.max_rows ..][0 .. w.rows() - 1] else w.tokens);
            self.lanes.getPtr(w.stream).?.pending_rows = w.rows();
        }
    }

    /// Each window keeps its path's rows (a shared round's draft requests may have committed them already).
    fn keepFn(ptr: *anyopaque, windows: []const be.Window, paths: []const []const u32) anyerror!void {
        const self = of(ptr);
        for (windows, paths) |w, path| {
            if (path.len == 0) return error.EmptyPath;
            for (path, 0..) |r, i| if (r != i) return error.TreesNotBuilt;
            _ = try self.bindLane(w.stream, path.len);
        }
    }

    /// Absorb kept rows with their following tokens, then draft the requested depth.
    fn draftFn(ptr: *anyopaque, requests: []const be.DraftRequest) anyerror!void {
        const self = of(ptr);
        const h = self.head orelse return error.NoDraftHead;
        for (requests) |r| {
            _ = try self.bindLane(r.stream, if (r.rows) |path| path.len else null);
            if (r.position != self.e.pos + 1) return error.PositionMismatch;
            if (r.rows) |rows| {
                for (rows, 0..) |row, i| if (row != i) return error.TreesNotBuilt;
                try h.chain(r.follow[0..rows.len], r.depth);
            } else try h.chain(&.{self.value(r.first orelse return error.NoFirstToken)}, r.depth);
        }
    }

    fn releaseFn(ptr: *anyopaque, s: *lanes.Stream) void {
        const self = of(ptr);
        const kv = self.lanes.fetchRemove(s) orelse return;
        self.e.stream.synchronize() catch {};
        self.drop(kv.value);
    }

    fn drop(self: *Cuda, l: Lane) void {
        if (l.own) self.own_free = true else self.e.freeSeq(l.seq);
    }
};
