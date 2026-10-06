//! The lane core's CUDA backend for Nemotron: each stream has its own sequence on the GPU; a round runs one stream's window.

const std = @import("std");
const cuda = @import("cuda");
const lanes = @import("lanes");
const Engine = @import("cuda_engine.zig").Engine;
const Head = @import("cuda_mtp.zig").Head;
const state = @import("cuda_state.zig");
const config = @import("config.zig");
const costs = @import("cuda_costs.zig");

const be = lanes.backend;

/// First tokens a handle names (the prompt's draw is on the host once prefill returns).
const ring = 1024;

const Lane = struct { seq: *state.Seq, pending_rows: ?usize = null };

pub const Cuda = struct {
    gpa: std.mem.Allocator,
    e: *Engine,
    head: ?*Head,
    lanes: std.AutoHashMapUnmanaged(*const lanes.Stream, Lane) = .empty,
    drawn: [ring]u32 = undefined,
    next: u64 = 0,
    pinned: cuda.HostBuffer, // a window's held drafts read back
    costs: [state.max_rows]lanes.config.Cost = undefined,
    cost_count: usize = 0,
    mtp_ms: f64 = 0,

    pub fn init(gpa: std.mem.Allocator, e: *Engine, head: ?*Head) !Cuda {
        return .{ .gpa = gpa, .e = e, .head = head, .pinned = try cuda.HostBuffer.alloc(e.ctx.d, state.max_rows * 4) };
    }

    pub fn deinit(self: *Cuda) void {
        var it = self.lanes.valueIterator();
        while (it.next()) |l| self.e.freeSeq(l.seq);
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
            .batch_rows = state.max_rows,
            .max_streams = 1,
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
    }

    /// The stream's lane, bound, with a verify whose rows all stayed committed (the round loop keeps only on drops).
    fn bindLane(self: *Cuda, s: *const lanes.Stream) !*Lane {
        const l = self.lanes.getPtr(s) orelse return error.UnknownStream;
        self.e.bind(l.seq);
        if (l.pending_rows) |rows| try self.e.commit(rows);
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
        if (gop.found_existing) e.freeSeq(gop.value_ptr.seq);
        gop.value_ptr.* = .{ .seq = e.newSeq() catch |err| {
            self.lanes.removeByPtr(gop.key_ptr);
            return err;
        } };
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

    /// The stream's window: its pending token, then its held drafts (on the device) or the host's, each row keyed at its position.
    fn verifyFn(ptr: *anyopaque, windows: []const be.Window, out: []be.Verified) anyerror!void {
        const self = of(ptr);
        if (windows.len != 1) return error.SharedRoundsNotBuilt;
        const w = windows[0];
        if (w.parents != null) return error.TreesNotBuilt;
        const l = try self.bindLane(w.stream);
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

    fn keepFn(ptr: *anyopaque, windows: []const be.Window, paths: []const []const u32) anyerror!void {
        const self = of(ptr);
        if (windows.len != 1 or paths[0].len == 0) return error.SharedRoundsNotBuilt;
        for (paths[0], 0..) |r, i| if (r != i) return error.TreesNotBuilt;
        const l = self.lanes.getPtr(windows[0].stream) orelse return error.UnknownStream;
        self.e.bind(l.seq);
        l.pending_rows = null;
        try self.e.commit(paths[0].len);
    }

    /// The head absorbs the kept rows with the token after each (after a prompt: its last row and first token), then drafts `depth`.
    fn draftFn(ptr: *anyopaque, requests: []const be.DraftRequest) anyerror!void {
        const self = of(ptr);
        const h = self.head orelse return error.NoDraftHead;
        for (requests) |r| {
            _ = try self.bindLane(r.stream);
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
        self.e.freeSeq(kv.value.seq);
    }
};
