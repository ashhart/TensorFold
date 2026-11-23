//! Inkling 2's lane backend: a slot per stream, decode rows shared across streams in one forward, chunked prompts.
const std = @import("std");
const lanes = @import("lanes");
const engine = @import("engine.zig");
const follow = @import("follow.zig");
const image = @import("image.zig");
const pictures = @import("pictures.zig");

const be = lanes.backend;

/// First tokens a handle names.
const ring = 1024;
pub const max_rows = engine.window_rows;

/// A stream's place in the backend: its slot and its first token.
const Lane = struct {
    slot: u32,
    first: ?u64 = null,
    fill: ?usize = null, // a prompt pass going a chunk at a time: where its next chunk starts
    pictures: bool = false, // an image prompt: never resumed from or kept in the prompt cache
};

pub const Lanes = struct {
    x: follow.Leader,
    lanes: std.AutoHashMapUnmanaged(*const lanes.Stream, Lane) = .empty,
    drawn: [ring]u32 = undefined,
    next: u64 = 0,
    shared: u64 = 0, // forwards that held several streams' windows

    pub fn deinit(self: *Lanes) void {
        self.lanes.deinit(self.x.e.gpa);
        self.x.deinit();
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
        .release = releaseFn,
    };

    /// Round facts: one row a stream, no drafts, several streams' rows in one shared forward.
    pub fn facts(self: *const Lanes) lanes.Model {
        _ = self;
        return .{
            .copy_drafts = false,
            .exact_width = max_rows,
            .gpu_tokens = false,
            .speculate = false,
            .speculate_early = false,
            .drafts = 0,
            .plain_share = true, // one-row windows of several streams share a forward
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

    fn taken(self: *const Lanes, slot: usize) bool {
        var it = self.lanes.valueIterator();
        while (it.next()) |l| if (l.slot == slot) return true;
        return false;
    }

    /// A free slot: the kept state's when it is free, else the free one holding the fewest kept states.
    fn pick(self: *const Lanes, s: *const lanes.Stream) !u32 {
        if (s.reuse.saved) |saved| {
            const k: *const follow.Kept = @ptrCast(@alignCast(saved));
            if (self.x.e.states.where(k.id)) |w| if (!self.taken(w.slot)) return @intCast(w.slot);
        }
        var best: ?u32 = null;
        var held: usize = std.math.maxInt(usize);
        for (0..self.x.e.slotCount()) |i| {
            if (self.taken(i)) continue;
            const n = self.x.e.states.on(i);
            if (n < held) {
                held = n;
                best = @intCast(i);
            }
        }
        return best orelse error.NoFreeSlot;
    }

    /// The stream's slot bound.
    fn bind(self: *Lanes, s: *const lanes.Stream) !*Lane {
        const l = self.lanes.getPtr(s) orelse return error.UnknownStream;
        try self.x.bind(l.slot);
        return l;
    }

    fn cancelled(ptr: ?*anyopaque) bool {
        return @as(*lanes.Stream, @ptrCast(@alignCast(ptr.?))).isCancelled();
    }

    /// A prompt pass begins: the stream's slot claimed and bound; where it resumes (a kept state in this slot).
    fn opening(self: *Lanes, s: *lanes.Stream) !?follow.Kept {
        const ids = s.prompt();
        if (ids.len + s.max_new + max_rows >= self.x.e.capacity) return error.PromptTooLong;
        if (s.images.len > 0 and self.x.e.vision == null) return error.NoVision;
        const gop = try self.lanes.getOrPut(self.x.e.gpa, s);
        if (!gop.found_existing) gop.value_ptr.* = .{ .slot = try self.pick(s) };
        const l = gop.value_ptr;
        l.fill = null;
        l.first = null;
        l.pictures = s.images.len > 0;
        errdefer _ = self.lanes.remove(s);
        try self.x.bind(l.slot);
        try self.x.begin(s.sampling);
        s.cached = 0;
        if (l.pictures) {
            // every picture shares the placeholder id, so a kept state cannot tell two pictures apart
            var pics: [64]image.Prepared = undefined;
            const spans = try pictures.fromRequest(self.x.e.gpa, ids, s.images, &pics);
            defer self.x.e.gpa.free(spans);
            try self.x.images(pics[0..s.images.len], spans);
            return null;
        }
        return if (s.prompt_logprobs != null) null else self.resumeFrom(s, ids.len, l.slot);
    }

    /// The stream's kept state, where its pass starts (null: none, off the grid, another slot's, or overwritten).
    fn resumeFrom(self: *Lanes, s: *lanes.Stream, len: usize, slot: u32) ?follow.Kept {
        const saved = s.reuse.saved orelse return null;
        const k: *const follow.Kept = @ptrCast(@alignCast(saved));
        const w = self.x.e.states.where(k.id);
        if (k.at != s.reuse.at or k.at == 0 or k.at >= len or k.at % engine.prompt_rows != 0 or w == null or w.?.slot != slot) {
            s.reuse_failed = true;
            s.reuse_error = if (k.at == 0 or k.at >= len) error.SnapshotNotPrefix else if (k.at != s.reuse.at or k.at % engine.prompt_rows != 0) error.SnapshotOutOfStep else error.SnapshotStale;
            return null;
        }
        s.cached = k.at;
        return k.*;
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
        var resume_id: ?u64 = if (from) |r| r.id else null;
        const keeps = !self.lanes.getPtr(s).?.pictures;
        while (true) : (at += engine.prompt_rows) {
            if (resume_id == null and keeps) mark(s, at);
            if (try self.x.chunk(s.prompt(), at, resume_id, .{ .ptr = s, .check = cancelled })) |first| {
                self.lanes.getPtr(s).?.first = self.take(first);
                return;
            }
            resume_id = null;
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
        if (from == null and !l.pictures) mark(s, at);
        if (try self.x.chunk(s.prompt(), at, if (from) |r| r.id else null, .{ .ptr = s, .check = cancelled })) |first| {
            l.first = self.take(first);
            l.fill = null;
            return true;
        }
        l.fill = at + engine.prompt_rows;
        return false;
    }

    fn cancelPrefillFn(ptr: *anyopaque, s: *lanes.Stream) !void {
        const l = of(ptr).lanes.getPtr(s) orelse return;
        l.fill = null;
    }

    fn firstFn(ptr: *anyopaque, s: *lanes.Stream, position: u64) anyerror!u64 {
        if (position != s.prompt_len) return error.PositionMismatch;
        const l = of(ptr).lanes.getPtr(s) orelse return error.UnknownStream;
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

    /// A round's one-row windows in one forward (Engine.verifyShared), one stream or several.
    fn verifyFn(ptr: *anyopaque, windows: []const be.Window, out: []be.Verified) anyerror!void {
        const self = of(ptr);
        if (windows.len > engine.max_shared) return error.WindowTooWide;
        var parts: [engine.max_shared]engine.Part = undefined;
        var ids: [engine.max_shared][max_rows]u32 = undefined;
        var sampled: [engine.max_shared][max_rows]u32 = undefined;
        for (windows, 0..) |w, k| {
            if (w.parents != null) return error.TreesNotBuilt;
            if (w.masks.len > 0) return error.Inkling2NoGrammar;
            const l = try self.bind(w.stream);
            if (w.held > 0) return error.NoHeldDrafts;
            const rows = w.rows();
            if (rows > max_rows) return error.WindowTooWide;
            const pos = self.x.e.pos();
            for (w.positions, 0..) |p, r| if (p != pos + 1 + r) return error.PositionMismatch;
            ids[k][0] = w.pending;
            parts[k] = .{ .slot = l.slot, .ids = ids[k][0..rows], .out = sampled[k][0..rows] };
        }
        try self.x.verifyShared(parts[0..windows.len]);
        if (windows.len > 1) self.shared += 1;
        for (out[0..windows.len], parts[0..windows.len]) |*o, p| @memcpy(o.sampled[0..p.ids.len], p.out);
    }

    /// A one-row window is committed by its verify: the only path keeps its row.
    fn keepFn(ptr: *anyopaque, windows: []const be.Window, paths: []const []const u32) anyerror!void {
        _ = ptr;
        for (windows, paths) |w, path| if (path.len != 1 or path[0] != 0 or w.rows() != 1) return error.TreesNotBuilt;
    }

    /// No drafter runs here: rounds hold no drafts.
    fn draftFn(ptr: *anyopaque, requests: []const be.DraftRequest) anyerror!void {
        _ = ptr;
        if (requests.len > 0) return error.NoDraftHead;
    }

    /// The stream left the rounds: its slot is free (its rows stay for the states kept there).
    fn releaseFn(ptr: *anyopaque, s: *lanes.Stream) void {
        const self = of(ptr);
        const kv = self.lanes.fetchRemove(s) orelse return;
        self.x.release(kv.value.slot) catch |err| std.log.err("releasing slot {d}: {s}", .{ kv.value.slot, @errorName(err) });
        if (self.lanes.count() == 0) self.x.idle() catch |err| std.log.err("the other ranks' idle step: {s}", .{@errorName(err)});
    }
};
