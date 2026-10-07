//! GLM-5.3-Flash behind the lane core: a stream a slot, each slot operation sent to the peer Mac before it runs.
const std = @import("std");
const lanes = @import("lanes");
const st = @import("state.zig");
const slots_mod = @import("slots.zig");
const mirror = @import("mirror.zig");
const Engine = @import("engine.zig").Engine;
const be = lanes.backend;
const Stream = lanes.Stream;

/// A forward's ms by rows on the M5 Ultra pair, until the depth rule has timed this server's own rounds.
const costs = blk: {
    var c: [st.max_rows]lanes.config.Cost = undefined;
    for (&c, 0..) |*x, i| x.* = .{ .width = i + 1, .ms = 12.9 + 3.5 * @as(f64, @floatFromInt(i)) };
    break :blk c;
};

pub const Backend = struct {
    gpa: std.mem.Allocator,
    sl: *slots_mod.Slots,
    by: std.AutoHashMapUnmanaged(*Stream, u32) = .empty,
    words: std.ArrayList(u32) = .empty, // the next command's words
    wins: std.ArrayList(slots_mod.Win) = .empty,

    pub fn deinit(b: *Backend) void {
        b.by.deinit(b.gpa);
        b.words.deinit(b.gpa);
        b.wins.deinit(b.gpa);
    }

    pub fn backend(b: *Backend) be.Backend {
        return .{ .ptr = b, .vtable = &.{ .prefill = prefill, .first = first, .queue = queue, .read = read, .verify = verify, .keep = keep, .draft = draft, .release = release } };
    }

    /// Windows of up to 16 rows (drafted == plain at every width), the MTP head's chains, every stream in one forward.
    pub fn facts(b: *const Backend) lanes.Model {
        return .{
            .exact_width = st.max_rows,
            .first_copy_rows = 4,
            .mtp = b.sl.e.hasMtp(),
            .speculate = true,
            .speculate_early = false,
            .drafts = 4,
            .window_costs = &costs,
            .mtp_step_ms = 0.8,
            .streams_exact = true,
            .hidden_rows = true,
            .batch_rows = st.max_rows,
            .max_streams = @intCast(b.sl.slots.len),
            .shared_costs = &costs,
            .draft_streams = true,
        };
    }

    fn self(ptr: *anyopaque) *Backend {
        return @ptrCast(@alignCast(ptr));
    }

    fn slotOf(b: *Backend, s: *Stream) !u32 {
        return b.by.get(s) orelse error.UnknownStream;
    }

    fn prefill(ptr: *anyopaque, s: *Stream) anyerror!void {
        const b = self(ptr);
        const e = b.sl.e;
        if (e.followsPeer()) return error.FollowsPeer;
        if (s.sampling) |p| if (p.temperature > 0) return error.GreedyOnly;
        const prompt = s.prompt();
        if (prompt.len == 0) return error.EmptyPrompt;
        if (prompt.len + s.max_new + st.max_rows + 1 > e.s.cap) return error.ContextFull;
        const i = b.sl.free() orelse return error.NoFreeSlot;
        try b.by.put(b.gpa, s, i);
        b.words.clearRetainingCapacity();
        try b.words.appendSlice(b.gpa, &.{ i, @intFromBool(s.drafts) });
        try b.words.appendSlice(b.gpa, prompt);
        try mirror.send(e, .begin, b.words.items);
        try b.sl.begin(i, prompt, s.drafts);
        s.cached = 0;
        var at: u32 = 0;
        while (at < prompt.len) {
            if (s.isCancelled()) return error.Cancelled; // the core releases the slot
            const n = b.sl.chunkRows(i, at);
            try mirror.send(e, .chunk, &.{ i, at, n });
            try b.sl.chunk(i, at, n);
            at += n;
        }
    }

    fn first(ptr: *anyopaque, s: *Stream, position: u64) anyerror!u64 {
        const b = self(ptr);
        const i = try b.slotOf(s);
        if (position != b.sl.length(i)) return error.PositionMismatch;
        return Engine.u32s(b.sl.e.sc.picks, 1)[0];
    }

    fn queue(_: *anyopaque, _: *Stream, _: be.Feed, _: u64) anyerror!u64 {
        return error.Unsupported; // GLM's tokens come back to the host every round: no step runs ahead
    }

    fn read(_: *anyopaque, handle: u64) anyerror!u32 {
        return @intCast(handle);
    }

    fn verify(ptr: *anyopaque, windows: []const be.Window, out: []be.Verified) anyerror!void {
        const b = self(ptr);
        b.wins.clearRetainingCapacity();
        var total: usize = 0;
        for (windows) |w| {
            if (w.parents != null or w.early) return error.TreeWindowsUnsupported;
            const i = try b.slotOf(w.stream);
            const at = b.sl.length(i);
            for (w.positions, 0..) |p, r| if (p != at + r + 1) return error.PositionMismatch;
            if (w.held > b.sl.slots[i].held_n or at + w.rows() + 1 > b.sl.e.s.cap) return error.WindowOutOfStep;
            total += w.rows();
            try b.wins.append(b.gpa, .{ .slot = i, .pending = w.pending, .held = w.held, .tokens = w.tokens });
        }
        if (total > st.max_rows) return error.WindowOutOfStep;
        try mirror.windowWords(&b.words, b.gpa, b.sl.digest, b.wins.items);
        try mirror.send(b.sl.e, .window, b.words.items);
        try b.sl.window(b.wins.items);
        const picks = Engine.u32s(b.sl.e.sc.picks, total);
        var row: usize = 0;
        for (windows, out, b.wins.items) |w, o, win| {
            @memcpy(o.sampled, picks[row..][0..w.rows()]);
            @memcpy(o.drafts[0..w.held], Engine.u32s(b.sl.slots[win.slot].held, w.held));
            @memcpy(o.drafts[w.held..], w.tokens);
            row += w.rows();
        }
    }

    fn keep(ptr: *anyopaque, windows: []const be.Window, paths: []const []const u32) anyerror!void {
        const b = self(ptr);
        for (windows, paths) |w, path| {
            for (path, 0..) |r, k| if (r != k) return error.TreeWindowsUnsupported;
            const i = try b.slotOf(w.stream);
            const kept: u32 = @intCast(path.len);
            if (kept == 0 or kept > b.sl.slots[i].rows) return error.KeepOutOfStep;
            try mirror.send(b.sl.e, .keep, &.{ i, kept });
            try b.sl.keep(i, kept);
        }
    }

    fn draft(ptr: *anyopaque, requests: []const be.DraftRequest) anyerror!void {
        const b = self(ptr);
        for (requests) |r| {
            if (r.early or r.lanes != null or r.ranks) return error.TreeDraftsUnsupported;
            const i = try b.slotOf(r.stream);
            const slot = &b.sl.slots[i];
            var one: [1]u32 = undefined;
            const follow: []const u32 = if (r.first) |f| blk: {
                one[0] = switch (f) {
                    .handle => |h| @intCast(h),
                    .value => |v| v,
                };
                break :blk &one;
            } else r.follow;
            const prompt = r.rows == null;
            if (r.rows) |path| {
                if (path.len != follow.len) return error.DraftOutOfStep;
                for (path, 0..) |x, k| if (x != k) return error.TreeDraftsUnsupported;
            }
            if (!slot.mtp or follow.len == 0 or follow.len > st.max_rows or r.depth >= st.max_rows or (prompt and follow.len != 1)) return error.DraftOutOfStep;
            if (!prompt and slot.seen != b.sl.windows) return error.DraftOutOfStep;
            b.words.clearRetainingCapacity();
            try b.words.appendSlice(b.gpa, &.{ i, @intFromBool(prompt), r.depth });
            try b.words.appendSlice(b.gpa, follow);
            try mirror.send(b.sl.e, .draft, b.words.items);
            try b.sl.draft(i, prompt, follow, r.depth);
        }
    }

    fn release(ptr: *anyopaque, s: *Stream) void {
        const b = self(ptr);
        const kv = b.by.fetchRemove(s) orelse return;
        mirror.send(b.sl.e, .release, &.{kv.value}) catch |err| std.log.err("glm: the peer missed a release: {s}", .{@errorName(err)});
        b.sl.release(kv.value);
    }
};
