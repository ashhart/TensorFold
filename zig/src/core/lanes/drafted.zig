//! A target backend with an external drafter in front; deinit releases no stream, so callers release first.
const std = @import("std");
const Allocator = std.mem.Allocator;
const be = @import("backend.zig");
const Model = @import("config.zig").Model;
const Stream = @import("stream.zig").Stream;
const dr = @import("drafter.zig");

/// What the wrapper knows of a drafting stream.
const Lane = struct {
    rows: bool = false, // the drafter has absorbed the row before the next draft (it can draft)
    filler: u32 = 0, // drafts the next verify takes as fillers instead (a prompt restored whole: no row to draft from)
};

pub const Drafted = struct {
    gpa: Allocator,
    target: be.Backend,
    drafter: dr.Drafter,
    lanes: std.AutoHashMapUnmanaged(*Stream, Lane) = .empty, // streams the drafter has opened
    // scratch kept between rounds, so a steady round allocates nothing
    windows: std.ArrayList(be.Window) = .empty, // the last verify's windows, held drafts swapped for host tokens
    tokens: std.ArrayList(u32) = .empty, // their drafts, back to back
    streams: std.ArrayList(*Stream) = .empty,
    outs: std.ArrayList([]u32) = .empty,
    absorbs: std.ArrayList(dr.Absorb) = .empty,
    holds: std.ArrayList(dr.Hold) = .empty,
    firsts: std.ArrayList(u32) = .empty,

    /// The wrapper, with the target told the drafter's taps before any forward.
    pub fn init(gpa: Allocator, target: be.Backend, drafter: dr.Drafter) !Drafted {
        if (target.vtable.prepare_features) |prep| try prep(target.ptr, drafter.taps());
        return .{ .gpa = gpa, .target = target, .drafter = drafter };
    }

    pub fn deinit(x: *Drafted) void {
        x.lanes.deinit(x.gpa);
        x.windows.deinit(x.gpa);
        x.tokens.deinit(x.gpa);
        x.streams.deinit(x.gpa);
        x.outs.deinit(x.gpa);
        x.absorbs.deinit(x.gpa);
        x.holds.deinit(x.gpa);
        x.firsts.deinit(x.gpa);
    }

    pub fn backend(x: *Drafted) be.Backend {
        return .{ .ptr = x, .vtable = &.{ .prefill = prefill, .first = first, .first_masked = firstMasked, .queue = queue, .read = read, .verify = verify, .keep = keep, .draft = draft, .release = release } };
    }

    /// The target's facts with the drafter's depth, step cost, prior, plain guard and batching; chains drafted late.
    pub fn facts(x: *const Drafted, target: Model) Model {
        const d = x.drafter.facts();
        var m = target;
        m.mtp = true;
        m.speculate = true;
        m.speculate_early = false;
        m.drafts = @min(d.depth, if (target.exact_width > 1) target.exact_width - 1 else 0);
        m.mtp_step_ms = d.step_ms;
        m.draft_prior = d.prior;
        m.plain_guard = d.plain_guard;
        m.draft_streams = d.batched;
        m.draft_probabilities = false;
        m.head_trees = false;
        return m;
    }

    /// Device bytes an admitted stream takes: the target's own and the drafter's for the same stream.
    pub fn streamBytes(x: *const Drafted, target: usize) usize {
        return target + x.drafter.facts().stream_bytes;
    }

    fn self(ptr: *anyopaque) *Drafted {
        return @ptrCast(@alignCast(ptr));
    }

    /// The target's prompt pass, then the drafter absorbs its rows but the last; a later failure releases both.
    fn prefill(ptr: *anyopaque, s: *Stream) anyerror!void {
        const x = self(ptr);
        try x.target.prefill(s);
        if (!s.drafts) return;
        errdefer x.dropStream(s);
        try x.lanes.ensureUnusedCapacity(x.gpa, 1); // before open: a stream opened is always one the wrapper releases
        try x.drafter.open(s);
        x.lanes.putAssumeCapacity(s, .{});
        const ids = s.prompt();
        const from: usize = s.cached;
        if (ids.len < from + 2) return;
        const n: u32 = @intCast(ids.len - 1 - from);
        const f = try x.target.features(s, x.drafter.taps(), from, n);
        try x.drafter.absorb(&.{.{ .stream = s, .features = f, .start = from, .follow = ids[from + 1 ..] }});
    }

    fn first(ptr: *anyopaque, s: *Stream, position: u64) anyerror!u64 {
        return self(ptr).target.first(s, position);
    }

    fn firstMasked(ptr: *anyopaque, s: *Stream, position: u64, mask: []const u32) anyerror!u64 {
        const t = self(ptr).target;
        const masked = t.vtable.first_masked orelse return error.StructuresUnsupported;
        return masked(t.ptr, s, position, mask);
    }

    fn queue(ptr: *anyopaque, s: *Stream, feed: be.Feed, position: u64) anyerror!u64 {
        return self(ptr).target.queue(s, feed, position);
    }

    fn read(ptr: *anyopaque, handle: u64) anyerror!u32 {
        return self(ptr).target.read(handle);
    }

    /// Held drafts reach the target as host tokens, all streams read back in one call (fillers: the pending token).
    fn verify(ptr: *anyopaque, windows: []const be.Window, out: []be.Verified) anyerror!void {
        const x = self(ptr);
        x.windows.clearRetainingCapacity();
        try x.windows.appendSlice(x.gpa, windows);
        var total: usize = 0;
        for (windows) |w| {
            if (w.held > 0 and w.tokens.len != 0) return error.MixedDrafts;
            total += w.held;
        }
        if (total == 0) return x.target.verify(x.windows.items, out);
        try x.tokens.resize(x.gpa, total);
        x.streams.clearRetainingCapacity();
        x.outs.clearRetainingCapacity();
        var at: usize = 0;
        for (windows, x.windows.items) |w, *v| {
            if (w.held == 0) continue;
            const lane = x.lanes.getPtr(w.stream) orelse return error.NotDrafting;
            const slot = x.tokens.items[at..][0..w.held];
            at += w.held;
            v.tokens = slot;
            v.held = 0;
            if (lane.filler > 0) {
                @memset(slot, w.pending);
                lane.filler = 0;
                continue;
            }
            try x.streams.append(x.gpa, w.stream);
            try x.outs.append(x.gpa, slot);
        }
        if (x.streams.items.len > 0) try x.drafter.held(x.streams.items, x.outs.items);
        return x.target.verify(x.windows.items, out);
    }

    /// The target keeps from the windows it verified (host tokens): the round loop's, position for position.
    fn keep(ptr: *anyopaque, windows: []const be.Window, paths: []const []const u32) anyerror!void {
        const x = self(ptr);
        if (windows.len != x.windows.items.len) return error.WindowMismatch;
        for (windows, x.windows.items) |w, v| if (w.stream != v.stream) return error.WindowMismatch;
        return x.target.keep(x.windows.items, paths);
    }

    /// One absorb for every request's kept rows, then one hold for every request that wants drafts and can draft.
    fn draft(ptr: *anyopaque, requests: []const be.DraftRequest) anyerror!void {
        const x = self(ptr);
        x.absorbs.clearRetainingCapacity();
        x.holds.clearRetainingCapacity();
        try x.firsts.resize(x.gpa, requests.len);
        for (requests, x.firsts.items) |r, *tok| {
            if (r.lanes != null) return error.TreesNotBuilt;
            const lane = x.lanes.getPtr(r.stream) orelse return error.NotDrafting;
            lane.filler = 0; // a new request supersedes fillers a copy round left unused
            var pending: u32 = undefined;
            if (r.rows) |rows| {
                for (rows, 0..) |row, i| if (row != i) return error.TreesNotBuilt;
                const n: u32 = @intCast(rows.len);
                const f = try x.target.features(r.stream, x.drafter.taps(), r.start, n);
                try x.absorbs.append(x.gpa, .{ .stream = r.stream, .features = f, .start = r.start, .follow = r.follow[0..rows.len] });
                pending = r.follow[rows.len - 1];
                lane.rows = true;
            } else {
                const feed = r.first orelse return error.NoFirstToken;
                tok.* = switch (feed) {
                    .handle => |h| try x.target.read(h),
                    .value => |v| v,
                };
                pending = tok.*;
                const last = r.stream.prompt_len - 1;
                if (r.stream.cached <= last) { // the pass computed the prompt's last row (not restored whole)
                    const f = try x.target.features(r.stream, x.drafter.taps(), last, 1);
                    try x.absorbs.append(x.gpa, .{ .stream = r.stream, .features = f, .start = last, .follow = @as(*const [1]u32, tok) });
                    lane.rows = true;
                }
            }
            if (r.depth == 0) continue;
            if (!lane.rows) { // nothing to draft from yet: fillers this round, the drafter from the next
                lane.filler = r.depth;
                continue;
            }
            try x.holds.append(x.gpa, .{ .stream = r.stream, .pending = pending, .position = r.position, .depth = r.depth });
        }
        if (x.absorbs.items.len > 0) try x.drafter.absorb(x.absorbs.items);
        if (x.holds.items.len > 0) try x.drafter.hold(x.holds.items);
    }

    fn dropStream(x: *Drafted, s: *Stream) void {
        if (x.lanes.remove(s)) x.drafter.release(s);
        x.target.release(s);
    }

    fn release(ptr: *anyopaque, s: *Stream) void {
        self(ptr).dropStream(s);
    }
};
