//! A target backend with an external drafter in front: the round loop sees one Backend. The target runs prefill, verify
//! and keep as before; its `draft` is never called. A drafting stream's drafter absorbs the prompt rows the target
//! computed, then each round's kept rows (from the target's `features`), and holds drafts the next verify takes as host
//! tokens. A stream with drafts off never reaches the drafter, so `"draft": false` stays the target alone.
const std = @import("std");
const Allocator = std.mem.Allocator;
const be = @import("backend.zig");
const Model = @import("config.zig").Model;
const Stream = @import("stream.zig").Stream;
const dr = @import("drafter.zig");

pub const Drafted = struct {
    gpa: Allocator,
    target: be.Backend,
    drafter: dr.Drafter,
    opened: std.AutoHashMapUnmanaged(*Stream, void) = .empty, // streams the drafter has state for
    windows: std.ArrayList(be.Window) = .empty, // the last verify's windows, held drafts swapped for host tokens
    tokens: std.ArrayList(u32) = .empty, // their drafts, back to back

    pub fn deinit(x: *Drafted) void {
        x.opened.deinit(x.gpa);
        x.windows.deinit(x.gpa);
        x.tokens.deinit(x.gpa);
    }

    pub fn backend(x: *Drafted) be.Backend {
        return .{ .ptr = x, .vtable = &.{ .prefill = prefill, .first = first, .queue = queue, .read = read, .verify = verify, .keep = keep, .draft = draft, .release = release } };
    }

    /// The target's facts with the drafter's own scheduling: its depth, step cost, prior, plain guard and batching (a
    /// target without a head of its own has nothing to lend there). Chains drafted late from each round's kept rows.
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

    fn self(ptr: *anyopaque) *Drafted {
        return @ptrCast(@alignCast(ptr));
    }

    /// The target's prompt pass; a drafting stream's drafter absorbs the rows the pass computed (from `s.cached`) but
    /// the last (its follow token is not drawn yet). A failure after the target's pass releases both here: the core
    /// releases only a cancelled pass (and then the drafter was never opened).
    fn prefill(ptr: *anyopaque, s: *Stream) anyerror!void {
        const x = self(ptr);
        try x.target.prefill(s);
        if (!s.drafts) return;
        errdefer x.dropStream(s);
        try x.drafter.open(s);
        try x.opened.put(x.gpa, s, {});
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

    fn queue(ptr: *anyopaque, s: *Stream, feed: be.Feed, position: u64) anyerror!u64 {
        return self(ptr).target.queue(s, feed, position);
    }

    fn read(ptr: *anyopaque, handle: u64) anyerror!u32 {
        return self(ptr).target.read(handle);
    }

    /// Held drafts go to the target as host tokens, every drafting stream's read back in one call.
    fn verify(ptr: *anyopaque, windows: []const be.Window, out: []be.Verified) anyerror!void {
        const x = self(ptr);
        x.windows.clearRetainingCapacity();
        try x.windows.appendSlice(x.gpa, windows);
        var total: usize = 0;
        for (windows) |w| {
            if (w.held > 0 and w.tokens.len != 0) return error.MixedDrafts;
            total += w.held;
        }
        if (total > 0) {
            try x.tokens.resize(x.gpa, total);
            const streams = try x.gpa.alloc(*Stream, windows.len);
            defer x.gpa.free(streams);
            const outs = try x.gpa.alloc([]u32, windows.len);
            defer x.gpa.free(outs);
            var n: usize = 0;
            var at: usize = 0;
            for (windows, x.windows.items) |w, *v| {
                if (w.held == 0) continue;
                if (!x.opened.contains(w.stream)) return error.NotDrafting;
                const slot = x.tokens.items[at..][0..w.held];
                at += w.held;
                streams[n] = w.stream;
                outs[n] = slot;
                n += 1;
                v.tokens = slot;
                v.held = 0;
            }
            try x.drafter.held(streams[0..n], outs[0..n]);
        }
        return x.target.verify(x.windows.items, out);
    }

    /// The target keeps from the windows it verified (host tokens): the round loop's, position for position.
    fn keep(ptr: *anyopaque, windows: []const be.Window, paths: []const []const u32) anyerror!void {
        const x = self(ptr);
        if (windows.len != x.windows.items.len) return error.WindowMismatch;
        for (windows, x.windows.items) |w, v| if (w.stream != v.stream) return error.WindowMismatch;
        return x.target.keep(x.windows.items, paths);
    }

    /// One absorb for every request's kept rows (a chain's prefix; after a prompt, its last row if the pass computed it),
    /// then one hold for every request that wants drafts.
    fn draft(ptr: *anyopaque, requests: []const be.DraftRequest) anyerror!void {
        const x = self(ptr);
        var absorbs: std.ArrayList(dr.Absorb) = .empty;
        defer absorbs.deinit(x.gpa);
        var holds: std.ArrayList(dr.Hold) = .empty;
        defer holds.deinit(x.gpa);
        const firsts = try x.gpa.alloc(u32, requests.len);
        defer x.gpa.free(firsts);
        for (requests, firsts) |r, *tok| {
            if (r.lanes != null) return error.TreesNotBuilt;
            if (!x.opened.contains(r.stream)) return error.NotDrafting;
            var pending: u32 = undefined;
            if (r.rows) |rows| {
                for (rows, 0..) |row, i| if (row != i) return error.TreesNotBuilt;
                const n: u32 = @intCast(rows.len);
                const f = try x.target.features(r.stream, x.drafter.taps(), r.start, n);
                try absorbs.append(x.gpa, .{ .stream = r.stream, .features = f, .start = r.start, .follow = r.follow[0..rows.len] });
                pending = r.follow[rows.len - 1];
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
                    try absorbs.append(x.gpa, .{ .stream = r.stream, .features = f, .start = last, .follow = @as(*const [1]u32, tok) });
                }
            }
            if (r.depth > 0) try holds.append(x.gpa, .{ .stream = r.stream, .pending = pending, .position = r.position, .depth = r.depth });
        }
        if (absorbs.items.len > 0) try x.drafter.absorb(absorbs.items);
        if (holds.items.len > 0) try x.drafter.hold(holds.items);
    }

    fn dropStream(x: *Drafted, s: *Stream) void {
        if (x.opened.remove(s)) x.drafter.release(s);
        x.target.release(s);
    }

    fn release(ptr: *anyopaque, s: *Stream) void {
        self(ptr).dropStream(s);
    }
};
