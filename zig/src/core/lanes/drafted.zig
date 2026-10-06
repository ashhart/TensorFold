//! A target backend with an external drafter in front: the round loop sees one Backend. The target runs prefill, verify
//! and keep as before; its `draft` is never called. After a prompt the drafter absorbs the prompt's rows, after a verify
//! the kept rows, each from the target's `features`, then holds drafts that the next verify takes as host tokens.
const std = @import("std");
const Allocator = std.mem.Allocator;
const be = @import("backend.zig");
const Model = @import("config.zig").Model;
const Stream = @import("stream.zig").Stream;
const Drafter = @import("drafter.zig").Drafter;

const max_window = 64;

pub const Drafted = struct {
    gpa: Allocator,
    target: be.Backend,
    drafter: Drafter,
    windows: std.ArrayList(be.Window) = .empty, // the last verify's windows, held drafts swapped for host tokens
    tokens: std.ArrayList([max_window]u32) = .empty,

    pub fn deinit(x: *Drafted) void {
        x.windows.deinit(x.gpa);
        x.tokens.deinit(x.gpa);
    }

    pub fn backend(x: *Drafted) be.Backend {
        return .{ .ptr = x, .vtable = &.{ .prefill = prefill, .first = first, .queue = queue, .read = read, .verify = verify, .keep = keep, .draft = draft, .release = release } };
    }

    /// The target's facts with this drafter's: chains of up to its depth, drafted late from each round's kept rows.
    pub fn facts(x: *const Drafted, target: Model) Model {
        var m = target;
        m.mtp = true;
        m.speculate = true;
        m.speculate_early = false;
        m.drafts = @min(x.drafter.depth(), if (target.exact_width > 1) target.exact_width - 1 else 0);
        m.draft_probabilities = false;
        m.head_trees = false;
        return m;
    }

    fn self(ptr: *anyopaque) *Drafted {
        return @ptrCast(@alignCast(ptr));
    }

    /// The target's prompt pass; the drafter absorbs the rows the pass computed (after a restored prefix, `s.cached`),
    /// every one but the last (its follow token is not drawn yet).
    fn prefill(ptr: *anyopaque, s: *Stream) anyerror!void {
        const x = self(ptr);
        try x.target.prefill(s);
        try x.drafter.open(s);
        const ids = s.prompt();
        const from: usize = s.cached;
        if (ids.len < from + 2) return;
        const n: u32 = @intCast(ids.len - 1 - from);
        try x.drafter.absorb(s, try x.target.features(s, x.drafter.taps(), from, n), from, ids[from + 1 ..]);
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

    /// Held drafts go to the target as host tokens; the drafts it echoes back are the same.
    fn verify(ptr: *anyopaque, windows: []const be.Window, out: []be.Verified) anyerror!void {
        const x = self(ptr);
        x.windows.clearRetainingCapacity();
        try x.tokens.resize(x.gpa, windows.len);
        for (windows, x.tokens.items) |w, *buf| {
            var v = w;
            if (w.held > 0) {
                if (w.tokens.len != 0 or w.held > max_window) return error.WindowTooWide;
                const n = try x.drafter.held(w.stream, buf);
                if (n < w.held) return error.TooFewDrafts;
                v.tokens = buf[0..w.held];
                v.held = 0;
            }
            try x.windows.append(x.gpa, v);
        }
        return x.target.verify(x.windows.items, out);
    }

    fn keep(ptr: *anyopaque, windows: []const be.Window, paths: []const []const u32) anyerror!void {
        const x = self(ptr);
        // the target's windows are the ones it verified (host tokens), matched to the round loop's by stream
        const ours = try x.gpa.alloc(be.Window, windows.len);
        defer x.gpa.free(ours);
        for (windows, ours) |w, *o| {
            o.* = w;
            for (x.windows.items) |v| if (v.stream == w.stream) {
                o.* = v;
            };
        }
        return x.target.keep(ours, paths);
    }

    /// The kept rows (a chain's prefix) or, after a prompt, its last row, absorbed with the token after each; then drafts.
    fn draft(ptr: *anyopaque, requests: []const be.DraftRequest) anyerror!void {
        const x = self(ptr);
        for (requests) |r| {
            if (r.lanes != null) return error.TreesNotBuilt;
            if (r.rows) |rows| {
                for (rows, 0..) |row, i| if (row != i) return error.TreesNotBuilt;
                const n: u32 = @intCast(rows.len);
                try x.drafter.absorb(r.stream, try x.target.features(r.stream, x.drafter.taps(), r.start, n), r.start, r.follow[0..rows.len]);
            } else {
                const feed = r.first orelse return error.NoFirstToken;
                const token = switch (feed) {
                    .handle => |h| try x.target.read(h),
                    .value => |v| v,
                };
                const last = r.stream.prompt_len - 1;
                try x.drafter.absorb(r.stream, try x.target.features(r.stream, x.drafter.taps(), last, 1), last, &.{token});
            }
            if (r.depth > 0) try x.drafter.hold(r.stream, r.position, r.depth);
        }
    }

    fn release(ptr: *anyopaque, s: *Stream) void {
        const x = self(ptr);
        x.drafter.release(s);
        x.target.release(s);
    }
};
