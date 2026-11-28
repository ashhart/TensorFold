//! A target fixture enforcing Features lifetimes and optionally reading held drafts without wrapper translation.
const std = @import("std");
const be = @import("backend.zig");
const dr = @import("drafter.zig");
const sm = @import("stream.zig");

const gpa = std.testing.allocator;

/// The fake target holding features as `Features` promises (a keep overwrites what it drops), refusing anything else.
pub const Strict = struct {
    inner: be.Backend,
    snaps: std.AutoHashMapUnmanaged(*sm.Stream, Snap) = .empty,
    refused: u32 = 0,
    taps: [8]u32 = undefined,
    tap_count: usize = 0, // the layers it keeps, told before any forward
    head_proposed: u64 = 0,
    head_accepted: u64 = 0,
    reads: ?dr.Drafter = null, // set: held windows' drafts are read from this drafter here, not by the wrapper
    stats: Stats = .{},
    filled: bool = false,
    sent: std.ArrayList(Sent) = .empty, // the last verify's windows as this target was handed them
    slots: std.ArrayList(be.Window) = .empty,
    drafts: std.ArrayList(u32) = .empty,

    const Snap = struct { start: u64 = 0, rows: std.ArrayList(u32) = .empty, valid: usize = 0 };
    const poison: u32 = 0xdead;

    /// Count held-window reads, host fillers and held reads following a filler.
    pub const Stats = struct { reads: u32 = 0, fillers: u32 = 0, after_filler: u32 = 0 };
    const Sent = struct { stream: *sm.Stream, held: u32, host: usize };

    pub fn deinit(x: *Strict) void {
        x.sent.deinit(gpa);
        x.slots.deinit(gpa);
        x.drafts.deinit(gpa);
        var it = x.snaps.valueIterator();
        while (it.next()) |v| v.rows.deinit(gpa);
        x.snaps.deinit(gpa);
    }

    pub fn backend(x: *Strict) be.Backend {
        return .{ .ptr = x, .vtable = &.{ .prefill = prefill, .first = first, .first_masked = firstMasked, .queue = queue, .read = read, .verify = verify, .keep = keep, .draft = draftNone, .prepare_features = prepare, .features = features, .observe_head = observeHead, .release = release } };
    }

    fn self(ptr: *anyopaque) *Strict {
        return @ptrCast(@alignCast(ptr));
    }

    fn observeHead(ptr: *anyopaque, _: *sm.Stream, proposed: u32, accepted: u32) void {
        const x = self(ptr);
        x.head_proposed += proposed;
        x.head_accepted += accepted;
    }

    fn snap(x: *Strict, s: *sm.Stream) !*Snap {
        const got = try x.snaps.getOrPut(gpa, s);
        if (!got.found_existing) got.value_ptr.* = .{};
        return got.value_ptr;
    }

    fn prefill(ptr: *anyopaque, s: *sm.Stream) anyerror!void {
        const x = self(ptr);
        try x.inner.prefill(s);
        if (s.head_context == false) return;
        const sn = try x.snap(s);
        sn.start = s.cached;
        sn.rows.clearRetainingCapacity();
        try sn.rows.appendSlice(gpa, s.prompt()[s.cached..]);
        sn.valid = sn.rows.items.len;
    }

    fn first(ptr: *anyopaque, s: *sm.Stream, position: u64) anyerror!u64 {
        return self(ptr).inner.first(s, position);
    }

    fn firstMasked(ptr: *anyopaque, s: *sm.Stream, position: u64, mask: []const u32) anyerror!u64 {
        const in = self(ptr).inner;
        return in.vtable.first_masked.?(in.ptr, s, position, mask);
    }

    fn queue(ptr: *anyopaque, s: *sm.Stream, feed: be.Feed, position: u64) anyerror!u64 {
        const x = self(ptr);
        if (s.head_context != false) (try x.snap(s)).valid = 0;
        return x.inner.queue(s, feed, position);
    }

    fn read(ptr: *anyopaque, handle: u64) anyerror!u32 {
        return self(ptr).inner.read(handle);
    }

    fn verify(ptr: *anyopaque, windows: []const be.Window, out: []be.Verified) anyerror!void {
        const x = self(ptr);
        x.sent.clearRetainingCapacity();
        for (windows) |w| try x.sent.append(gpa, .{ .stream = w.stream, .held = w.held, .host = w.tokens.len });
        const asked = try x.readHeld(windows);
        try x.inner.verify(asked, out);
        for (asked) |w| {
            if (w.held != 0) return error.HeldReachedTarget; // Drafted hands host tokens
            if (w.stream.head_context == false) continue;
            const sn = try x.snap(w.stream);
            sn.start = w.positions[0] - 1;
            sn.rows.clearRetainingCapacity();
            try sn.rows.append(gpa, w.pending);
            try sn.rows.appendSlice(gpa, w.tokens);
            sn.valid = sn.rows.items.len;
        }
    }

    /// Read held drafts on the target when `reads` is set.
    fn readHeld(x: *Strict, windows: []const be.Window) ![]const be.Window {
        const drafter = x.reads orelse return windows;
        var total: usize = 0;
        for (windows) |w| total += w.held;
        try x.drafts.resize(gpa, total);
        x.slots.clearRetainingCapacity();
        try x.slots.appendSlice(gpa, windows);
        var at: usize = 0;
        for (windows, x.slots.items) |w, *v| {
            if (w.held == 0) { // fillers reach the target as the pending token on the host
                if (w.tokens.len >= 1 and std.mem.allEqual(u32, w.tokens, w.pending)) {
                    x.stats.fillers += 1;
                    x.filled = true;
                }
                continue;
            }
            const slot = x.drafts.items[at..][0..w.held];
            at += w.held;
            try drafter.held(&.{w.stream}, &.{slot});
            x.stats.reads += 1;
            if (x.filled) x.stats.after_filler += 1;
            v.tokens = slot;
            v.held = 0;
        }
        return x.slots.items;
    }

    fn keep(ptr: *anyopaque, windows: []const be.Window, paths: []const []const u32) anyerror!void {
        const x = self(ptr);
        if (x.reads != null) for (windows) |w| { // Match this verify's windows, not stale copies.
            const sent = for (x.sent.items) |s| {
                if (s.stream == w.stream) break s;
            } else return error.KeptUnverifiedStream;
            if (w.held != sent.held or w.tokens.len != sent.host) return error.KeptOtherWindows;
        };
        try x.inner.keep(windows, paths);
        for (windows, paths) |w, path| {
            if (w.stream.head_context == false) continue;
            const sn = try x.snap(w.stream);
            sn.valid = @min(sn.valid, path.len);
            @memset(sn.rows.items[sn.valid..], poison);
        }
    }

    fn prepare(ptr: *anyopaque, taps: []const u32) anyerror!void {
        const x = self(ptr);
        if (x.snaps.count() > 0) return error.PreparedAfterForward;
        @memcpy(x.taps[0..taps.len], taps);
        x.tap_count = taps.len;
    }

    fn draftNone(_: *anyopaque, _: []const be.DraftRequest) anyerror!void {
        return error.TargetDraftCalled;
    }

    fn features(ptr: *anyopaque, s: *sm.Stream, taps: []const u32, start: u64, count: u32) anyerror!be.Features {
        const x = self(ptr);
        if (s.head_context == false) return error.RowsNotHeld;
        if (!std.mem.eql(u32, taps, x.taps[0..x.tap_count])) return error.TapsNotPrepared;
        const sn = x.snaps.getPtr(s) orelse return error.UnknownStream;
        if (start < sn.start or start + count > sn.start + sn.valid) {
            x.refused += 1;
            return error.RowsNotHeld;
        }
        return .{ .buffer = @intFromPtr(sn.rows.items.ptr), .offset = (start - sn.start) * 4, .rows = count, .row_bytes = 4, .space = .host, .dtype = .u32 };
    }

    fn release(ptr: *anyopaque, s: *sm.Stream) void {
        const x = self(ptr);
        x.inner.release(s);
        if (x.snaps.fetchRemove(s)) |kv| {
            var v = kv.value;
            v.rows.deinit(gpa);
        }
    }
};
