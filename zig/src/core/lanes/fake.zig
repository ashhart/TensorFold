//! A host-only test backend whose tokens depend on all fed history, so a rollback or position mistake changes them.
const std = @import("std");
const Allocator = std.mem.Allocator;
const be = @import("backend.zig");
const Stream = @import("stream.zig").Stream;
const Sampling = @import("sampling.zig").Sampling;
const keyBits = @import("sampling.zig").keyBits;
const accept = @import("accept.zig");
const Row = @import("logprob.zig").Row;

pub const vocab = 97;

/// The fake target's draw after `history` at `position` (keyed noise when sampled).
pub fn next(history: []const u32, s: ?Sampling, position: u64) u32 {
    var sum: u64 = 0;
    for (history) |t| sum +%= t;
    var token = (sum *% 7 +% history.len *% 3) % vocab;
    if (s) |st| token = (token + keyBits(st.seed, position, 0) % 5) % vocab;
    return @intCast(token);
}

/// The fake target's row after `history` for `pick`, a function of the history as logits are; best ids from the pick.
pub fn rowAt(history: []const u32, pick: u32, count: u8) Row {
    var sum: u64 = 0;
    for (history) |t| sum +%= t;
    const base = -@as(f32, @floatFromInt(sum % 50)) / 64.0;
    var r: Row = .{ .token = pick, .logprob = base, .count = count };
    for (0..count) |i| {
        r.ids[i] = @intCast((pick + i) % vocab);
        r.logprobs[i] = base - @as(f32, @floatFromInt(i));
    }
    return r;
}

const Lane = struct {
    history: std.ArrayList(u32) = .empty, // the stream's cache: prompt and fed tokens
    held: std.ArrayList(u32) = .empty, // drafts for the next round
    base: usize = 0, // the history length before the last verify
    rows: std.ArrayList(u32) = .empty, // the last verify's row tokens
    filled: usize = 0, // prompt chunks a stepped prefill has run
    filling: bool = false,
};

pub const Fake = struct {
    gpa: Allocator,
    lanes: std.AutoHashMapUnmanaged(*Stream, Lane) = .empty,
    drawn: std.ArrayList(u32) = .empty, // handle -> token
    rounds: u64 = 0,
    cycle_after: ?usize = null,
    probe_prompt: usize = 0,
    pattern: []const u32 = &.{ 11, 12, 13 },
    answer_cycles: bool = false,
    prefill_chunks: usize = 0,
    prefill_count: usize = 0,
    prefill_hook: ?*const fn (ctx: *anyopaque, s: *Stream, chunk: usize) void = null,
    prefill_hook_ctx: ?*anyopaque = null,
    refuse_sampled: bool = false, // prefill refuses a sampled stream with error.SamplingRefused
    memory_lanes: ?usize = null, // prefill past this many live lanes fails with error.LaneMemory (no room yet)

    pub fn deinit(x: *Fake) void {
        var it = x.lanes.valueIterator();
        while (it.next()) |l| freeLane(x.gpa, l);
        x.lanes.deinit(x.gpa);
        x.drawn.deinit(x.gpa);
    }

    fn freeLane(gpa: Allocator, l: *Lane) void {
        l.history.deinit(gpa);
        l.held.deinit(gpa);
        l.rows.deinit(gpa);
    }

    pub fn backend(x: *Fake) be.Backend {
        return .{ .ptr = x, .vtable = &.{ .prefill = prefill, .first = first, .queue = queue, .read = read, .verify = verify, .keep = keep, .draft = draft, .features = features, .release = release, .first_row = firstRow } };
    }

    /// The same target filling its prompt a chunk a call (`prefill_chunks`), as a backend with `prefill_step`.
    pub fn stepped(x: *Fake) be.Backend {
        return .{ .ptr = x, .vtable = &.{ .prefill = prefill, .prefill_step = prefillStep, .first = first, .queue = queue, .read = read, .verify = verify, .keep = keep, .draft = draft, .features = features, .release = release, .first_row = firstRow } };
    }

    fn self(ptr: *anyopaque) *Fake {
        return @ptrCast(@alignCast(ptr));
    }

    fn lane(x: *Fake, s: *Stream) *Lane {
        return x.lanes.getPtr(s).?;
    }

    fn draw(x: *Fake, l: *Lane, s: *Stream, position: u64) !u64 {
        if (position != l.history.items.len) return error.PositionMismatch;
        try x.drawn.append(x.gpa, x.targetNext(l.history.items, s.sampling, position));
        return x.drawn.items.len - 1;
    }

    fn targetNext(x: *Fake, history: []const u32, sampling: ?Sampling, position: u64) u32 {
        if (x.cycle_after) |start| if (history.len >= x.probe_prompt) {
            const reply = history[x.probe_prompt..];
            if (std.mem.indexOfScalar(u32, reply, 91)) |close| {
                if (reply.len == close + 1) return 92;
                const after = reply.len - close - 2;
                if (x.answer_cycles) {
                    const answer = [_]u32{ 21, 22, 23 };
                    return answer[after % 3];
                }
                const answer = [_]u32{ 40, 41, 42, 96 };
                return answer[@min(after, 3)];
            }
            if (reply.len >= start) return x.pattern[(reply.len - start) % x.pattern.len];
        };
        return next(history, sampling, position);
    }

    fn value(x: *Fake, feed: be.Feed) u32 {
        return switch (feed) {
            .handle => |h| x.drawn.items[h],
            .value => |v| v,
        };
    }

    fn prefill(ptr: *anyopaque, s: *Stream) anyerror!void {
        const x = self(ptr);
        if (x.refuse_sampled and s.sampling != null) return error.SamplingRefused;
        if (x.memory_lanes) |n| if (!x.lanes.contains(s) and x.lanes.count() >= n) return error.LaneMemory;
        const got = try x.lanes.getOrPut(x.gpa, s);
        if (got.found_existing) freeLane(x.gpa, got.value_ptr);
        got.value_ptr.* = .{};
        const chunks = if (x.prefill_chunks == 0) 1 else x.prefill_chunks;
        for (0..chunks) |chunk| {
            x.prefill_count += 1;
            if (x.prefill_hook) |hook| hook(x.prefill_hook_ctx.?, s, chunk);
            if (s.isCancelled()) return error.Cancelled;
        }
        try x.fillPrompt(got.value_ptr, s);
    }

    fn prefillStep(ptr: *anyopaque, s: *Stream) anyerror!bool {
        const x = self(ptr);
        if (x.refuse_sampled and s.sampling != null) return error.SamplingRefused;
        if (x.memory_lanes) |n| if (!x.lanes.contains(s) and x.lanes.count() >= n) return error.LaneMemory;
        const got = try x.lanes.getOrPut(x.gpa, s);
        if (got.found_existing and !got.value_ptr.filling) freeLane(x.gpa, got.value_ptr);
        if (!got.found_existing or !got.value_ptr.filling) got.value_ptr.* = .{ .filling = true };
        const l = got.value_ptr;
        x.prefill_count += 1;
        if (x.prefill_hook) |hook| hook(x.prefill_hook_ctx.?, s, l.filled);
        if (s.isCancelled()) return error.Cancelled;
        l.filled += 1;
        if (l.filled < @max(1, x.prefill_chunks)) return false;
        l.filling = false;
        try x.fillPrompt(l, s);
        return true;
    }

    /// The pass's end: the cache holds the prompt, from a kept state where one was restored.
    fn fillPrompt(x: *Fake, l: *Lane, s: *Stream) !void {
        const h = &l.history;
        s.cached = 0;
        if (s.reuse.saved) |saved| { // the kept state's tokens stand in for the prompt's first `at`
            const kept: *std.ArrayList(u32) = @ptrCast(@alignCast(saved));
            try h.appendSlice(x.gpa, kept.items);
            s.cached = s.reuse.at;
        }
        for (s.prompt()[h.items.len..], h.items.len + 1..) |t, at| {
            try h.append(x.gpa, t);
            if (std.mem.indexOfScalar(u32, s.reuse.marks, @intCast(at)) != null) if (s.reuse.hook) |k| k.at(k.ptr, s, @intCast(at));
        }
    }

    /// A copy of the stream's cache at `at` tokens (the prompt pass stands there), for core/prompt_cache.zig.
    pub fn save(x: *Fake, s: *Stream, at: u32) !*anyopaque {
        const l = x.lane(s);
        if (l.history.items.len != at) return error.NotAtMark;
        const kept = try x.gpa.create(std.ArrayList(u32));
        kept.* = .empty;
        try kept.appendSlice(x.gpa, l.history.items);
        return kept;
    }

    pub fn drop(x: *Fake, saved: *anyopaque) void {
        const kept: *std.ArrayList(u32) = @ptrCast(@alignCast(saved));
        kept.deinit(x.gpa);
        x.gpa.destroy(kept);
    }

    fn first(ptr: *anyopaque, s: *Stream, position: u64) anyerror!u64 {
        const x = self(ptr);
        return x.draw(x.lane(s), s, position);
    }

    fn firstRow(ptr: *anyopaque, s: *Stream) anyerror!Row {
        const x = self(ptr);
        const h = x.lane(s).history.items;
        return rowAt(h, x.targetNext(h, s.sampling, h.len), s.logprobs.?);
    }

    fn queue(ptr: *anyopaque, s: *Stream, feed: be.Feed, position: u64) anyerror!u64 {
        const x = self(ptr);
        const l = x.lane(s);
        try l.history.append(x.gpa, x.value(feed));
        return x.draw(l, s, position);
    }

    fn read(ptr: *anyopaque, handle: u64) anyerror!u32 {
        return self(ptr).drawn.items[handle];
    }

    fn verify(ptr: *anyopaque, windows: []const be.Window, out: []be.Verified) anyerror!void {
        const x = self(ptr);
        x.rounds += 1;
        for (windows, out) |w, o| {
            const l = x.lane(w.stream);
            l.base = l.history.items.len;
            l.rows.clearRetainingCapacity();
            try l.rows.append(x.gpa, w.pending);
            try l.rows.appendSlice(x.gpa, l.held.items[0..w.held]);
            try l.rows.appendSlice(x.gpa, w.tokens);
            @memcpy(o.drafts, l.rows.items[1..]);
            const parents = try accept.rowParents(x.gpa, l.rows.items.len, w.parents);
            defer x.gpa.free(parents);
            for (0..l.rows.items.len) |r| {
                // each row reads the cache and its own path from the pending row
                l.history.shrinkRetainingCapacity(l.base);
                var path: std.ArrayList(u32) = .empty;
                defer path.deinit(x.gpa);
                var at: i32 = @intCast(r);
                while (at >= 0) : (at = parents[@intCast(at)]) try path.insert(x.gpa, 0, l.rows.items[@intCast(at)]);
                try l.history.appendSlice(x.gpa, path.items);
                if (w.positions[r] != l.history.items.len) return error.PositionMismatch;
                o.sampled[r] = x.targetNext(l.history.items, w.stream.sampling, w.positions[r]);
                if (o.rows.len > 0) o.rows[r] = rowAt(l.history.items, o.sampled[r], w.stream.logprobs.?);
            }
            l.history.shrinkRetainingCapacity(l.base);
            try l.history.appendSlice(x.gpa, l.rows.items); // a chain keeps every row unless rolled back
        }
    }

    fn keep(ptr: *anyopaque, windows: []const be.Window, paths: []const []const u32) anyerror!void {
        const x = self(ptr);
        for (windows, paths) |w, path| {
            const l = x.lane(w.stream);
            l.history.shrinkRetainingCapacity(l.base);
            for (path) |r| try l.history.append(x.gpa, l.rows.items[r]);
        }
    }

    fn draft(ptr: *anyopaque, requests: []const be.DraftRequest) anyerror!void {
        const x = self(ptr);
        for (requests) |r| {
            const l = x.lane(r.stream);
            // the head reads the last verify's kept rows (a shared round drafts before its rollback)
            var guess: std.ArrayList(u32) = .empty;
            defer guess.deinit(x.gpa);
            if (r.rows) |path| {
                try guess.appendSlice(x.gpa, l.history.items[0..l.base]);
                for (path) |row| try guess.append(x.gpa, l.rows.items[row]);
            } else try guess.appendSlice(x.gpa, l.history.items);
            if (r.position != guess.items.len + 1) return error.PositionMismatch;
            const pending = if (r.first) |feed| x.value(feed) else r.follow[r.follow.len - 1];
            l.held.clearRetainingCapacity();
            try guess.append(x.gpa, pending);
            for (0..r.depth) |j| {
                const at = r.position + j;
                var t = x.targetNext(guess.items, r.stream.sampling, at);
                if ((at *% 2654435761 + j) % 5 == 0) t = (t + 1) % vocab; // a wrong guess now and then
                try l.held.append(x.gpa, t);
                try guess.append(x.gpa, t);
            }
        }
    }

    /// A row's "state" is its token (one u32 a row, whatever the taps): enough for a drafter to rebuild the history.
    fn features(ptr: *anyopaque, s: *Stream, taps: []const u32, start: u64, count: u32) anyerror!be.Features {
        _ = taps;
        const l = self(ptr).lane(s);
        if (start + count > l.history.items.len) return error.RowsNotHeld;
        return .{ .buffer = @intFromPtr(l.history.items.ptr), .offset = start * 4, .rows = count, .row_bytes = 4, .space = .host, .dtype = .u32 };
    }

    fn release(ptr: *anyopaque, s: *Stream) void {
        const x = self(ptr);
        if (x.lanes.fetchRemove(s)) |kv| {
            var l = kv.value;
            freeLane(x.gpa, &l);
        }
    }
};

/// Round times from a fixed cost model, so tests decide depth the same way every run.
pub const FixedClock = struct {
    calls: u64 = 0,

    pub fn clock(c: *FixedClock) be.Clock {
        return .{ .ptr = c, .vtable = &.{ .start = start, .elapsed_ms = elapsed } };
    }

    fn start(_: *anyopaque) void {}

    fn elapsed(ptr: *anyopaque, _: be.Mark) f64 {
        const c: *FixedClock = @ptrCast(@alignCast(ptr));
        c.calls += 1;
        return 6.0 + @as(f64, @floatFromInt(c.calls % 7)) * 0.5;
    }
};
