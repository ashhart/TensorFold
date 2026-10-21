//! V4.1 lane slots: a stream a slot, greedy, drafts off; a window is a slot's sub-forward, so rows keep solo bits.
const std = @import("std");
const ds = @import("model.zig");
const kv = @import("kv_state.zig");
const lanes = @import("lanes");
const be = lanes.backend;
const Stream = lanes.Stream;

const RING = 1024;

pub const Backend = struct {
    gpa: std.mem.Allocator,
    model: *ds.Model,
    by: std.AutoHashMapUnmanaged(*Stream, u32) = .empty,
    /// Drawn tokens by handle: every read waits for its forward anyway (one sub-forward a window).
    ring: [RING]u32 = undefined,
    puts: u64 = 0,
    first: []u32,
    /// Each slot's cache cursor before its in-flight verify, for the rollback-and-refeed keep.
    verify_len: []u32,

    pub fn init(gpa: std.mem.Allocator, model: *ds.Model) !Backend {
        return .{
            .gpa = gpa,
            .model = model,
            .first = try gpa.alloc(u32, model.lanes),
            .verify_len = try gpa.alloc(u32, model.lanes),
        };
    }

    pub fn deinit(b: *Backend) void {
        b.by.deinit(b.gpa);
        b.gpa.free(b.first);
        b.gpa.free(b.verify_len);
    }

    pub fn backend(b: *Backend) be.Backend {
        return .{ .ptr = b, .vtable = &.{
            .prefill = prefill,
            .first = firstFn,
            .queue = queue,
            .read = read,
            .verify = verify,
            .keep = keep,
            .draft = draft,
            .release = release,
        } };
    }

    /// A window of any width runs exactly as its slot's own forward; the lane core may bundle streams freely.
    pub fn facts(b: *const Backend) lanes.Model {
        const rows = b.model.workspace.plan.rows;
        return .{
            .exact_width = rows,
            .first_copy_rows = 1,
            .mtp = false,
            .speculate = false,
            .speculate_early = false,
            .gpu_tokens = false,
            .plain_guard = true,
            .drafts = 0,
            .window_costs = &.{},
            .streams_exact = true,
            .hidden_rows = true,
            .batch_rows = rows,
            .max_streams = @intCast(b.model.lanes),
            .draft_streams = false,
        };
    }

    fn self(ptr: *anyopaque) *Backend {
        return @ptrCast(@alignCast(ptr));
    }

    fn slotOf(b: *Backend, s: *Stream) !u32 {
        return b.by.get(s) orelse error.UnknownStream;
    }

    fn put(b: *Backend, v: u32) u64 {
        const h = b.puts;
        b.ring[h % RING] = v;
        b.puts += 1;
        return h;
    }

    fn free(b: *const Backend) ?u32 {
        var used: [256]bool = @splat(false);
        var it = b.by.valueIterator();
        while (it.next()) |v| used[v.*] = true;
        for (used, 0..) |u, i| if (b.model.lanes > i and !u) return @intCast(i);
        return null;
    }

    fn draw(b: *Backend, slot: u32) !u32 {
        const row = b.model.workspace.logits(1);
        var best: u32 = 0;
        var maximum = -std.math.inf(f32);
        for (row, 0..) |value, id| {
            if (!std.math.isFinite(value)) return error.NonfiniteLogits;
            if (value > maximum) {
                maximum = value;
                best = @intCast(id);
            }
        }
        _ = slot;
        return best;
    }

    fn prefill(ptr: *anyopaque, s: *Stream) anyerror!void {
        const b = self(ptr);
        if (s.sampling) |p| if (p.temperature > 0) return error.GreedyOnly;
        const prompt = s.prompt();
        if (prompt.len == 0) return error.EmptyPrompt;
        const i = b.free() orelse return error.NoFreeSlot;
        const state = &b.model.caches[i];
        if (prompt.len + s.max_new + 1 > state.plan.capacity) return error.ContextFull;
        try b.by.put(b.gpa, s, i);
        try b.model.resetSlot(i);
        s.cached = 0;
        var at: u32 = 0;
        const rows = b.model.workspace.plan.rows;
        while (at < prompt.len) {
            if (s.isCancelled()) return error.Cancelled;
            const end = @min(at + rows, @as(u32, @intCast(prompt.len)));
            const last = end == prompt.len;
            if (last) {
                _ = try b.model.forwardOn(i, prompt[at..end], .last);
                b.first[i] = try b.draw(i);
            } else {
                _ = try b.model.forwardOn(i, prompt[at..end], .none);
            }
            at = end;
        }
    }

    fn firstFn(ptr: *anyopaque, s: *Stream, position: u64) anyerror!u64 {
        const b = self(ptr);
        const i = try b.slotOf(s);
        if (position != b.model.caches[i].position) return error.PositionMismatch;
        return b.put(b.first[i]);
    }

    fn queue(ptr: *anyopaque, s: *Stream, feed: be.Feed, position: u64) anyerror!u64 {
        const b = self(ptr);
        const i = try b.slotOf(s);
        const state = &b.model.caches[i];
        if (position != state.position) return error.PositionMismatch;
        const token: u32 = switch (feed) {
            .value => |v| v,
            .handle => |h| try read(ptr, h),
        };
        _ = try b.model.forwardOn(i, &.{token}, .last);
        return b.put(try b.draw(i));
    }

    fn read(ptr: *anyopaque, handle: u64) anyerror!u32 {
        const b = self(ptr);
        if (handle >= b.puts or b.puts - handle > RING) return error.StaleHandle;
        return b.ring[handle % RING];
    }

    fn verify(ptr: *anyopaque, windows: []const be.Window, out: []be.Verified) anyerror!void {
        const b = self(ptr);
        for (windows) |w| {
            if (w.parents != null or w.early or w.masks.len > 0) return error.TreeWindowsUnsupported;
        }
        for (windows, out) |w, o| {
            const i = try b.slotOf(w.stream);
            const state = &b.model.caches[i];
            if (w.held > 0) return error.WindowOutOfStep; // no draft head: nothing is held on the GPU
            const rows = w.rows();
            if (w.positions.len != rows) return error.WindowOutOfStep;
            for (w.positions, 0..) |p, r| if (p != state.position + r + 1) return error.PositionMismatch;
            try b.model.beginSlot(i);
            errdefer b.model.commitSlot(i) catch {};
            var feeds = try b.gpa.alloc(u32, rows);
            defer b.gpa.free(feeds);
            feeds[0] = w.pending;
            @memcpy(feeds[1..], w.tokens);
            const row = try b.model.forwardOn(i, feeds, .all);
            for (0..rows) |r| {
                var best: u32 = 0;
                var maximum = -std.math.inf(f32);
                for (row[r * 129280 ..][0..129280], 0..) |value, id| {
                    if (!std.math.isFinite(value)) return error.NonfiniteLogits;
                    if (value > maximum) {
                        maximum = value;
                        best = @intCast(id);
                    }
                }
                o.sampled[r] = best;
            }
            // The window's fed tokens after the pending one landed: the verified drafts, GLM's contract.
            if (o.drafts.len > 0) @memcpy(o.drafts, feeds[1 .. 1 + o.drafts.len]);
            try b.model.commitSlot(i);
        }
    }

    fn keep(ptr: *anyopaque, windows: []const be.Window, paths: []const []const u32) anyerror!void {
        const b = self(ptr);
        for (windows, paths) |w, path| {
            const i = try b.slotOf(w.stream);
            const rows = w.rows();
            const kept = path.len;
            if (kept >= rows) continue; // the whole window landed
            // Roll back before the window, re-feed kept rows; lanes receipt: re-fed rows bit-identical to pass one.
            try b.model.rollbackSlot(i);
            if (kept == 0) continue;
            var feeds = try b.gpa.alloc(u32, kept);
            defer b.gpa.free(feeds);
            feeds[0] = w.pending;
            @memcpy(feeds[1..], w.tokens[0 .. kept - 1]);
            _ = try b.model.forwardOn(i, feeds, .none);
        }
    }

    fn draft(_: *anyopaque, requests: []const be.DraftRequest) anyerror!void {
        for (requests) |r| if (r.depth > 0) return error.Unsupported;
    }

    fn release(ptr: *anyopaque, s: *Stream) void {
        const b = self(ptr);
        const entry = b.by.fetchRemove(s) orelse return;
        b.model.resetSlot(entry.value) catch {};
    }
};
