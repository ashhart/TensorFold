//! Qwen recurrence, windows and keyed draws behind the existing lane engine's backend interface.
const std = @import("std");
const mtl = @import("metal");
const lanes = @import("lanes");
const c = @import("config.zig");
const st = @import("state.zig");
const fwd = @import("forward.zig");
const Model = @import("model.zig").Model;
const be = lanes.backend;

pub const Options = struct { capacity: usize, chunk: usize = 128, streams: u32 = 8 };

pub const Metal = struct {
    residency: ?mtl.ResidencySet.Follow = null,
    gpa: std.mem.Allocator,
    model: *Model,
    options: Options,
    scratch: st.Scratch,
    width: u32 = 1,
    caches: std.AutoHashMapUnmanaged(*lanes.Stream, *st.Cache) = .empty,

    pub fn init(gpa: std.mem.Allocator, model: *Model, options: Options) !*Metal {
        const out = try gpa.create(Metal);
        errdefer gpa.destroy(out);
        out.* = .{ .gpa = gpa, .model = model, .options = options, .scratch = try st.Scratch.init(gpa, model.device, @max(options.chunk, st.batch_rows), options.capacity) };
        errdefer out.scratch.deinit();
        @import("qualify.zig").check(model, &out.scratch) catch |err| switch (err) {
            error.QwenCacheLengthMismatch, error.QwenRecurrenceMismatch, error.QwenAttentionCacheMismatch, error.QwenWindowLogitsMismatch, error.QwenCommittedLogitsMismatch => {
                std.log.warn("Qwen wider lanes failed the local exactness check ({s}); keeping one-row lane rounds", .{@errorName(err)});
                return out;
            },
            else => return err,
        };
        out.width = st.window_rows;
        return out;
    }

    pub fn deinit(self: *Metal) void {
        mtl.resident.streams.set(self, null);
        var it = self.caches.valueIterator();
        while (it.next()) |cache| {
            cache.*.deinit();
            self.gpa.destroy(cache.*);
        }
        self.caches.deinit(self.gpa);
        self.scratch.deinit();
        self.gpa.destroy(self);
    }

    pub fn facts(self: *const Metal) lanes.Model {
        return .{ .exact_width = self.width, .hidden_rows = self.width > 1, .batch_rows = st.batch_rows, .max_streams = self.options.streams, .streams_exact = true };
    }

    pub fn backend(self: *Metal) be.Backend {
        return .{ .ptr = self, .vtable = &.{ .prefill = prefillFn, .first = firstFn, .first_masked = firstMaskedFn, .queue = queueFn, .read = readFn, .verify = verifyFn, .keep = keepFn, .draft = draftFn, .release = releaseFn, .reports_fill = true } };
    }

    fn cast(ptr: *anyopaque) *Metal {
        return @ptrCast(@alignCast(ptr));
    }

    fn cacheOf(self: *Metal, stream: *lanes.Stream) !*st.Cache {
        return self.caches.get(stream) orelse error.UnknownQwenStream;
    }

    pub fn newCache(self: *Metal, capacity: usize) !*st.Cache {
        const cache = try self.gpa.create(st.Cache);
        errdefer self.gpa.destroy(cache);
        cache.* = try st.Cache.init(self.gpa, self.model.device, capacity);
        cache.memory.setResident(self.residency);
        return cache;
    }

    fn prefillFn(ptr: *anyopaque, stream: *lanes.Stream) !void {
        const self = cast(ptr);
        const ids = stream.prompt();
        if (ids.len == 0 or ids.len + stream.max_new > self.options.capacity or self.caches.contains(stream)) return error.InvalidQwenPrompt;
        const capacity = @min(self.options.capacity, ids.len + stream.max_new + st.window_rows);
        const cache = try self.newCache(capacity);
        errdefer self.gpa.destroy(cache);
        errdefer cache.deinit();
        var at: usize = 0;
        while (at < ids.len) {
            var end: usize = @min(ids.len, at + self.options.chunk);
            for (stream.chunks) |boundary| if (boundary > at) {
                end = @min(end, boundary);
                break;
            };
            try fwd.run(self.model, &self.scratch, &.{.{ .cache = cache, .rows = end - at }}, ids[at..end], false, .last);
            at = end;
            stream.fill_at.store(@intCast(at), .monotonic); // fwd.run waits: `at` is landed progress
        }
        try self.caches.put(self.gpa, stream, cache);
    }

    fn firstFn(ptr: *anyopaque, stream: *lanes.Stream, position: u64) !u64 {
        const self = cast(ptr);
        const cache = try self.cacheOf(stream);
        if (position != cache.len) return error.InvalidQwenPosition;
        return try draw(self.gpa, cache.logits.slice(u16, c.vocab), stream.sampling, position, &.{});
    }

    fn firstMaskedFn(ptr: *anyopaque, stream: *lanes.Stream, position: u64, mask: []const u32) !u64 {
        const self = cast(ptr);
        const cache = try self.cacheOf(stream);
        if (position != cache.len) return error.InvalidQwenPosition;
        return try draw(self.gpa, cache.logits.slice(u16, c.vocab), stream.sampling, position, mask);
    }

    fn queueFn(ptr: *anyopaque, stream: *lanes.Stream, feed: be.Feed, position: u64) !u64 {
        const self = cast(ptr);
        const cache = try self.cacheOf(stream);
        if (position != cache.len + 1) return error.InvalidQwenPosition;
        const token: u32 = switch (feed) {
            .value => |v| v,
            .handle => |h| @intCast(h),
        };
        try fwd.run(self.model, &self.scratch, &.{.{ .cache = cache, .rows = 1 }}, &.{token}, false, .all);
        return try draw(self.gpa, cache.logits.slice(u16, c.vocab), stream.sampling, position, &.{});
    }

    fn readFn(_: *anyopaque, handle: u64) !u32 {
        if (handle >= c.vocab) return error.InvalidQwenToken;
        return @intCast(handle);
    }

    fn verifyFn(ptr: *anyopaque, windows: []const be.Window, out: []be.Verified) !void {
        const self = cast(ptr);
        var segments: [st.batch_rows]fwd.Segment = undefined;
        var ids: [st.batch_rows]u32 = undefined;
        var total: usize = 0;
        if (windows.len > segments.len or windows.len != out.len) return error.InvalidQwenWindows;
        for (windows, 0..) |w, i| {
            const rows = w.rows();
            if (w.parents != null or w.held != 0 or rows > st.window_rows or total + rows > st.batch_rows or w.positions.len != rows) return error.UnsupportedQwenWindow;
            if (w.masks.len != 0 and w.masks.len != rows * mask_words) return error.InvalidQwenMasks;
            const cache = try self.cacheOf(w.stream);
            for (w.positions, 0..) |p, r| if (p != cache.len + r + 1) return error.InvalidQwenPosition;
            segments[i] = .{ .cache = cache, .rows = rows };
            ids[total] = w.pending;
            @memcpy(ids[total + 1 ..][0..w.tokens.len], w.tokens);
            total += rows;
        }
        try fwd.run(self.model, &self.scratch, segments[0..windows.len], ids[0..total], true, .all);
        const logits = self.scratch.logits.slice(u16, total * c.vocab);
        var base: usize = 0;
        for (windows, out) |w, o| {
            if (o.sampled.len != w.rows() or o.drafts.len != w.tokens.len) return error.InvalidQwenOutputs;
            for (o.sampled, w.positions, 0..) |*t, p, r| {
                const mask = if (w.masks.len > 0) w.masks[r * mask_words ..][0..mask_words] else &.{};
                t.* = try draw(self.gpa, logits[(base + r) * c.vocab ..][0..c.vocab], w.stream.sampling, p, mask);
            }
            @memcpy(o.drafts, w.tokens);
            base += w.rows();
        }
    }

    fn keepFn(ptr: *anyopaque, windows: []const be.Window, paths: []const []const u32) !void {
        const self = cast(ptr);
        for (windows, paths) |w, path| try (try self.cacheOf(w.stream)).keep(&self.scratch, path);
    }

    fn draftFn(_: *anyopaque, requests: []const be.DraftRequest) !void {
        for (requests) |r| if (r.depth != 0) return error.QwenCheckpointHasNoDraftHead;
    }

    fn releaseFn(ptr: *anyopaque, stream: *lanes.Stream) void {
        const self = cast(ptr);
        if (self.caches.fetchRemove(stream)) |entry| {
            entry.value.deinit();
            self.gpa.destroy(entry.value);
        }
    }
};

const draws = @import("draw.zig");
const mask_words = draws.mask_words;
pub const value = draws.value;
pub const draw = draws.draw;

test {
    _ = draws;
}
