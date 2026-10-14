//! Gemma 4 dense behind the lane engine's backend: windows of up to 16 rows, several streams a forward, copy drafts.
const std = @import("std");
const lanes = @import("lanes");
const cfg = @import("config.zig");
const st = @import("state.zig");
const fwd = @import("forward.zig");
const Model = @import("model.zig").Model;
const draw = @import("../qwen3_5/backend.zig").draw;
const be = lanes.backend;

pub const Options = struct { capacity: usize, streams: u32 = 8 };

pub const Metal = struct {
    gpa: std.mem.Allocator,
    model: *Model,
    options: Options,
    scratch: st.Scratch,
    caches: std.AutoHashMapUnmanaged(*lanes.Stream, *st.Cache) = .empty,

    pub fn init(gpa: std.mem.Allocator, model: *Model, options: Options) !*Metal {
        const out = try gpa.create(Metal);
        errdefer gpa.destroy(out);
        out.* = .{ .gpa = gpa, .model = model, .options = options, .scratch = try st.Scratch.init(model.device, options.capacity) };
        return out;
    }

    pub fn deinit(self: *Metal) void {
        var it = self.caches.valueIterator();
        while (it.next()) |cache| {
            cache.*.deinit();
            self.gpa.destroy(cache.*);
        }
        self.caches.deinit(self.gpa);
        self.scratch.deinit();
        self.gpa.destroy(self);
    }

    /// Every kernel keeps a row's bits at any window width and beside any other stream's rows (tf-gemma4-glue).
    pub fn facts(self: *const Metal) lanes.Model {
        return .{ .exact_width = st.max_rows, .hidden_rows = true, .batch_rows = st.max_rows, .max_streams = self.options.streams, .streams_exact = true };
    }

    pub fn backend(self: *Metal) be.Backend {
        return .{ .ptr = self, .vtable = &.{ .prefill = prefillFn, .first = firstFn, .first_masked = firstMaskedFn, .queue = queueFn, .read = readFn, .verify = verifyFn, .keep = keepFn, .draft = draftFn, .release = releaseFn } };
    }

    fn cast(ptr: *anyopaque) *Metal {
        return @ptrCast(@alignCast(ptr));
    }

    fn cacheOf(self: *Metal, stream: *lanes.Stream) !*st.Cache {
        return self.caches.get(stream) orelse error.UnknownGemmaStream;
    }

    /// The last row of the scratch logits into the stream's cache.
    fn keepLast(self: *Metal, cache: *st.Cache, row: usize) void {
        @memcpy(cache.logits.slice(u16, cfg.vocab), self.scratch.logits.slice(u16, st.max_rows * cfg.vocab)[row * cfg.vocab ..][0..cfg.vocab]);
    }

    fn prefillFn(ptr: *anyopaque, stream: *lanes.Stream) !void {
        const self = cast(ptr);
        const ids = stream.prompt();
        if (ids.len == 0 or ids.len + stream.max_new > self.options.capacity or self.caches.contains(stream)) return error.InvalidGemmaPrompt;
        const cache = try self.gpa.create(st.Cache);
        errdefer self.gpa.destroy(cache);
        const capacity = @min(self.options.capacity, (ids.len + stream.max_new + st.max_rows + 63) / 64 * 64);
        cache.* = try st.Cache.init(self.model.device, capacity);
        errdefer cache.deinit();
        var at: usize = 0;
        while (at < ids.len) {
            const end = @min(ids.len, at + st.max_rows);
            try fwd.run(self.model, &self.scratch, &.{.{ .cache = cache, .rows = end - at }}, ids[at..end]);
            at = end;
        }
        self.keepLast(cache, (ids.len - 1) % st.max_rows);
        try self.caches.put(self.gpa, stream, cache);
    }

    fn firstFn(ptr: *anyopaque, stream: *lanes.Stream, position: u64) !u64 {
        const self = cast(ptr);
        const cache = try self.cacheOf(stream);
        if (position != cache.len) return error.InvalidGemmaPosition;
        return try draw(self.gpa, cache.logits.slice(u16, cfg.vocab), stream.sampling, position, &.{});
    }

    fn firstMaskedFn(ptr: *anyopaque, stream: *lanes.Stream, position: u64, mask: []const u32) !u64 {
        const self = cast(ptr);
        const cache = try self.cacheOf(stream);
        if (position != cache.len) return error.InvalidGemmaPosition;
        return try draw(self.gpa, cache.logits.slice(u16, cfg.vocab), stream.sampling, position, mask);
    }

    fn queueFn(ptr: *anyopaque, stream: *lanes.Stream, feed: be.Feed, position: u64) !u64 {
        const self = cast(ptr);
        const cache = try self.cacheOf(stream);
        if (position != cache.len + 1) return error.InvalidGemmaPosition;
        const token: u32 = switch (feed) {
            .value => |v| v,
            .handle => |h| @intCast(h),
        };
        cache.from = cache.len;
        try fwd.run(self.model, &self.scratch, &.{.{ .cache = cache, .rows = 1 }}, &.{token});
        self.keepLast(cache, 0);
        return try draw(self.gpa, cache.logits.slice(u16, cfg.vocab), stream.sampling, position, &.{});
    }

    fn readFn(_: *anyopaque, handle: u64) !u32 {
        if (handle >= cfg.vocab) return error.InvalidGemmaToken;
        return @intCast(handle);
    }

    fn verifyFn(ptr: *anyopaque, windows: []const be.Window, out: []be.Verified) !void {
        const self = cast(ptr);
        var segments: [st.max_rows]fwd.Segment = undefined;
        var ids: [st.max_rows]u32 = undefined;
        var total: usize = 0;
        if (windows.len > segments.len or windows.len != out.len) return error.InvalidGemmaWindows;
        for (windows, 0..) |w, i| {
            const rows = w.rows();
            if (w.parents != null or w.held != 0 or total + rows > st.max_rows or w.positions.len != rows) return error.UnsupportedGemmaWindow;
            if (w.masks.len != 0 and w.masks.len != rows * mask_words) return error.InvalidGemmaMasks;
            const cache = try self.cacheOf(w.stream);
            for (w.positions, 0..) |p, r| if (p != cache.len + r + 1) return error.InvalidGemmaPosition;
            cache.from = cache.len;
            segments[i] = .{ .cache = cache, .rows = rows };
            ids[total] = w.pending;
            @memcpy(ids[total + 1 ..][0..w.tokens.len], w.tokens);
            total += rows;
        }
        try fwd.run(self.model, &self.scratch, segments[0..windows.len], ids[0..total]);
        const logits = self.scratch.logits.slice(u16, total * cfg.vocab);
        var base: usize = 0;
        for (windows, out) |w, o| {
            if (o.sampled.len != w.rows() or o.drafts.len != w.tokens.len) return error.InvalidGemmaOutputs;
            for (o.sampled, w.positions, 0..) |*t, p, r| {
                const mask = if (w.masks.len > 0) w.masks[r * mask_words ..][0..mask_words] else &.{};
                t.* = try draw(self.gpa, logits[(base + r) * cfg.vocab ..][0..cfg.vocab], w.stream.sampling, p, mask);
            }
            @memcpy(o.drafts, w.tokens);
            base += w.rows();
        }
    }

    /// A chain's kept rows: the cache rolls back to where the verify began plus them; later rows are rewritten.
    fn keepFn(ptr: *anyopaque, windows: []const be.Window, paths: []const []const u32) !void {
        const self = cast(ptr);
        for (windows, paths) |w, path| {
            const cache = try self.cacheOf(w.stream);
            for (path, 0..) |r, j| if (r != j) return error.UnsupportedGemmaTree;
            if (cache.from + path.len > cache.len) return error.InvalidGemmaKeep;
            cache.len = cache.from + path.len;
        }
    }

    fn draftFn(_: *anyopaque, requests: []const be.DraftRequest) !void {
        for (requests) |r| if (r.depth != 0) return error.GemmaCheckpointHasNoDraftHead;
    }

    fn releaseFn(ptr: *anyopaque, stream: *lanes.Stream) void {
        const self = cast(ptr);
        if (self.caches.fetchRemove(stream)) |entry| {
            entry.value.deinit();
            self.gpa.destroy(entry.value);
        }
    }
};

/// A row's allowed-token bits (structured output): one bit a logit.
const mask_words = (cfg.vocab + 31) / 32;
