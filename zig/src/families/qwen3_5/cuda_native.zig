//! The Qwen3.5 hybrids' CUDA lane backend (the 2B, Qwen3.8-27B) behind the native family interface (native/cuda.zig's
//! registry): windows of up to 16 rows (the lane core's copies), every stream's window in one forward, prompts in
//! chunks, draws on the device.
const std = @import("std");
const cuda = @import("cuda");
const lanes = @import("lanes");
const c = @import("config.zig");
const E = @import("cuda_engine.zig");
const draws = @import("draw.zig");
const be = lanes.backend;

pub const model_type = "qwen3_5";
pub const formats: []const []const u8 = &.{"mlx-q4g64"};
pub const default_context: i64 = 32768;
pub const max_segments: u32 = 1;
pub const prompt_rows: u32 = E.max_rows;

pub const Options = struct { context: usize, drafts: bool, segments: usize = 1, slide: bool = false };

/// What the native server drives: the lane backend, the facts its round loop reads, and how to free it.
pub const Loaded = struct {
    backend: be.Backend,
    facts: lanes.Model,
    rows: u32,
    /// Device bytes each admitted stream allocates (recurrences, convolution tails, keys and values).
    stream_bytes: usize,
    free_streams: u32 = 0,
    ctx: *anyopaque,
    deinit: *const fn (*anyopaque) void,
    /// Called with the admitted stream count: that many caches allocated and touched now, since the driver commits
    /// memory on first touch (seconds for a wide model's keys), and a request should not pay it.
    reserve: *const fn (*anyopaque, u32) anyerror!void,
};

pub fn open(gpa: std.mem.Allocator, io: std.Io, ctx: *const cuda.Context, dir: []const u8, kernels: ?[]const u8, o: Options) !Loaded {
    if (kernels != null) return error.QwenHasNoCapturedKernels;
    const e = try E.Engine.init(gpa, io, ctx, dir);
    errdefer e.deinit();
    if (e.g.vocab != c.vocab) return error.UnsupportedQwenVocab; // draw.zig's masks are sized by the 2B's vocabulary
    const l = try gpa.create(Lanes);
    l.* = .{ .gpa = gpa, .e = e, .capacity = o.context };
    return .{
        .backend = l.backend(),
        .facts = .{ .exact_width = @intCast(e.window_rows), .hidden_rows = true, .batch_rows = @intCast(e.verify_rows), .max_streams = E.max_segs, .streams_exact = true },
        .rows = @intCast(e.window_rows),
        .stream_bytes = e.streamBytes(o.context),
        .ctx = l,
        .deinit = release,
        .reserve = reserve,
    };
}

fn reserve(p: *anyopaque, streams: u32) anyerror!void {
    const l: *Lanes = @ptrCast(@alignCast(p));
    try l.pool.ensureTotalCapacity(l.gpa, streams);
    while (l.pool.items.len < streams) {
        const cache = try l.gpa.create(E.Cache);
        errdefer l.gpa.destroy(cache);
        cache.* = try E.Cache.init(l.gpa, l.e, l.capacity);
        errdefer cache.deinit(l.gpa);
        // a row through it: kernels have to have referenced a buffer once before the driver's stall is behind it
        try l.e.forward(&.{.{ .cache = cache, .rows = 1 }}, &.{0}, .last);
        try cache.reset(l.e);
        l.pool.appendAssumeCapacity(cache);
    }
}

/// A request this engine refuses, in words; null: none of its own.
pub fn explain(_: ?*anyopaque, err: anyerror) ?[]const u8 {
    return switch (err) {
        error.InvalidQwenPrompt => "the prompt and its reply exceed this server's context window: shorten it or lower max_tokens",
        error.OutOfMemory => "the GPU had no memory left for this request's caches: retry once another request ends",
        else => null,
    };
}

fn release(p: *anyopaque) void {
    const l: *Lanes = @ptrCast(@alignCast(p));
    l.deinit();
}

/// The device draw of a stream's rule at `position`: sample.cu's keyed rule, greedy at temperature 0 or without one.
fn pick(sampling: ?lanes.Sampling, position: u64) E.Draw {
    const s = sampling orelse return .{ .rule = null, .position = position };
    if (s.temperature <= 0) return .{ .rule = null, .position = position };
    return .{ .rule = cuda.sample.rule(s.seed, s.temperature, s.top_p, s.minLog(), s.top_k), .position = position };
}

const Lanes = struct {
    gpa: std.mem.Allocator,
    e: *E.Engine,
    capacity: usize,
    caches: std.AutoHashMapUnmanaged(*lanes.Stream, *E.Cache) = .empty,
    /// Caches at the full window, kept between streams: allocated and touched once (`reserve`), reset on reuse.
    pool: std.ArrayList(*E.Cache) = .empty,

    fn deinit(self: *Lanes) void {
        var it = self.caches.valueIterator();
        while (it.next()) |cache| self.drop(cache.*);
        self.caches.deinit(self.gpa);
        for (self.pool.items) |cache| self.drop(cache);
        self.pool.deinit(self.gpa);
        self.e.deinit();
        self.gpa.destroy(self);
    }

    fn drop(self: *Lanes, cache: *E.Cache) void {
        cache.deinit(self.gpa);
        self.gpa.destroy(cache);
    }

    /// A cache for a new stream: a pooled one reset, else one more (a window beyond the reserve).
    fn take(self: *Lanes) !*E.Cache {
        if (self.pool.pop()) |cache| {
            try cache.reset(self.e);
            return cache;
        }
        const cache = try self.gpa.create(E.Cache);
        errdefer self.gpa.destroy(cache);
        cache.* = try E.Cache.init(self.gpa, self.e, self.capacity);
        return cache;
    }

    fn give(self: *Lanes, cache: *E.Cache) void {
        self.pool.append(self.gpa, cache) catch self.drop(cache);
    }

    fn backend(self: *Lanes) be.Backend {
        return .{ .ptr = self, .vtable = &.{ .prefill = prefillFn, .first = firstFn, .first_masked = firstMaskedFn, .queue = queueFn, .read = readFn, .verify = verifyFn, .keep = keepFn, .draft = draftFn, .release = releaseFn } };
    }

    fn cast(ptr: *anyopaque) *Lanes {
        return @ptrCast(@alignCast(ptr));
    }

    fn cacheOf(self: *Lanes, stream: *lanes.Stream) !*E.Cache {
        return self.caches.get(stream) orelse error.UnknownQwenStream;
    }

    fn prefillFn(ptr: *anyopaque, stream: *lanes.Stream) !void {
        const self = cast(ptr);
        const ids = stream.prompt();
        if (ids.len == 0 or ids.len + stream.max_new > self.capacity or self.caches.contains(stream)) return error.InvalidQwenPrompt;
        const cache = try self.take();
        errdefer self.give(cache);
        var at: usize = 0;
        while (at < ids.len) {
            var end: usize = @min(ids.len, at + E.max_rows);
            for (stream.chunks) |boundary| if (boundary > at) {
                end = @min(end, boundary);
                break;
            };
            try self.e.forward(&.{.{ .cache = cache, .rows = end - at }}, ids[at..end], .last);
            at = end;
        }
        try self.caches.put(self.gpa, stream, cache);
    }

    fn firstFn(ptr: *anyopaque, stream: *lanes.Stream, position: u64) !u64 {
        return firstMaskedFn(ptr, stream, position, &.{});
    }

    fn firstMaskedFn(ptr: *anyopaque, stream: *lanes.Stream, position: u64, mask: []const u32) !u64 {
        const self = cast(ptr);
        const cache = try self.cacheOf(stream);
        if (position != cache.len) return error.InvalidQwenPosition;
        return try draws.draw(self.gpa, cache.logits, stream.sampling, position, mask);
    }

    fn queueFn(ptr: *anyopaque, stream: *lanes.Stream, feed: be.Feed, position: u64) !u64 {
        const self = cast(ptr);
        const cache = try self.cacheOf(stream);
        if (position != cache.len + 1) return error.InvalidQwenPosition;
        const token: u32 = switch (feed) {
            .value => |v| v,
            .handle => |h| @intCast(h),
        };
        try self.e.forward(&.{.{ .cache = cache, .rows = 1 }}, &.{token}, .{ .draws = &.{pick(stream.sampling, position)} });
        cache.last = null;
        return self.e.tokens[0];
    }

    fn readFn(_: *anyopaque, handle: u64) !u32 {
        if (handle >= c.vocab) return error.InvalidQwenToken;
        return @intCast(handle);
    }

    /// Each window (a stream's pending token and copies) checked, then as many as fit in each forward.
    fn verifyFn(ptr: *anyopaque, windows: []const be.Window, out: []be.Verified) !void {
        const self = cast(ptr);
        if (windows.len != out.len) return error.InvalidQwenWindows;
        for (windows, out) |w, o| {
            const rows = w.rows();
            if (w.parents != null or w.held != 0 or rows > self.e.window_rows or w.positions.len != rows) return error.UnsupportedQwenWindow;
            if (w.masks.len != 0 and w.masks.len != rows * draws.mask_words) return error.InvalidQwenMasks;
            if (o.sampled.len != rows or o.drafts.len != w.tokens.len) return error.InvalidQwenOutputs;
            const cache = try self.cacheOf(w.stream);
            for (w.positions, 0..) |p, r| if (p != cache.len + r + 1) return error.InvalidQwenPosition;
        }
        var at: usize = 0;
        while (at < windows.len) {
            var parts: [E.max_segs]E.Part = undefined;
            var ids: [E.max_rows]u32 = undefined;
            var picks: [E.max_rows]E.Draw = undefined;
            var n: usize = 0;
            var total: usize = 0;
            var masked = false;
            while (at + n < windows.len and n < E.max_segs and total + windows[at + n].rows() <= self.e.verify_rows) : (n += 1) {
                const w = windows[at + n];
                parts[n] = .{ .cache = self.caches.get(w.stream).?, .rows = w.rows() };
                ids[total] = w.pending;
                @memcpy(ids[total + 1 ..][0..w.tokens.len], w.tokens);
                for (w.positions, 0..) |p, r| picks[total + r] = pick(w.stream.sampling, p);
                masked = masked or w.masks.len != 0;
                total += w.rows();
            }
            // structured output draws on the host, over each row's logits with its mask applied
            try self.e.forward(parts[0..n], ids[0..total], if (masked) .logits else .{ .draws = picks[0..total] });
            var row: usize = 0;
            for (windows[at..][0..n], out[at..][0..n]) |w, o| {
                for (o.sampled, w.positions, 0..) |*t, p, r| {
                    const mask = if (w.masks.len > 0) w.masks[r * draws.mask_words ..][0..draws.mask_words] else &.{};
                    t.* = if (masked) try draws.draw(self.gpa, self.e.rowLogits(row + r), w.stream.sampling, p, mask) else self.e.tokens[row + r];
                }
                @memcpy(o.drafts, w.tokens);
                row += w.rows();
            }
            at += n;
        }
    }

    /// The accepted prefix of each window: its state as it was after that row.
    fn keepFn(ptr: *anyopaque, windows: []const be.Window, paths: []const []const u32) !void {
        const self = cast(ptr);
        for (windows, paths) |w, path| {
            for (path, 0..) |r, i| if (r != i) return error.UnsupportedQwenTree;
            try self.e.rewind(try self.cacheOf(w.stream), path.len);
        }
    }

    fn draftFn(_: *anyopaque, requests: []const be.DraftRequest) !void {
        for (requests) |r| if (r.depth != 0) return error.QwenCheckpointHasNoDraftHead;
    }

    fn releaseFn(ptr: *anyopaque, stream: *lanes.Stream) void {
        const self = cast(ptr);
        if (self.caches.fetchRemove(stream)) |entry| self.give(entry.value);
    }
};
