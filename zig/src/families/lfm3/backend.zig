//! LFM3 convolution, full/sliding attention and keyed draws behind the shared lane engine.
const std = @import("std");
const platform = @import("sys");
const mtl = @import("metal");
const lanes = @import("lanes");
const c = @import("config.zig");
const st = @import("state.zig");
const fwd = @import("forward.zig");
const snapshot = @import("snapshot.zig");
const Model = @import("model.zig").Model;
const be = lanes.backend;
const prefill_control = @import("prefill_control.zig");
const steps = @import("prefill_steps.zig");
const image_rows = @import("image_rows.zig");
const image_fill = @import("image_fill.zig");
pub const greedy = @import("greedy.zig");

pub const Options = struct { capacity: usize, chunk: usize = 128, streams: u32 = 8 };

pub const Metal = struct {
    residency: ?mtl.ResidencySet.Follow = null,
    gpa: std.mem.Allocator,
    model: *Model,
    options: Options,
    scratch: st.Scratch,
    images: ?@import("image_rows.zig").Provider = null,
    width: u32 = 1,
    caches: std.AutoHashMapUnmanaged(*lanes.Stream, *st.Cache) = .empty,
    /// Forward ms by window width, timed at load: the lane core weighs drafts against them.
    costs: [st.window_rows]lanes.config.Cost = undefined,
    timed: usize = 0,
    /// Streams whose prompt passes are in progress: each one's next row and its images, prepared once (#3021).
    fills: std.AutoHashMapUnmanaged(*lanes.Stream, Pass) = .empty,

    /// A prompt pass in progress: the next chunk's first row and the images it opened.
    const Fill = image_fill.Fill(image_rows.Provider, image_rows.Images);
    const Pass = struct { at: usize, fill: Fill };

    pub fn init(gpa: std.mem.Allocator, model: *Model, options: Options) !*Metal {
        if (options.chunk == 0 or options.capacity == 0 or options.capacity > st.max_context or options.streams == 0)
            return error.InvalidLfmOptions;
        const out = try gpa.create(Metal);
        errdefer gpa.destroy(out);
        out.* = .{ .gpa = gpa, .model = model, .options = options, .scratch = try st.Scratch.init(gpa, model.device, &model.config.layer_types, @max(options.chunk, st.batch_rows)) };
        errdefer out.scratch.deinit();
        try @import("qualify.zig").check(model, &out.scratch);
        out.width = st.window_rows;
        out.timed = try out.time();
        return out;
    }

    pub fn deinit(self: *Metal) void {
        mtl.resident.streams.set(self, null);
        while (self.fills.count() > 0) {
            var keys = self.fills.keyIterator();
            self.closing(keys.next().?.*);
        }
        var it = self.caches.valueIterator();
        while (it.next()) |cache| {
            cache.*.deinit();
            self.gpa.destroy(cache.*);
        }
        self.caches.deinit(self.gpa);
        self.fills.deinit(self.gpa);
        self.scratch.deinit();
        self.gpa.destroy(self);
    }

    pub fn facts(self: *const Metal) lanes.Model {
        return factsFor(self.width, self.options.streams, self.costs[0..self.timed]);
    }

    /// Each window width's forward ms on a 256-token history (the fastest of three, after the GPU's clocks ramp up).
    fn time(self: *Metal) !usize {
        var cache = try st.Cache.init(self.gpa, self.model.device, &self.model.config.layer_types, 512);
        defer cache.deinit();
        var ids: [256]u32 = undefined;
        for (&ids, 0..) |*id, i| id.* = @intCast((i * 7919 + 13) % c.vocab);
        var at: usize = 0;
        while (at < ids.len) {
            const end = @min(ids.len, at + self.scratch.rows);
            try fwd.run(self.model, &self.scratch, &.{.{ .cache = &cache, .rows = end - at }}, ids[at..end], false, .last);
            at = end;
        }
        const saved = try snapshot.save(self.gpa, &self.model.config.layer_types, &cache, @intCast(cache.len));
        defer snapshot.drop(self.gpa, saved);
        for (0..8) |_| {
            try fwd.run(self.model, &self.scratch, &.{.{ .cache = &cache, .rows = 1 }}, ids[0..1], true, .all);
            cache.len = 0;
            try snapshot.restore(&cache, saved);
        }
        for (&self.costs, 1..) |*cost, w| {
            var best: f64 = std.math.floatMax(f64);
            for (0..3) |_| {
                var t0: platform.c.timespec = undefined;
                var t1: platform.c.timespec = undefined;
                _ = platform.c.clock_gettime(platform.c.CLOCK.MONOTONIC, &t0);
                try fwd.run(self.model, &self.scratch, &.{.{ .cache = &cache, .rows = w }}, ids[0..w], true, .all);
                _ = platform.c.clock_gettime(platform.c.CLOCK.MONOTONIC, &t1);
                cache.len = 0;
                try snapshot.restore(&cache, saved);
                const ms = @as(f64, @floatFromInt(t1.sec - t0.sec)) * 1e3 + @as(f64, @floatFromInt(t1.nsec - t0.nsec)) / 1e6;
                best = @min(best, ms);
            }
            cost.* = .{ .width = @intCast(w), .ms = best };
        }
        std.log.info("LFM3 window ms by width: 1 {d:.1}, 2 {d:.1}, 4 {d:.1}, 8 {d:.1}, 16 {d:.1}", .{ self.costs[0].ms, self.costs[1].ms, self.costs[3].ms, self.costs[7].ms, self.costs[15].ms });
        return self.costs.len;
    }

    pub fn backend(self: *Metal) be.Backend {
        return .{ .ptr = self, .vtable = &.{ .prefill = prefillFn, .prefill_step = prefillStepFn, .cancel_prefill = cancelPrefillFn, .first = firstFn, .first_masked = firstMaskedFn, .queue = queueFn, .read = readFn, .verify = verifyFn, .keep = keepFn, .draft = draftFn, .release = releaseFn, .reports_fill = true } };
    }

    fn cast(ptr: *anyopaque) *Metal {
        return @ptrCast(@alignCast(ptr));
    }

    fn cacheOf(self: *Metal, stream: *lanes.Stream) !*st.Cache {
        return self.caches.get(stream) orelse error.UnknownLfmStream;
    }

    pub fn newCache(self: *Metal, capacity: usize) !*st.Cache {
        const cache = try self.gpa.create(st.Cache);
        errdefer self.gpa.destroy(cache);
        cache.* = try st.Cache.init(self.gpa, self.model.device, &self.model.config.layer_types, capacity);
        cache.memory.setResident(self.residency);
        return cache;
    }

    fn prefillFn(ptr: *anyopaque, stream: *lanes.Stream) !void {
        const self = cast(ptr);
        const p = try self.opening(stream, true);
        errdefer self.drop(stream);
        const r: Run = .{ .m = self, .cache = try self.cacheOf(stream), .fill = &p.fill };
        while (!try steps.step(r, stream, &p.at, self.options.chunk, true)) {}
        self.closing(stream);
        try prefill_control.finished(stream);
    }

    /// One chunk's forward for `steps.step`, and a kept mark's hook.
    const Run = struct {
        m: *Metal,
        cache: *st.Cache,
        fill: *Fill,

        pub fn run(r: Run, stream: *lanes.Stream, at: usize, end: usize) !void {
            const ids = stream.prompt();
            const images = try r.fill.chunk(at, end);
            try fwd.runWithImages(r.m.model, &r.m.scratch, &.{.{ .cache = r.cache, .rows = end - at }}, ids[at..end], false, .last, images);
        }

        pub fn keep(_: Run, stream: *lanes.Stream, at: u32) void {
            if (stream.reuse.hook) |hk| hk.at(hk.ptr, stream, at);
        }
    };

    /// The stream's cache made, a kept state restored and its images prepared: its pass from the returned row.
    fn opening(self: *Metal, stream: *lanes.Stream, whole: bool) !*Pass {
        const ids = stream.prompt();
        if (stream.images.len != 0 and self.images == null) return error.Lfm3VisionNotImplemented;
        if (ids.len == 0 or ids.len + stream.max_new > self.options.capacity or self.caches.contains(stream)) return error.InvalidLfmPrompt;
        // a step leaves yielding to the lane host, which runs other streams' rounds between chunks
        if (whole) try prefill_control.before(stream, 0) else if (stream.isCancelled()) return error.Cancelled;
        const capacity = @min(self.options.capacity, ids.len + stream.max_new + st.window_rows);
        const cache = try self.newCache(capacity);
        errdefer self.gpa.destroy(cache);
        errdefer cache.deinit();
        try self.caches.put(self.gpa, stream, cache);
        errdefer _ = self.caches.remove(stream);
        var at: usize = 0;
        stream.cached = 0;
        if (stream.reuse.saved) |saved| { // a kept state of this prompt's prefix: the pass starts there
            const snap: *snapshot.Snap = @ptrCast(@alignCast(saved));
            if (snap.at < ids.len) if (snapshot.restore(cache, snap)) |_| {
                at = snap.at;
            } else |_| {};
            stream.cached = @intCast(at);
            stream.reuse_failed = at == 0;
        }
        prefill_control.completed(stream, at);
        var fill: Fill = .{ .provider = self.images, .stream = stream };
        try fill.begin(at);
        errdefer fill.close();
        try self.fills.put(self.gpa, stream, .{ .at = at, .fill = fill });
        return self.fills.getPtr(stream).?;
    }

    /// The pass is over: its fill and its images' projected rows go.
    fn closing(self: *Metal, stream: *lanes.Stream) void {
        var pass = self.fills.fetchRemove(stream) orelse return;
        pass.value.fill.close();
    }

    /// A pass that failed: its fill and its cache go, as the whole pass's error left them.
    fn drop(self: *Metal, stream: *lanes.Stream) void {
        self.closing(stream);
        if (self.caches.fetchRemove(stream)) |entry| {
            entry.value.deinit();
            self.gpa.destroy(entry.value);
        }
    }

    /// One chunk a call at the whole pass's cuts (`max_rows` is never set: this family declares no chunk_invariant).
    fn prefillStepFn(ptr: *anyopaque, stream: *lanes.Stream, _: u32) !bool {
        const self = cast(ptr);
        const p = self.fills.getPtr(stream) orelse try self.opening(stream, false);
        errdefer self.drop(stream);
        const r: Run = .{ .m = self, .cache = try self.cacheOf(stream), .fill = &p.fill };
        const done = try steps.step(r, stream, &p.at, self.options.chunk, false);
        if (done) self.closing(stream);
        return done;
    }

    /// The host cancels between chunks: the last chunk's state is kept (its forward has waited; no prompt work runs).
    fn cancelPrefillFn(ptr: *anyopaque, stream: *lanes.Stream) !void {
        const self = cast(ptr);
        steps.cancel(stream, if (self.fills.get(stream)) |p| p.at else null);
    }

    fn firstFn(ptr: *anyopaque, stream: *lanes.Stream, position: u64) !u64 {
        const self = cast(ptr);
        const cache = try self.cacheOf(stream);
        if (position != cache.len) return error.InvalidLfmPosition;
        return try draw(self.gpa, cache.logits.slice(u16, c.vocab), stream.sampling, position, &.{});
    }

    fn firstMaskedFn(ptr: *anyopaque, stream: *lanes.Stream, position: u64, mask: []const u32) !u64 {
        const self = cast(ptr);
        const cache = try self.cacheOf(stream);
        if (position != cache.len) return error.InvalidLfmPosition;
        return try draw(self.gpa, cache.logits.slice(u16, c.vocab), stream.sampling, position, mask);
    }

    fn queueFn(ptr: *anyopaque, stream: *lanes.Stream, feed: be.Feed, position: u64) !u64 {
        const self = cast(ptr);
        const cache = try self.cacheOf(stream);
        if (position != cache.len + 1) return error.InvalidLfmPosition;
        const token: u32 = switch (feed) {
            .value => |v| v,
            .handle => |h| std.math.cast(u32, h) orelse return error.InvalidLfmToken,
        };
        try fwd.run(self.model, &self.scratch, &.{.{ .cache = cache, .rows = 1 }}, &.{token}, false, .all);
        const best = self.scratch.best.slice(u32, 2);
        const greedy_row = if (stream.sampling) |settings| settings.temperature <= 0 else true;
        if (greedy_row and best[1] == 0) return best[0];
        return try draw(self.gpa, cache.logits.slice(u16, c.vocab), stream.sampling, position, &.{});
    }

    fn readFn(_: *anyopaque, handle: u64) !u32 {
        if (handle >= c.vocab) return error.InvalidLfmToken;
        return @intCast(handle);
    }

    fn verifyFn(ptr: *anyopaque, windows: []const be.Window, out: []be.Verified) !void {
        const self = cast(ptr);
        var segments: [st.batch_rows]fwd.Segment = undefined;
        var ids: [st.batch_rows]u32 = undefined;
        var total: usize = 0;
        if (windows.len > segments.len or windows.len != out.len) return error.InvalidLfmWindows;
        for (windows, out) |w, o| {
            if (o.sampled.len != w.rows() or o.drafts.len != w.tokens.len) return error.InvalidLfmOutputs;
        }
        for (windows, 0..) |w, i| {
            const rows = w.rows();
            if (w.parents != null or w.held != 0 or rows > st.window_rows or total + rows > st.batch_rows or w.positions.len != rows) return error.UnsupportedLfmWindow;
            if (w.masks.len != 0 and w.masks.len != rows * mask_words) return error.InvalidLfmMasks;
            const cache = try self.cacheOf(w.stream);
            for (w.positions, 0..) |p, r| if (p != cache.len + r + 1) return error.InvalidLfmPosition;
            segments[i] = .{ .cache = cache, .rows = rows };
            ids[total] = w.pending;
            @memcpy(ids[total + 1 ..][0..w.tokens.len], w.tokens);
            total += rows;
        }
        try fwd.run(self.model, &self.scratch, segments[0..windows.len], ids[0..total], true, .all);
        const logits = self.scratch.logits.slice(u16, total * c.vocab);
        const best = self.scratch.best.slice(u32, 2 * total);
        var base: usize = 0;
        for (windows, out) |w, o| {
            if (o.sampled.len != w.rows() or o.drafts.len != w.tokens.len) return error.InvalidLfmOutputs;
            for (o.sampled, w.positions, 0..) |*t, p, r| {
                const mask = if (w.masks.len > 0) w.masks[r * mask_words ..][0..mask_words] else &.{};
                // A greedy row takes the GPU's pick (greedy.select's); a nonfinite row errors in draw.
                const greedy_row = if (w.stream.sampling) |s| s.temperature <= 0 else true;
                if (greedy_row and mask.len == 0 and best[2 * (base + r) + 1] == 0) {
                    t.* = best[2 * (base + r)];
                    continue;
                }
                t.* = try draw(self.gpa, logits[(base + r) * c.vocab ..][0..c.vocab], w.stream.sampling, p, mask);
            }
            @memcpy(o.drafts, w.tokens);
            base += w.rows();
        }
    }

    fn keepFn(ptr: *anyopaque, windows: []const be.Window, paths: []const []const u32) !void {
        const self = cast(ptr);
        if (windows.len != paths.len) return error.InvalidLfmWindows;
        for (windows, paths) |w, path| try (try self.cacheOf(w.stream)).keep(&self.scratch, path);
    }

    fn draftFn(_: *anyopaque, requests: []const be.DraftRequest) !void {
        for (requests) |r| if (r.depth != 0) return error.LfmCheckpointHasNoDraftHead;
    }

    fn releaseFn(ptr: *anyopaque, stream: *lanes.Stream) void {
        const self = cast(ptr);
        self.closing(stream); // a stream that leaves mid-pass
        if (self.caches.fetchRemove(stream)) |entry| {
            entry.value.deinit();
            self.gpa.destroy(entry.value);
        }
    }
};

/// The prompt cache's view of the streams: kept states copied at chunk ends, restored by the prompt pass.
pub const Snaps = struct {
    fn metal(ptr: *anyopaque) *Metal {
        return @ptrCast(@alignCast(ptr));
    }
    pub fn bytes(ptr: *anyopaque, at: u32) u64 {
        return snapshot.bytes(&metal(ptr).model.config.layer_types, at);
    }
    pub fn save(ptr: *anyopaque, owner: ?*anyopaque, at: u32) anyerror!*anyopaque {
        const self = metal(ptr);
        const cache = try self.cacheOf(@ptrCast(@alignCast(owner orelse return error.NoStream)));
        return @ptrCast(try snapshot.save(self.gpa, &self.model.config.layer_types, cache, at));
    }
    pub fn restore(_: *anyopaque, _: ?*anyopaque, _: *anyopaque) anyerror!void {
        return error.RestoredInPrefill; // the prompt pass restores, before its first chunk
    }
    pub fn drop(ptr: *anyopaque, saved: *anyopaque) void {
        snapshot.drop(metal(ptr).gpa, @ptrCast(@alignCast(saved)));
    }
};

/// A row's allowed-token bits (structured output): one bit a logit.
const mask_words = (c.vocab + 31) / 32;

pub const value = lanes.sampling_row.value;

/// Greedy unmasked rows take greedy.select; the rest core's row draw (lanes.sampling_row) under the LFM error names.
pub fn draw(gpa: std.mem.Allocator, logits: []const u16, settings: ?lanes.Sampling, position: u64, mask: []const u32) !u32 {
    if (mask.len == 0 and (settings == null or settings.?.temperature <= 0)) return greedy.select(logits);
    return lanes.sampling_row.draw(gpa, logits, settings, position, mask) catch |e| switch (e) {
        error.NonfiniteLogits => error.NonfiniteLfmLogits,
        error.NoAllowedToken => error.NoAllowedLfmToken,
        else => e,
    };
}

/// The shared lane facts after the load-time exactness gate succeeds.
fn factsFor(width: u32, streams: u32, costs: []const lanes.config.Cost) lanes.Model {
    return .{
        .exact_width = width,
        .hidden_rows = width > 1,
        .batch_rows = st.batch_rows,
        .max_streams = streams,
        .streams_exact = true,
        .window_costs = costs,
        .plain_guard = true,
    };
}

test "qualified LFM3 windows share rounds across streams with drafts enabled" {
    const gpa = std.testing.allocator;
    var wide = try lanes.Config.init(gpa, factsFor(st.window_rows, 8, &.{}), st.window_rows, st.window_rows - 1);
    defer wide.deinit(gpa);
    try std.testing.expect(wide.family_streams and wide.batch_streams == 8 and wide.most_drafts > 0);
}

test "candidate selection preserves full-vocabulary keyed draws and tie order" {
    const gpa = std.testing.allocator;
    const words = try gpa.alloc(u16, c.vocab);
    defer gpa.free(words);
    const values = try gpa.alloc(f64, c.vocab);
    defer gpa.free(values);
    const ids = try gpa.alloc(u64, c.vocab);
    defer gpa.free(ids);
    for (words, values, ids, 0..) |*w, *v, *id, i| {
        const x: f32 = @as(f32, @floatFromInt((i * 37) % 257)) / 8.0 - 16.0;
        w.* = @intCast(@as(u32, @bitCast(x)) >> 16);
        v.* = value(w.*);
        id.* = i;
    }
    for ([_]u32{ 0, 1, 20, 257, c.vocab }) |k| {
        for ([_]u64{ 0, 128, 65536 }) |position| {
            const s: lanes.Sampling = .{ .seed = 1234, .top_k = k, .temperature = 0.7, .top_p = 0.95, .min_p = 0.1 };
            const want = try lanes.sampling.choose(gpa, values, ids, position, s);
            try std.testing.expectEqual(want, try draw(gpa, words, s, position, &.{}));
        }
    }
    try std.testing.expectEqual(try draw(gpa, words, null, 0, &.{}), try draw(gpa, words, .{ .seed = 7, .temperature = 0 }, 99, &.{}));
    words[0] = 0x7fc0;
    try std.testing.expectError(error.NonfiniteLfmLogits, draw(gpa, words, null, 0, &.{}));
    try std.testing.expectError(error.NonfiniteLfmLogits, draw(gpa, words, .{ .seed = 7, .temperature = 0.7 }, 0, &.{}));
}

test "a masked draw is the full-vocabulary draw with the other tokens at -inf" {
    const gpa = std.testing.allocator;
    const words = try gpa.alloc(u16, c.vocab);
    defer gpa.free(words);
    const values = try gpa.alloc(f64, c.vocab);
    defer gpa.free(values);
    const ids = try gpa.alloc(u64, c.vocab);
    defer gpa.free(ids);
    var mask: [mask_words]u32 = @splat(0);
    for (words, values, ids, 0..) |*w, *v, *id, i| {
        const x: f32 = @as(f32, @floatFromInt((i * 37) % 257)) / 8.0 - 16.0;
        w.* = @intCast(@as(u32, @bitCast(x)) >> 16);
        const on = i % 7 == 3 or i % 1000 == 1;
        if (on) mask[i / 32] |= @as(u32, 1) << @intCast(i % 32);
        v.* = if (on) value(w.*) else -std.math.inf(f64);
        id.* = i;
    }
    for ([_]u32{ 0, 1, 20, 257, c.vocab }) |k| {
        for ([_]u64{ 0, 128, 65536 }) |position| {
            const s: lanes.Sampling = .{ .seed = 99, .top_k = k, .temperature = 0.8, .top_p = 0.9, .min_p = 0.05 };
            const want = try lanes.sampling.choose(gpa, values, ids, position, s);
            try std.testing.expectEqual(want, try draw(gpa, words, s, position, &mask));
        }
    }
    try std.testing.expectEqual(try lanes.sampling.choose(gpa, values, ids, 0, .{ .seed = 1, .temperature = 0 }), try draw(gpa, words, null, 0, &mask));
    @memset(&mask, 0);
    try std.testing.expectError(error.NoAllowedLfmToken, draw(gpa, words, null, 0, &mask));
}
