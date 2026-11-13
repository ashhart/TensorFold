//! Qwen recurrence, windows and keyed draws on the CPU behind the lane engine's backend interface (backend.zig's twin).
const std = @import("std");
const lanes = @import("lanes");
const c = @import("config.zig");
const st = @import("cpu_state.zig");
const fwd = @import("cpu_forward.zig");
const Model = @import("cpu_model.zig").Model;
const be = lanes.backend;
const draws = @import("draw.zig");
const mask_words = draws.mask_words;
pub const draw = draws.draw;

pub const Options = struct {
    capacity: usize,
    chunk: usize = 128,
    streams: u32 = 8,
    /// Each one-token draw's top two logits and their margin on stderr; the caller reads TF_QWEN35_CPU_TRACE once.
    trace: bool = false,
};

pub const Cpu = struct {
    gpa: std.mem.Allocator,
    model: *Model,
    options: Options,
    scratch: st.Scratch,
    width: u32 = 1,
    caches: std.AutoHashMapUnmanaged(*lanes.Stream, *st.Cache) = .empty,

    /// `wide`: verify windows of up to 16 rows once the local exactness check passes; else one-row rounds.
    pub fn init(gpa: std.mem.Allocator, model: *Model, options: Options, wide: bool) !*Cpu {
        const out = try gpa.create(Cpu);
        errdefer gpa.destroy(out);
        out.* = .{ .gpa = gpa, .model = model, .options = options, .scratch = try st.Scratch.init(gpa, @max(options.chunk, st.batch_rows), options.capacity, model.pool.workers()) };
        errdefer out.scratch.deinit();
        if (!wide) return out;
        @import("cpu_qualify.zig").check(model, &out.scratch) catch |err| switch (err) {
            error.QwenCacheLengthMismatch, error.QwenRecurrenceMismatch, error.QwenAttentionCacheMismatch, error.QwenWindowLogitsMismatch, error.QwenCommittedLogitsMismatch => {
                std.log.warn("Qwen wider lanes failed the local exactness check ({s}); keeping one-row lane rounds", .{@errorName(err)});
                return out;
            },
            else => return err,
        };
        out.width = st.window_rows;
        return out;
    }

    pub fn deinit(self: *Cpu) void {
        var it = self.caches.valueIterator();
        while (it.next()) |cache| {
            cache.*.deinit();
            self.gpa.destroy(cache.*);
        }
        self.caches.deinit(self.gpa);
        self.scratch.deinit();
        self.gpa.destroy(self);
    }

    pub fn facts(self: *const Cpu) lanes.Model {
        return .{ .exact_width = self.width, .hidden_rows = self.width > 1, .batch_rows = st.batch_rows, .max_streams = self.options.streams, .streams_exact = true };
    }

    pub fn backend(self: *Cpu) be.Backend {
        return .{ .ptr = self, .vtable = &.{ .prefill = prefillFn, .first = firstFn, .first_masked = firstMaskedFn, .queue = queueFn, .read = readFn, .verify = verifyFn, .keep = keepFn, .draft = draftFn, .release = releaseFn, .reports_fill = true } };
    }

    fn cast(ptr: *anyopaque) *Cpu {
        return @ptrCast(@alignCast(ptr));
    }

    fn cacheOf(self: *Cpu, stream: *lanes.Stream) !*st.Cache {
        return self.caches.get(stream) orelse error.UnknownQwenStream;
    }

    pub fn newCache(self: *Cpu, capacity: usize) !*st.Cache {
        const cache = try self.gpa.create(st.Cache);
        errdefer self.gpa.destroy(cache);
        cache.* = try st.Cache.init(self.gpa, capacity);
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
        try self.caches.put(self.gpa, stream, cache);
        errdefer _ = self.caches.remove(stream);
        try @import("prefill.zig").run(cache, stream, self.options.chunk, self, promptChunk, st);
    }

    fn promptChunk(self: *Cpu, cache: *st.Cache, ids: []const u32) !void {
        try fwd.run(self.model, &self.scratch, &.{.{ .cache = cache, .rows = ids.len }}, ids, false, .last);
    }

    fn firstFn(ptr: *anyopaque, stream: *lanes.Stream, position: u64) !u64 {
        const self = cast(ptr);
        const cache = try self.cacheOf(stream);
        if (position != cache.len) return error.InvalidQwenPosition;
        const drawn = try draw(self.gpa, cache.logits, stream.sampling, position, &.{});
        if (self.options.trace) trace(cache.logits, position, drawn);
        return drawn;
    }

    fn firstMaskedFn(ptr: *anyopaque, stream: *lanes.Stream, position: u64, mask: []const u32) !u64 {
        const self = cast(ptr);
        const cache = try self.cacheOf(stream);
        if (position != cache.len) return error.InvalidQwenPosition;
        return try draw(self.gpa, cache.logits, stream.sampling, position, mask);
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
        const drawn = try draw(self.gpa, cache.logits, stream.sampling, position, &.{});
        if (self.options.trace) trace(cache.logits, position, drawn);
        return drawn;
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
        const logits = self.scratch.logits[0 .. total * c.vocab];
        var base: usize = 0;
        for (windows, out) |w, o| {
            if (o.sampled.len != w.rows() or o.drafts.len != w.tokens.len) return error.InvalidQwenOutputs;
            for (o.sampled, w.positions, 0..) |*t, p, r| {
                const mask = if (w.masks.len > 0) w.masks[r * mask_words ..][0..mask_words] else &.{};
                t.* = try draw(self.gpa, logits[(base + r) * c.vocab ..][0..c.vocab], w.stream.sampling, p, mask);
                if (self.options.trace) trace(logits[(base + r) * c.vocab ..][0..c.vocab], p, t.*);
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

fn trace(logits: []const u16, position: u64, drawn: u64) void {
    var best: usize = 0;
    var second: ?usize = null;
    for (logits, 0..) |word, id| {
        const x = draws.value(word);
        if (x > draws.value(logits[best])) {
            second = best;
            best = id;
        } else if (id != best and (second == null or x > draws.value(logits[second.?]))) second = id;
    }
    const top = draws.value(logits[best]);
    const runner = if (second) |s| draws.value(logits[s]) else -std.math.inf(f64);
    std.debug.print("qwen3.5 cpu: position {d} drew {d}: top {d} at {d:.4}, next {d} at {d:.4}, margin {d:.4}\n", .{ position, drawn, best, top, second orelse 0, runner, top - runner });
}

/// The prompt cache's saved-state hooks: a committed prompt's bytes in host memory.
pub const Snaps = struct {
    fn cpu(ptr: *anyopaque) *Cpu {
        return @ptrCast(@alignCast(ptr));
    }
    pub fn bytes(_: *anyopaque, at: u32) u64 {
        return st.snapBytes(at);
    }
    pub fn save(ptr: *anyopaque, owner: ?*anyopaque, at: u32) anyerror!*anyopaque {
        const self = cpu(ptr);
        const cache = try self.cacheOf(@ptrCast(@alignCast(owner orelse return error.NoStream)));
        return @ptrCast(try st.save(self.gpa, cache, at));
    }
    pub fn restore(_: *anyopaque, _: ?*anyopaque, _: *anyopaque) anyerror!void {
        return error.RestoredInPrefill;
    }
    pub fn drop(ptr: *anyopaque, saved: *anyopaque) void {
        st.drop(cpu(ptr).gpa, @ptrCast(@alignCast(saved)));
    }
};

test "zero grant: an attached Qwen drafter resumes after a long shared skip on the CPU backend's facts" {
    var b: Cpu = undefined;
    b.width = 16;
    b.options = .{ .capacity = 512, .streams = 16 };
    try lanes.shared_draft_check.check(b.facts(), true);
}
