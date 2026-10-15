//! Step-5 behind the lane engine's backend: one-row rounds, a cache a stream, draws by the shared keyed sampler.
const std = @import("std");
const lanes = @import("lanes");
const st = @import("state.zig");
const fwd = @import("forward.zig");
const Model = @import("model.zig").Model;
const draw = @import("../qwen3_5/backend.zig").draw;
const be = lanes.backend;

/// The longest context a stream may take, the streams at once, and the bytes every stream's caches may hold.
pub const Options = struct { capacity: usize, streams: u32 = 4, budget: u64 };

pub const Metal = struct {
    gpa: std.mem.Allocator,
    model: *Model,
    options: Options,
    scratch: st.Scratch,
    caches: std.AutoHashMapUnmanaged(*lanes.Stream, *st.Cache) = .empty,
    held: u64 = 0, // bytes the streams' caches hold

    pub fn init(gpa: std.mem.Allocator, model: *Model, options: Options) !*Metal {
        const out = try gpa.create(Metal);
        errdefer gpa.destroy(out);
        out.* = .{ .gpa = gpa, .model = model, .options = options, .scratch = try st.Scratch.init(model.device, &model.config, options.capacity) };
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

    pub fn facts(self: *const Metal) lanes.Model {
        return .{ .exact_width = 1, .max_streams = self.options.streams, .streams_exact = true };
    }

    pub fn backend(self: *Metal) be.Backend {
        return .{ .ptr = self, .vtable = &.{ .prefill = prefillFn, .first = firstFn, .first_masked = firstMaskedFn, .queue = queueFn, .read = readFn, .verify = verifyFn, .keep = keepFn, .draft = draftFn, .release = releaseFn, .first_row = firstRowFn } };
    }

    fn cast(ptr: *anyopaque) *Metal {
        return @ptrCast(@alignCast(ptr));
    }

    fn cacheOf(self: *Metal, stream: *lanes.Stream) !*st.Cache {
        return self.caches.get(stream) orelse error.UnknownStep5Stream;
    }

    fn logits(self: *Metal, cache: *st.Cache) []const u16 {
        return cache.logits.slice(u16, self.model.config.vocab);
    }

    /// Run `ids` at the cache's end and keep them.
    fn feed(self: *Metal, cache: *st.Cache, ids: []const u32) !void {
        var at: usize = 0;
        while (at < ids.len) {
            const end = @min(ids.len, at + st.window_rows);
            try fwd.run(self.model, &self.scratch, cache, ids[at..end]);
            cache.len += end - at;
            at = end;
        }
    }

    fn prefillFn(ptr: *anyopaque, stream: *lanes.Stream) !void {
        const self = cast(ptr);
        const ids = stream.prompt();
        if (ids.len == 0 or ids.len + stream.max_new > self.options.capacity or self.caches.contains(stream)) return error.InvalidStep5Prompt;
        const capacity = ids.len + stream.max_new + 1;
        const need = st.cacheBytes(&self.model.config, capacity);
        if (self.held + need > self.options.budget) {
            std.log.warn("step5: a {d}-token stream needs {d:.1} GiB of caches; {d:.1} of {d:.1} GiB are held", .{ capacity, gib(need), gib(self.held), gib(self.options.budget) });
            return error.Step5CachesFull;
        }
        const cache = try self.gpa.create(st.Cache);
        errdefer self.gpa.destroy(cache);
        cache.* = try st.Cache.init(self.model.device, &self.model.config, capacity);
        errdefer cache.deinit();
        self.held += need;
        errdefer self.held -= need;
        try self.feed(cache, ids);
        try self.caches.put(self.gpa, stream, cache);
    }

    fn firstFn(ptr: *anyopaque, stream: *lanes.Stream, position: u64) !u64 {
        const self = cast(ptr);
        const cache = try self.cacheOf(stream);
        if (position != cache.len) return error.InvalidStep5Position;
        return try draw(self.gpa, self.logits(cache), stream.sampling, position, &.{});
    }

    fn firstMaskedFn(ptr: *anyopaque, stream: *lanes.Stream, position: u64, mask: []const u32) !u64 {
        const self = cast(ptr);
        const cache = try self.cacheOf(stream);
        if (position != cache.len) return error.InvalidStep5Position;
        return try draw(self.gpa, self.logits(cache), stream.sampling, position, mask);
    }

    fn queueFn(ptr: *anyopaque, stream: *lanes.Stream, f: be.Feed, position: u64) !u64 {
        const self = cast(ptr);
        const cache = try self.cacheOf(stream);
        if (position != cache.len + 1) return error.InvalidStep5Position;
        const token: u32 = switch (f) {
            .value => |v| v,
            .handle => |h| @intCast(h),
        };
        try self.feed(cache, &.{token});
        return try draw(self.gpa, self.logits(cache), stream.sampling, position, &.{});
    }

    fn firstRowFn(ptr: *anyopaque, stream: *lanes.Stream) !lanes.LogprobRow {
        const self = cast(ptr);
        const l = self.logits(try self.cacheOf(stream));
        return row(l, @intCast(argmax(l)), stream.logprobs orelse 0);
    }

    fn readFn(ptr: *anyopaque, handle: u64) !u32 {
        if (handle >= cast(ptr).model.config.vocab) return error.InvalidStep5Token;
        return @intCast(handle);
    }

    /// One-row windows (the exact width is 1): each stream's pending token runs and is kept; `keep` rolls back.
    fn verifyFn(ptr: *anyopaque, windows: []const be.Window, out: []be.Verified) !void {
        const self = cast(ptr);
        const words = (self.model.config.vocab + 31) / 32;
        if (windows.len != out.len) return error.InvalidStep5Windows;
        for (windows, out) |w, o| {
            if (w.rows() != 1 or w.parents != null or o.sampled.len != 1 or o.drafts.len != 0) return error.UnsupportedStep5Window;
            const cache = try self.cacheOf(w.stream);
            if (w.positions[0] != cache.len + 1) return error.InvalidStep5Position;
            try self.feed(cache, &.{w.pending});
            const mask = if (w.masks.len >= words) w.masks[0..words] else &.{};
            o.sampled[0] = try draw(self.gpa, self.logits(cache), w.stream.sampling, w.positions[0], mask);
            if (o.rows.len == 1) o.rows[0] = row(self.logits(cache), o.sampled[0], w.stream.logprobs orelse 0);
        }
    }

    fn keepFn(ptr: *anyopaque, windows: []const be.Window, paths: []const []const u32) !void {
        const self = cast(ptr);
        for (windows, paths) |w, path| {
            const cache = try self.cacheOf(w.stream);
            if (path.len > w.rows() or w.rows() > cache.len) return error.UnsupportedStep5Window;
            cache.len = cache.len - w.rows() + path.len;
        }
    }

    fn draftFn(_: *anyopaque, requests: []const be.DraftRequest) !void {
        for (requests) |r| if (r.depth != 0) return error.Step5HasNoDraftHead;
    }

    fn releaseFn(ptr: *anyopaque, stream: *lanes.Stream) void {
        const self = cast(ptr);
        if (self.caches.fetchRemove(stream)) |entry| {
            self.held -= st.cacheBytes(&self.model.config, entry.value.capacity);
            entry.value.deinit();
            self.gpa.destroy(entry.value);
        }
    }
};

fn gib(bytes: u64) f64 {
    return @as(f64, @floatFromInt(bytes)) / (1 << 30);
}

fn logit(word: u16) f32 {
    return @bitCast(@as(u32, word) << 16);
}

fn argmax(l: []const u16) usize {
    var best: usize = 0;
    for (l, 0..) |w, i| if (logit(w) > logit(l[best])) {
        best = i;
    };
    return best;
}

/// The logprob row of bf16 logits `l` for `pick`: an fp64 normalizer, `count` best ids, ties to the lower id.
pub fn row(l: []const u16, pick: u32, count: u8) lanes.LogprobRow {
    const top = lanes.logprob.max_top;
    var most: f64 = -std.math.inf(f64);
    for (l) |w| most = @max(most, logit(w));
    var sum: f64 = 0;
    for (l) |w| sum += @exp(@as(f64, logit(w)) - most);
    const lse: f32 = @floatCast(most + @log(sum));
    var r: lanes.LogprobRow = .{ .token = pick, .logprob = logit(l[pick]) - lse, .count = top };
    var filled: usize = 0;
    for (l, 0..) |w, id| {
        const x = logit(w);
        if (filled == top and x <= r.logprobs[top - 1] + lse) continue;
        var at = @min(filled, top - 1);
        while (at > 0 and x > r.logprobs[at - 1] + lse) : (at -= 1) {
            r.ids[at] = r.ids[at - 1];
            r.logprobs[at] = r.logprobs[at - 1];
        }
        r.ids[at] = @intCast(id);
        r.logprobs[at] = x - lse;
        filled = @min(filled + 1, top);
    }
    r.count = @min(count, top);
    return r;
}

test "a logprob row: the normalizer, the best ids in order, ties to the lower id" {
    var l: [40]u16 = undefined;
    for (&l, 0..) |*w, i| w.* = @intCast(@as(u32, @bitCast(@as(f32, @floatFromInt(i % 7)))) >> 16);
    const r = row(&l, 6, 20);
    try std.testing.expectEqual(@as(u32, 6), r.ids[0]);
    try std.testing.expectEqual(@as(u32, 13), r.ids[1]);
    try std.testing.expect(r.logprobs[0] == r.logprob and r.logprobs[19] <= r.logprobs[18]);
}
