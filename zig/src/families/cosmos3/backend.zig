//! Cosmos3 streams behind the lane engine's backend interface: per-stream caches with their M-RoPE layout, the image
//! tower's rows placed over each image's placeholders in the prompt pass, keyed draws.
const std = @import("std");
const mtl = @import("metal");
const lanes = @import("lanes");
const c = @import("config.zig");
const st = @import("state.zig");
const fwd = @import("forward.zig");
const snapshot = @import("snapshot.zig");
const vision = @import("vision.zig");
const rope = @import("../../core/qwen_vl/rope.zig");
const Model = @import("model.zig").Model;
const be = lanes.backend;

pub const Options = struct { capacity: usize, chunk: usize = st.prompt_rows, streams: u32 = 4, image_tokens: usize = 0 };

pub const Metal = struct {
    residency: ?mtl.ResidencySet.Follow = null,
    gpa: std.mem.Allocator,
    model: *Model,
    options: Options,
    scratch: st.Scratch,
    tower: ?*vision.Tower = null,
    caches: std.AutoHashMapUnmanaged(*lanes.Stream, *st.Cache) = .empty,

    pub fn init(gpa: std.mem.Allocator, model: *Model, options: Options) !*Metal {
        const out = try gpa.create(Metal);
        errdefer gpa.destroy(out);
        out.* = .{ .gpa = gpa, .model = model, .options = options, .scratch = try st.Scratch.init(gpa, model.device, @max(options.chunk, st.batch_rows)) };
        errdefer out.scratch.deinit();
        if (options.image_tokens > 0) out.tower = try vision.Tower.init(gpa, model, options.image_tokens);
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
        if (self.tower) |t| t.deinit(self.gpa);
        self.scratch.deinit();
        self.gpa.destroy(self);
    }

    pub fn facts(self: *const Metal) lanes.Model {
        return factsFor(1, self.options.streams);
    }

    pub fn backend(self: *Metal) be.Backend {
        return .{ .ptr = self, .vtable = &.{ .prefill = prefillFn, .first = firstFn, .first_masked = firstMaskedFn, .queue = queueFn, .read = readFn, .verify = verifyFn, .keep = keepFn, .draft = draftFn, .release = releaseFn } };
    }

    fn cast(ptr: *anyopaque) *Metal {
        return @ptrCast(@alignCast(ptr));
    }

    fn cacheOf(self: *Metal, stream: *lanes.Stream) !*st.Cache {
        return self.caches.get(stream) orelse error.UnknownCosmosStream;
    }

    pub fn newCache(self: *Metal, capacity: usize) !*st.Cache {
        const cache = try self.gpa.create(st.Cache);
        errdefer self.gpa.destroy(cache);
        cache.* = try st.Cache.init(self.gpa, self.model.device, capacity);
        cache.memory.setResident(self.residency);
        return cache;
    }

    /// The tower's rows for every image of `stream` past prompt row `from`, and each prompt row's image row (-1: none).
    fn encodeImages(self: *Metal, stream: *const lanes.Stream, from: usize, rows: *?mtl.Buffer) !?[]i32 {
        if (stream.images.len == 0) return null;
        const tower = self.tower orelse return error.CosmosHasNoImageTower;
        const ids = stream.prompt();
        var total: usize = 0;
        for (stream.images) |im| total += if (im.at + im.tokens > from) im.tokens else 0;
        if (total == 0) return null;
        rows.* = try self.model.device.buffer(total * c.hidden * 2, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
        const map = try self.gpa.alloc(i32, ids.len);
        errdefer self.gpa.free(map);
        @memset(map, -1);
        var off: usize = 0;
        for (stream.images) |im| {
            if (im.at + im.tokens <= from) continue; // restored: its rows are in the kept state
            try tower.encode(self.model, im.pixels, im.gh, im.gw, .{ .buf = rows.*.?, .off = off * c.hidden * 2 });
            for (0..im.tokens) |k| map[im.at + k] = @intCast(off + k);
            off += im.tokens;
        }
        return map;
    }

    fn prefillFn(ptr: *anyopaque, stream: *lanes.Stream) !void {
        const self = cast(ptr);
        const ids = stream.prompt();
        if (ids.len == 0 or ids.len + stream.max_new > self.options.capacity or self.caches.contains(stream)) return error.InvalidCosmosPrompt;
        const capacity = @min(self.options.capacity, ids.len + stream.max_new + st.window_rows);
        const cache = try self.newCache(capacity);
        errdefer self.gpa.destroy(cache);
        errdefer cache.deinit();
        try self.caches.put(self.gpa, stream, cache);
        errdefer _ = self.caches.remove(stream);
        const cfg = &self.model.config;
        if (stream.images.len > 0) cache.layout = try rope.Layout.init(ids, stream.images, .{ .start = cfg.vision_start, .image = cfg.image_token, .end = cfg.vision_end }, c.merge);
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
        var rows: ?mtl.Buffer = null;
        defer if (rows) |r| r.deinit();
        const map = try self.encodeImages(stream, at, &rows);
        defer if (map) |mp| self.gpa.free(mp);
        var mark: usize = 0;
        while (at < ids.len) {
            var end: usize = @min(ids.len, at + self.options.chunk);
            for (stream.chunks) |boundary| if (boundary > at) {
                end = @min(end, boundary);
                break;
            };
            while (mark < stream.reuse.marks.len and stream.reuse.marks[mark] <= at) mark += 1;
            if (mark < stream.reuse.marks.len) end = @min(end, stream.reuse.marks[mark]);
            var images: ?fwd.Images = null;
            if (map) |mp| {
                const part = mp[at..end];
                const any = for (part) |x| {
                    if (x >= 0) break true;
                } else false;
                if (any) images = .{ .rows = rows.?, .map = part };
            }
            try fwd.run(self.model, &self.scratch, &.{.{ .cache = cache, .rows = end - at }}, ids[at..end], false, .last, images);
            at = end;
            if (stream.reuse.hook) |hk| if (std.mem.indexOfScalar(u32, stream.reuse.marks, @intCast(at)) != null)
                hk.at(hk.ptr, stream, @intCast(at));
        }
        dump(cache);
    }

    fn firstFn(ptr: *anyopaque, stream: *lanes.Stream, position: u64) !u64 {
        const self = cast(ptr);
        const cache = try self.cacheOf(stream);
        if (position != cache.len) return error.InvalidCosmosPosition;
        return try draw(self.gpa, cache.logits.slice(u16, c.vocab), stream.sampling, position, &.{});
    }

    fn firstMaskedFn(ptr: *anyopaque, stream: *lanes.Stream, position: u64, mask: []const u32) !u64 {
        const self = cast(ptr);
        const cache = try self.cacheOf(stream);
        if (position != cache.len) return error.InvalidCosmosPosition;
        return try draw(self.gpa, cache.logits.slice(u16, c.vocab), stream.sampling, position, mask);
    }

    fn queueFn(ptr: *anyopaque, stream: *lanes.Stream, feed: be.Feed, position: u64) !u64 {
        const self = cast(ptr);
        const cache = try self.cacheOf(stream);
        if (position != cache.len + 1) return error.InvalidCosmosPosition;
        const token: u32 = switch (feed) {
            .value => |v| v,
            .handle => |h| @intCast(h),
        };
        try fwd.run(self.model, &self.scratch, &.{.{ .cache = cache, .rows = 1 }}, &.{token}, false, .all, null);
        return try draw(self.gpa, cache.logits.slice(u16, c.vocab), stream.sampling, position, &.{});
    }

    fn readFn(_: *anyopaque, handle: u64) !u32 {
        if (handle >= c.vocab) return error.InvalidCosmosToken;
        return @intCast(handle);
    }

    fn verifyFn(ptr: *anyopaque, windows: []const be.Window, out: []be.Verified) !void {
        const self = cast(ptr);
        var segments: [st.batch_rows]fwd.Segment = undefined;
        var ids: [st.batch_rows]u32 = undefined;
        var total: usize = 0;
        if (windows.len > segments.len or windows.len != out.len) return error.InvalidCosmosWindows;
        for (windows, 0..) |w, i| {
            const rows = w.rows();
            if (w.parents != null or w.held != 0 or rows > st.window_rows or total + rows > st.batch_rows or w.positions.len != rows) return error.UnsupportedCosmosWindow;
            if (w.masks.len != 0 and w.masks.len != rows * mask_words) return error.InvalidCosmosMasks;
            const cache = try self.cacheOf(w.stream);
            for (w.positions, 0..) |p, r| if (p != cache.len + r + 1) return error.InvalidCosmosPosition;
            segments[i] = .{ .cache = cache, .rows = rows };
            ids[total] = w.pending;
            @memcpy(ids[total + 1 ..][0..w.tokens.len], w.tokens);
            total += rows;
        }
        try fwd.run(self.model, &self.scratch, segments[0..windows.len], ids[0..total], true, .all, null);
        const logits = self.scratch.logits.slice(u16, total * c.vocab);
        var base: usize = 0;
        for (windows, out) |w, o| {
            if (o.sampled.len != w.rows() or o.drafts.len != w.tokens.len) return error.InvalidCosmosOutputs;
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
        for (requests) |r| if (r.depth != 0) return error.CosmosCheckpointHasNoDraftHead;
    }

    fn releaseFn(ptr: *anyopaque, stream: *lanes.Stream) void {
        const self = cast(ptr);
        if (self.caches.fetchRemove(stream)) |entry| {
            entry.value.deinit();
            self.gpa.destroy(entry.value);
        }
    }
};

/// TF_COSMOS3_DUMP=path: each prompt's first logits (bf16, the vocabulary) written there, for the parity checks.
fn dump(cache: *const st.Cache) void {
    const path = std.c.getenv("TF_COSMOS3_DUMP") orelse return;
    const f = std.c.fopen(path, "wb") orelse return;
    defer _ = std.c.fclose(f);
    _ = std.c.fwrite(cache.logits.contents(), 1, c.vocab * 2, f);
}

/// The prompt cache's view of the streams: kept states copied at chunk ends, restored by the prompt pass.
pub const Snaps = struct {
    fn metal(ptr: *anyopaque) *Metal {
        return @ptrCast(@alignCast(ptr));
    }
    pub fn bytes(_: *anyopaque, at: u32) u64 {
        return snapshot.bytes(at);
    }
    pub fn save(ptr: *anyopaque, owner: ?*anyopaque, at: u32) anyerror!*anyopaque {
        const self = metal(ptr);
        const cache = try self.cacheOf(@ptrCast(@alignCast(owner orelse return error.NoStream)));
        return @ptrCast(try snapshot.save(self.gpa, cache, at));
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

pub fn value(word: u16) f64 {
    return @as(f32, @bitCast(@as(u32, word) << 16));
}

/// The unchanged top-k candidates for the shared fp64 keyed sampler; a `mask` keeps its allowed ones (-inf's draw).
pub fn draw(gpa: std.mem.Allocator, logits: []const u16, settings: ?lanes.Sampling, position: u64, mask: []const u32) !u32 {
    var best: ?usize = null;
    var allowed: usize = 0;
    for (logits, 0..) |word, id| {
        const x = value(word);
        if (!std.math.isFinite(x)) return error.NonfiniteCosmosLogits;
        if (mask.len > 0 and !lanes.grammar.allows(mask, id)) continue;
        allowed += 1;
        if (best == null or x > value(logits[best.?])) best = id;
    }
    if (best == null) return error.NoAllowedCosmosToken;
    const sampling = settings orelse return @intCast(best.?);
    if (sampling.temperature <= 0) return @intCast(best.?);
    const k = if (sampling.top_k == 0) allowed else @min(allowed, sampling.top_k);
    const values = try gpa.alloc(f64, k);
    defer gpa.free(values);
    const ids = try gpa.alloc(u64, k);
    defer gpa.free(ids);
    var filled: usize = 0;
    for (logits, 0..) |word, id| {
        if (mask.len > 0 and !lanes.grammar.allows(mask, id)) continue;
        const x = value(word);
        if (k == allowed) {
            values[filled] = x;
            ids[filled] = id;
            filled += 1;
            continue;
        }
        if (filled == k and x <= values[k - 1]) continue;
        var at = @min(filled, k - 1);
        while (at > 0 and x > values[at - 1]) : (at -= 1) {
            values[at] = values[at - 1];
            ids[at] = ids[at - 1];
        }
        values[at] = x;
        ids[at] = id;
        filled = @min(filled + 1, k);
    }
    return @intCast(try lanes.sampling.choose(gpa, values, ids, position, sampling));
}

/// One exact row a forward: streams take turns rather than sharing a window.
fn factsFor(width: u32, streams: u32) lanes.Model {
    return .{
        .exact_width = width,
        .hidden_rows = width > 1,
        .batch_rows = st.batch_rows,
        .max_streams = streams,
        .streams_exact = true,
        .window_costs = &.{},
        .plain_guard = true,
    };
}
