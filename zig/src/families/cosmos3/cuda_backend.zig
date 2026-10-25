//! Cosmos3 streams behind the lane backend on CUDA, as backend.zig on Metal: per-stream M-RoPE caches, image rows.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const lanes = @import("lanes");
const c = @import("config.zig");
const md = @import("cuda_model.zig");
const rope = core.qwen_vl.rope;
const Model = md.Model;
const be = lanes.backend;

pub const Options = struct { capacity: usize, chunk: usize = md.prompt_rows, streams: u32 = 4, image_tokens: usize = 0 };

/// A stream's keys and values on the device, its layout, and its last row's logits (bf16 words, on the host).
pub const Cache = struct {
    buffer: cuda.DeviceBuffer,
    capacity: usize,
    len: usize = 0,
    layout: rope.Layout = .{},
    last: ?struct { start: usize, base: usize, rows: usize } = null,
    logits: []u16,

    fn kv(self: *const Cache) md.Kv {
        return .{ .base = self.buffer.ptr, .capacity = self.capacity };
    }

    pub fn position(self: *const Cache, p: usize) [3]i32 {
        return self.layout.position(@intCast(p));
    }

    fn commit(self: *Cache, base: usize, rows: usize, record: bool) void {
        if (record) self.last = .{ .start = self.len, .base = base, .rows = rows } else self.last = null;
        self.len += rows;
    }

    /// A verified window's accepted rows (a chain from its first): the rest are dropped, the last row's logits kept.
    fn keep(self: *Cache, m: *const Model, path: []const u32) !void {
        const last = self.last orelse return error.NothingToKeep;
        if (path.len == 0 or path.len > last.rows) return error.NothingToKeep;
        for (path, 0..) |p, i| if (p != i) return error.UnsupportedCosmosTree;
        self.len = last.start + path.len;
        @memcpy(self.logits, m.logitsRow(last.base + path.len - 1));
        self.last = null;
    }
};

pub const Cuda = struct {
    gpa: std.mem.Allocator,
    model: *Model,
    options: Options,
    caches: std.AutoHashMapUnmanaged(*lanes.Stream, *Cache) = .empty,
    image_rows: ?u64 = null, // the tower's text rows for one call's images (image_tokens rows)

    pub fn init(gpa: std.mem.Allocator, model: *Model, options: Options) !*Cuda {
        const out = try gpa.create(Cuda);
        errdefer gpa.destroy(out);
        out.* = .{ .gpa = gpa, .model = model, .options = options };
        if (options.image_tokens > 0) {
            try model.ctx.makeCurrent();
            out.image_rows = try model.alloc(2 * options.image_tokens * c.hidden);
        }
        return out;
    }

    pub fn deinit(self: *Cuda) void {
        self.model.ctx.makeCurrent() catch {};
        var it = self.caches.valueIterator();
        while (it.next()) |cache| self.destroy(cache.*);
        self.caches.deinit(self.gpa);
        self.gpa.destroy(self);
    }

    fn destroy(self: *Cuda, cache: *Cache) void {
        cache.buffer.free();
        self.gpa.free(cache.logits);
        self.gpa.destroy(cache);
    }

    pub fn facts(self: *const Cuda) lanes.Model {
        return .{
            .exact_width = 1,
            .hidden_rows = false,
            .batch_rows = md.batch_rows,
            .max_streams = self.options.streams,
            .streams_exact = true,
            .window_costs = &.{},
            .plain_guard = true,
        };
    }

    pub fn backend(self: *Cuda) be.Backend {
        return .{ .ptr = self, .vtable = &.{ .prefill = prefillFn, .first = firstFn, .first_masked = firstMaskedFn, .queue = queueFn, .read = readFn, .verify = verifyFn, .keep = keepFn, .draft = draftFn, .release = releaseFn } };
    }

    fn cast(ptr: *anyopaque) *Cuda {
        const self: *Cuda = @ptrCast(@alignCast(ptr));
        self.model.ctx.makeCurrent() catch {}; // the lane thread steps rounds; CUDA binds a context per thread
        return self;
    }

    fn cacheOf(self: *Cuda, stream: *lanes.Stream) !*Cache {
        return self.caches.get(stream) orelse error.UnknownCosmosStream;
    }

    fn newCache(self: *Cuda, capacity: usize) !*Cache {
        const cache = try self.gpa.create(Cache);
        errdefer self.gpa.destroy(cache);
        const logits = try self.gpa.alloc(u16, c.vocab);
        errdefer self.gpa.free(logits);
        cache.* = .{ .buffer = try cuda.DeviceBuffer.alloc(&self.model.driver, md.Kv.bytes(capacity)), .capacity = capacity, .logits = logits };
        return cache;
    }

    /// The tower's rows for every image of `stream` past prompt row `from`, and each prompt row's image row (-1: none).
    fn encodeImages(self: *Cuda, stream: *const lanes.Stream, from: usize) !?[]i32 {
        if (stream.images.len == 0) return null;
        const tower = if (self.model.tower) |*t| t else return error.CosmosHasNoImageTower;
        const ids = stream.prompt();
        var total: usize = 0;
        for (stream.images) |im| total += if (im.at + im.tokens > from) im.tokens else 0;
        if (total == 0) return null;
        if (total > self.options.image_tokens) return error.ImageTooLarge;
        const map = try self.gpa.alloc(i32, ids.len);
        errdefer self.gpa.free(map);
        @memset(map, -1);
        var off: usize = 0;
        for (stream.images) |im| {
            if (im.at + im.tokens <= from) continue; // restored: its rows are in the kept state
            try tower.encode(self.model, im.pixels, im.gh, im.gw, self.image_rows.? + 2 * off * c.hidden);
            for (0..im.tokens) |k| map[im.at + k] = @intCast(off + k);
            off += im.tokens;
        }
        return map;
    }

    /// One call over `segments` (rows from each cache's len on), its positions and slots from the caches' layouts.
    fn forward(self: *Cuda, caches: []const *Cache, rows: []const usize, ids: []const u32, head: Model.Head, map: ?[]const i32, record: bool) !void {
        const total = ids.len;
        var segments: [md.batch_rows]md.Segment = undefined;
        var pos: [md.prompt_rows][3]i32 = undefined;
        var slots: [md.prompt_rows]u32 = undefined;
        if (total > pos.len or caches.len > segments.len) return error.CosmosRowsExceeded;
        var r: usize = 0;
        for (caches, rows, 0..) |cache, n, i| {
            if (n == 0 or cache.len + n > cache.capacity) return error.CosmosContextFull;
            segments[i] = .{ .kv = cache.kv(), .first = cache.len, .rows = n };
            for (0..n) |j| {
                pos[r] = cache.position(cache.len + j);
                slots[r] = @intCast(cache.len + j);
                r += 1;
            }
        }
        if (r != total) return error.InvalidCosmosSegments;
        for (ids) |id| if (id >= c.vocab) return error.InvalidCosmosToken;
        const images: ?Model.Images = if (map) |mp| .{ .rows = self.image_rows.?, .map = mp } else null;
        try self.model.run(segments[0..caches.len], ids, pos[0..total], slots[0..total], head, images);
        var base: usize = 0;
        for (caches, rows) |cache, n| {
            @memcpy(cache.logits, self.model.logitsRow(if (head == .all) base + n - 1 else 0));
            cache.commit(base, n, record);
            base += n;
        }
    }

    fn prefillFn(ptr: *anyopaque, stream: *lanes.Stream) !void {
        const self = cast(ptr);
        const ids = stream.prompt();
        if (ids.len == 0 or ids.len + stream.max_new > self.options.capacity or self.caches.contains(stream)) return error.InvalidCosmosPrompt;
        const capacity = @min(self.options.capacity, ids.len + stream.max_new + md.window_rows);
        const cache = try self.newCache(capacity);
        errdefer self.destroy(cache);
        try self.caches.put(self.gpa, stream, cache);
        errdefer _ = self.caches.remove(stream);
        const cfg = &self.model.config;
        if (stream.images.len > 0) cache.layout = try rope.Layout.init(ids, stream.images, .{ .start = cfg.vision_start, .image = cfg.image_token, .end = cfg.vision_end }, c.merge);
        stream.cached = 0;
        const t0 = now();
        const timing = std.c.getenv("TF_COSMOS3_TIMING") != null;
        const map = try self.encodeImages(stream, 0);
        defer if (map) |mp| self.gpa.free(mp);
        if (timing) try self.model.stream.synchronize(); // the tower's own time, for the log line
        const t1 = now();
        var at: usize = 0;
        while (at < ids.len) {
            var end: usize = @min(ids.len, at + self.options.chunk);
            for (stream.chunks) |boundary| if (boundary > at) {
                end = @min(end, boundary);
                break;
            };
            var part: ?[]const i32 = null;
            if (map) |mp| {
                const p = mp[at..end];
                if (for (p) |x| {
                    if (x >= 0) break true;
                } else false) part = p;
            }
            try self.forward(&.{cache}, &.{end - at}, ids[at..end], .last, part, false);
            at = end;
        }
        if (timing)
            std.log.info("cosmos3 prefill: {d} rows, image tower {d:.1} ms, text {d:.1} ms", .{ ids.len, (t1 - t0) * 1e3, (now() - t1) * 1e3 });
        dump(cache);
    }

    fn firstFn(ptr: *anyopaque, stream: *lanes.Stream, position: u64) !u64 {
        const self = cast(ptr);
        const cache = try self.cacheOf(stream);
        if (position != cache.len) return error.InvalidCosmosPosition;
        return try draw(self.gpa, cache.logits, stream.sampling, position, &.{});
    }

    fn firstMaskedFn(ptr: *anyopaque, stream: *lanes.Stream, position: u64, mask: []const u32) !u64 {
        const self = cast(ptr);
        const cache = try self.cacheOf(stream);
        if (position != cache.len) return error.InvalidCosmosPosition;
        return try draw(self.gpa, cache.logits, stream.sampling, position, mask);
    }

    fn queueFn(ptr: *anyopaque, stream: *lanes.Stream, feed: be.Feed, position: u64) !u64 {
        const self = cast(ptr);
        const cache = try self.cacheOf(stream);
        if (position != cache.len + 1) return error.InvalidCosmosPosition;
        const token: u32 = switch (feed) {
            .value => |v| v,
            .handle => |h| @intCast(h),
        };
        try self.forward(&.{cache}, &.{1}, &.{token}, .last, null, false);
        return try draw(self.gpa, cache.logits, stream.sampling, position, &.{});
    }

    fn readFn(_: *anyopaque, handle: u64) !u32 {
        if (handle >= c.vocab) return error.InvalidCosmosToken;
        return @intCast(handle);
    }

    fn verifyFn(ptr: *anyopaque, windows: []const be.Window, out: []be.Verified) !void {
        const self = cast(ptr);
        var caches: [md.batch_rows]*Cache = undefined;
        var rows: [md.batch_rows]usize = undefined;
        var ids: [md.batch_rows]u32 = undefined;
        var total: usize = 0;
        if (windows.len > caches.len or windows.len != out.len) return error.InvalidCosmosWindows;
        for (windows, 0..) |w, i| {
            const n = w.rows();
            if (w.parents != null or w.held != 0 or n > md.window_rows or total + n > md.batch_rows or w.positions.len != n) return error.UnsupportedCosmosWindow;
            if (w.masks.len != 0 and w.masks.len != n * mask_words) return error.InvalidCosmosMasks;
            const cache = try self.cacheOf(w.stream);
            for (w.positions, 0..) |p, r| if (p != cache.len + r + 1) return error.InvalidCosmosPosition;
            caches[i] = cache;
            rows[i] = n;
            ids[total] = w.pending;
            @memcpy(ids[total + 1 ..][0..w.tokens.len], w.tokens);
            total += n;
        }
        try self.forward(caches[0..windows.len], rows[0..windows.len], ids[0..total], .all, null, true);
        var base: usize = 0;
        for (windows, out) |w, o| {
            if (o.sampled.len != w.rows() or o.drafts.len != w.tokens.len) return error.InvalidCosmosOutputs;
            for (o.sampled, w.positions, 0..) |*t, p, r| {
                const mask = if (w.masks.len > 0) w.masks[r * mask_words ..][0..mask_words] else &.{};
                t.* = try draw(self.gpa, self.model.logitsRow(base + r), w.stream.sampling, p, mask);
            }
            @memcpy(o.drafts, w.tokens);
            base += w.rows();
        }
    }

    fn keepFn(ptr: *anyopaque, windows: []const be.Window, paths: []const []const u32) !void {
        const self = cast(ptr);
        for (windows, paths) |w, path| try (try self.cacheOf(w.stream)).keep(self.model, path);
    }

    fn draftFn(_: *anyopaque, requests: []const be.DraftRequest) !void {
        for (requests) |r| if (r.depth != 0) return error.CosmosCheckpointHasNoDraftHead;
    }

    fn releaseFn(ptr: *anyopaque, stream: *lanes.Stream) void {
        const self = cast(ptr);
        if (self.caches.fetchRemove(stream)) |entry| self.destroy(entry.value);
    }
};

/// Monotonic seconds.
fn now() f64 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.MONOTONIC, &ts);
    return @as(f64, @floatFromInt(ts.sec)) + @as(f64, @floatFromInt(ts.nsec)) * 1e-9;
}

/// TF_COSMOS3_DUMP=path: each prompt's first logits (bf16, the vocabulary) written there, for the parity checks.
fn dump(cache: *const Cache) void {
    const path = std.c.getenv("TF_COSMOS3_DUMP") orelse return;
    const f = std.c.fopen(path, "wb") orelse return;
    defer _ = std.c.fclose(f);
    _ = std.c.fwrite(std.mem.sliceAsBytes(cache.logits).ptr, 1, c.vocab * 2, f);
}

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
