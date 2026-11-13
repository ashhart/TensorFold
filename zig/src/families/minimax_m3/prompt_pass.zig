//! MiniMax-M3 Metal's prompt pass, whole or a chunk a call (#3021), moved out of backend.zig: main's cuts and taps.
const std = @import("std");
const mtl = @import("metal");
const lanes = @import("lanes");
const st = @import("state.zig");
const fwd = @import("forward.zig");
const snapshot = @import("snapshot.zig");
const steps = @import("prefill_steps.zig");
const Metal = @import("backend.zig").Metal;

/// A pass in progress: its row and late mark, and the image rows its chunks read, made once at its first call.
pub const Pass = struct {
    fill: steps.Fill,
    rows: ?mtl.Buffer = null,
    spans: std.ArrayList(fwd.Span) = .empty,

    pub fn deinit(p: *Pass, gpa: std.mem.Allocator) void {
        if (p.rows) |b| b.deinit();
        p.spans.deinit(gpa);
    }
};

/// The whole pass (prefill, and prefill_features with `self.sink` set): every chunk, marks kept as each lands.
pub fn whole(self: *Metal, stream: *lanes.Stream) !void {
    try self.drain();
    const p = try opening(self, stream);
    errdefer drop(self, stream);
    const r: Run = .{ .m = self, .cache = try self.cacheOf(stream), .spans = p.spans.items, .later = false };
    while (!try steps.step(r, stream, &p.fill, st.prompt_rows, false)) {}
    forget(self, stream);
}

/// One chunk a call at the whole pass's cuts (`max_rows` is never set: this family declares no chunk_invariant).
pub fn step(self: *Metal, stream: *lanes.Stream) !bool {
    try self.drain(); // other streams' steps in flight land before this chunk's forward
    // a drafting stream's wrapper absorbs each chunk after the call (featuresFn): its marks keep at the next call
    const later = stream.drafts and self.scratch.taps != null;
    const p = self.passes.getPtr(stream) orelse try opening(self, stream);
    errdefer drop(self, stream);
    const r: Run = .{ .m = self, .cache = try self.cacheOf(stream), .spans = p.spans.items, .later = later };
    const done = try steps.step(r, stream, &p.fill, st.prompt_rows, later);
    if (done) forget(self, stream);
    return done;
}

/// The host cancels between chunks: the last chunk's state is kept (its forward has waited; no prompt work runs).
pub fn cancel(self: *Metal, stream: *lanes.Stream) !void {
    try self.drain(); // the kept state is copied on the GPU, after any step in flight
    steps.cancel(stream, if (self.passes.get(stream)) |p| p.fill else null);
}

/// A stream that leaves: its pass, if any, goes with it.
pub fn forget(self: *Metal, stream: *lanes.Stream) void {
    if (self.passes.fetchRemove(stream)) |kv| {
        var p = kv.value;
        p.deinit(self.gpa);
    }
}

/// Whether a chunk's rows `tap`, written at forward `gen`, hold [start, start + count) at forward `now`.
pub fn within(tap: st.Tap, gen: u64, now: u64, start: u64, count: u32) bool {
    return tap.buf != null and steps.holds(.{ .start = tap.start, .rows = tap.rows, .gen = gen }, now, start, count);
}

/// A pass that failed: its record and its cache go, as the whole pass's error left them.
fn drop(self: *Metal, stream: *lanes.Stream) void {
    forget(self, stream);
    if (self.caches.fetchRemove(stream)) |entry| {
        self.held -= st.cacheBytes(&self.model.config, entry.value.capacity);
        entry.value.deinit();
        self.gpa.destroy(entry.value);
    }
}

/// One chunk's work for `steps.step`: the forward, the drafter's rows (a sink, or `later` via featuresFn), marks.
const Run = struct {
    m: *Metal,
    cache: *st.Cache,
    spans: []const fwd.Span,
    later: bool,

    pub fn run(r: Run, stream: *lanes.Stream, at: usize, end: usize) !void {
        const self = r.m;
        const cache = r.cache;
        const ids = stream.prompt();
        const s = if (end - at > st.batch_rows) &self.prompt else &self.scratch;
        const taps = s.taps;
        try fwd.run(self.model, s, cache, ids[at..end], .{ .taps = if (taps) |x| fwd.b(x) else null, .images = r.spans });
        self.gen +%= 1;
        if (taps) |x| {
            const width = self.tapBytes();
            if (self.sink) |k| try k.take(k.ptr, stream, .{ .buffer = @intFromPtr(x.id), .rows = @intCast(end - at), .row_bytes = @intCast(width), .space = .device, .dtype = .bf16 }, cache.len);
            if (r.later) cache.chunk_tap = .{ .tap = .{ .buf = x, .start = cache.len, .rows = end - at }, .gen = self.gen };
            if (end == ids.len) { // the prompt's last row, kept for the first draft
                if (cache.last_tap == null) cache.last_tap = try self.model.device.buffer(width, mtl.ResourceOptions.shared);
                @memcpy(cache.last_tap.?.slice(u8, width), x.slice(u8, (end - at) * width)[(end - at - 1) * width ..][0..width]);
                cache.tap = .{ .buf = cache.last_tap, .start = cache.len + (end - at) - 1, .rows = 1 };
            }
        }
        cache.len += end - at;
        lanes.prefill_widths.completed(self.gpa, stream, end - at);
    }

    pub fn keep(_: Run, stream: *lanes.Stream, at: u32) void {
        if (stream.reuse.hook) |k| k.at(k.ptr, stream, at);
    }
};

/// main's prefillFn up to its feed: the budget, the cache, a restored state and the images; the pass's record.
fn opening(self: *Metal, stream: *lanes.Stream) !*Pass {
    const ids = stream.prompt();
    if (ids.len == 0 or ids.len + stream.max_new > self.options.capacity or self.caches.contains(stream)) return error.InvalidMiniMaxPrompt;
    const capacity = ids.len + stream.max_new + 1;
    const need = st.cacheBytes(&self.model.config, capacity);
    if (self.held + need > self.options.budget) {
        std.log.warn("minimax_m3: a {d}-token stream needs {d:.1} GiB of caches; {d:.1} of {d:.1} GiB are held", .{ capacity, gib(need), gib(self.held), gib(self.options.budget) });
        return error.MiniMaxCachesFull;
    }
    self.held += need;
    errdefer self.held -= need;
    if (stream.reuse.saved) |saved| { // a state the stream cannot hold beside its caches: freed now, no restore
        const snap: *snapshot.Snap = @ptrCast(@alignCast(saved));
        if (self.held + snap.bytes > self.options.budget) snap.free();
    }
    // kept states give way first
    if (self.room) |r| r.fit(r.ctx, self.options.budget - self.held, stream.reuse.saved, ids);
    const cache = try self.createCache(capacity);
    errdefer self.gpa.destroy(cache);
    errdefer cache.deinit();
    try self.caches.put(self.gpa, stream, cache); // a mark's save finds it there
    errdefer _ = self.caches.remove(stream);
    stream.cached = 0;
    if (stream.reuse.saved) |saved| { // a kept state of this prompt's prefix: the pass starts there
        const snap: *const snapshot.Snap = @ptrCast(@alignCast(saved));
        if (snap.at < ids.len) {
            if (snapshot.restore(self.model.queue, &self.model.config, snap, cache)) stream.cached = snap.at else |_| stream.reuse_failed = true;
        } else stream.reuse_failed = true;
    }
    var p: Pass = .{ .fill = .{ .at = cache.len } };
    errdefer p.deinit(self.gpa);
    p.rows = try encodeImages(self, stream, cache.len, &p.spans);
    try self.passes.put(self.gpa, stream, p);
    stream.fill_at.store(@intCast(cache.len), .monotonic); // what the restore kept: 0 when none
    return self.passes.getPtr(stream).?;
}

fn gib(bytes: u64) f64 {
    return @as(f64, @floatFromInt(bytes)) / (1 << 30);
}

/// The tower's rows of the stream's images past `from` (one before it is in the restored state), a span each.
fn encodeImages(self: *Metal, stream: *const lanes.Stream, from: usize, spans: *std.ArrayList(fwd.Span)) !?mtl.Buffer {
    var total: usize = 0;
    for (stream.images) |im| {
        if (im.at + im.tokens > stream.prompt().len or im.tokens * 4 != im.gh * im.gw) return error.InvalidMiniMaxImage;
        if (im.at + im.tokens > from) total += im.tokens;
    }
    if (total == 0) return null;
    const v = self.model.vision orelse return error.NoMiniMaxVision;
    const width: usize = self.model.config.hidden;
    const buf = try self.model.device.buffer(total * width * 2, mtl.ResourceOptions.private | mtl.ResourceOptions.untracked);
    errdefer buf.deinit();
    var done: usize = 0;
    for (stream.images) |im| { // one command buffer an image: the tower's host-written tables are one image's
        if (im.at + im.tokens <= from) continue;
        const span: fwd.Span = .{ .at = im.at, .n = im.tokens, .rows = .{ .buf = buf, .off = done * width * 2 } };
        const cb = self.model.queue.commandBuffer();
        const e = cb.compute(.serial);
        try v.encode(e, im.pixels, im.gh, im.gw, span.rows);
        e.end();
        cb.commit();
        cb.wait();
        if (cb.failure()) |msg| {
            std.log.err("minimax_m3: the image tower failed: {s}", .{msg});
            return error.GpuFailed;
        }
        try spans.append(self.gpa, span);
        done += im.tokens;
    }
    return buf;
}
