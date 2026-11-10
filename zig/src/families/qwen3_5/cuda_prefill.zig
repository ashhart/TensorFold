//! Qwen CUDA prompt cuts and interrupted-prefill lifetime, shared by whole and stepped host calls.
const std = @import("std");
const lanes = @import("lanes");
const E = @import("cuda_engine.zig");
const vision_prompt = @import("cuda_vision_prompt.zig");
const reuse = @import("cuda_reuse.zig");

pub fn chunkRows(raw: ?[]const u8) !u32 {
    const text = raw orelse return E.max_rows;
    const n = std.fmt.parseInt(u32, text, 10) catch return error.InvalidPrefillChunk;
    if (n < 64 or n > E.max_rows or n % 64 != 0) return error.InvalidPrefillChunk;
    return n;
}

fn chunkLimit(self: anytype) u32 {
    return if (@hasField(@TypeOf(self.*), "prefill_rows")) self.prefill_rows else E.max_rows;
}

/// Where a prompt pass from `at` cuts its chunk: max_rows rows on, or the request's next chunk start before that.
fn nextCut(len: usize, chunks: []const u32, at: usize, max_rows: u32) usize {
    const bound = if (max_rows > 0) @min(E.max_rows, max_rows) else E.max_rows;
    var end: usize = @min(len, at + bound);
    for (chunks) |boundary| if (boundary > at) {
        end = @min(end, boundary);
        break;
    };
    return end;
}

/// Whether a cold pass over `len` rows (from 0) cuts a chunk at `pos`.
fn coldCut(len: usize, chunks: []const u32, pos: usize, limit: u32) bool {
    var at: usize = 0;
    while (at < pos and at < len) at = nextCut(len, chunks, at, limit);
    return at == pos;
}

test "a pass resumes only where a cold pass cuts: every max_rows rows from 0 or from the request's chunk starts" {
    const m = E.max_rows;
    const len = 10 * m;
    const start = m + m / 2; // a request chunk start between the first two cuts
    try std.testing.expect(coldCut(len, &.{}, m * 3, 0));
    try std.testing.expect(!coldCut(len, &.{}, m * 3 + 1, 0));
    try std.testing.expect(coldCut(len, &.{start}, start, 0));
    try std.testing.expect(coldCut(len, &.{start}, start + m, 0));
    try std.testing.expect(!coldCut(len, &.{start}, 2 * m, 0)); // cut at the start instead, then max_rows on
    try std.testing.expectEqual(@as(usize, start), nextCut(len, &.{start}, m, 0));
}

pub fn run(self: anytype, stream: *lanes.Stream) !void {
    var f = try opening(self, stream);
    defer f.deinit();
    while (!try chunk(self, stream, &f, 0)) {}
}

/// One cut of the prompt pass a call (prefillFn's forwards, in order): the lane host runs rounds between them.
pub fn step(self: anytype, stream: *lanes.Stream, _: u32) !bool {
    return stepWith(StepOps, self, stream);
}

const StepOps = struct {
    const open = opening;
    const advance = chunk;
    const finish = endFill;
};

/// The real fill lifecycle with compile-time operators for model-free sequencing checks.
fn stepWith(comptime Ops: type, self: anytype, stream: *lanes.Stream) !bool {
    const got = try self.fills.getOrPut(self.gpa, stream);
    if (!got.found_existing) {
        got.value_ptr.* = Ops.open(self, stream) catch |err| {
            self.fills.removeByPtr(got.key_ptr);
            return err;
        };
        const restored = if (@hasField(@TypeOf(self.*), "restored_features")) self.restored_features else false;
        // Absorb retained draft history before the next target forward, keeping the cold suffix's row groups.
        if (restored and stream.drafts and stream.cached > 0 and !stream.isCancelled()) return false;
    }
    const f = self.fills.getPtr(stream).?;
    const done = Ops.advance(self, stream, f, 0) catch |err| {
        Ops.finish(self, stream);
        return err;
    };
    if (done) Ops.finish(self, stream);
    return done;
}

/// The host cancels between iterations, while the last completed cut still owns its cache.
pub fn cancel(self: anytype, stream: *lanes.Stream) !void {
    if (self.fills.get(stream)) |f| stream.keepInterruptedPrefill(@intCast(f.at));
}

pub fn endFill(self: anytype, stream: *lanes.Stream) void {
    if (self.fills.fetchRemove(stream)) |kv| {
        var f = kv.value;
        f.deinit();
    }
}

/// A prompt pass under way: where its next cut starts, an image prompt's encoded images, TF_QWEN_KV_SHARE's marks.
pub const Fill = struct {
    at: usize,
    media: ?vision_prompt.Media,
    kept: ?struct { gpa: std.mem.Allocator, marks: []const u32 } = null,

    pub fn deinit(f: *Fill) void {
        if (f.media) |*m| m.deinit();
        if (f.kept) |k| k.gpa.free(k.marks);
    }
};

/// A prompt pass begins: its images encoded, a cache, a kept state restored where a cold pass cuts.
fn opening(self: anytype, stream: *lanes.Stream) !Fill {
    const ids = stream.prompt();
    if (ids.len == 0 or ids.len + stream.max_new > self.capacity or self.caches.contains(stream)) return error.InvalidQwenPrompt;
    // an image prompt: its images through the tower first, then every chunk at its t, h, w positions
    var media: ?vision_prompt.Media = null;
    if (stream.images.len > 0) {
        const v = self.vision orelse return error.QwenServedWithoutVision;
        media = try vision_prompt.Media.init(self.gpa, v, ids, stream.images);
    }
    errdefer if (media) |*m| m.deinit();
    const seen = try self.seenOf(stream.images);
    self.seen.put(self.gpa, stream, seen) catch |err| {
        self.gpa.free(seen);
        return err;
    };
    errdefer if (self.seen.fetchRemove(stream)) |kv| self.gpa.free(kv.value);
    // TF_QWEN_KV_SHARE: the kept pages this prompt resumes, adopted; images and request cuts keep their own grid
    const share = self.prefix != null and media == null and stream.chunks.len == 0;
    const shared = if (share) try self.prefix.?.open(ids, stream.max_new, stream.history_len, stream.shared_prefixes) else null;
    errdefer if (shared) |x| self.gpa.free(x.plan.marks);
    const cache = if (shared) |x| x.cache else try self.take(ids.len, stream.max_new);
    errdefer self.give(cache);
    // the stream's cache is found by a save at a mark (cuda_reuse.zig) while its pass runs
    try self.caches.put(self.gpa, stream, cache);
    errdefer _ = self.caches.remove(stream);
    // a kept state resumes only where a cold pass cuts a chunk, so the chunks after it keep a cold pass's bits
    var at: usize = 0;
    stream.cached = 0;
    if (shared) |x| {
        at = x.plan.from;
        stream.cached = @intCast(at);
        cache.len = at;
        cache.written = at;
    } else if (stream.reuse.saved) |saved| {
        if (stream.reuse.at > 0 and stream.reuse.at < ids.len and coldCut(ids.len, stream.chunks, stream.reuse.at, chunkLimit(self))) {
            if (reuse.restoreCache(self.e, cache, saved, seen)) {
                at = stream.reuse.at;
                stream.cached = stream.reuse.at;
            } else |_| {
                try cache.reset(self.e);
                stream.reuse_failed = true;
            }
        } else stream.reuse_failed = true;
    }
    cache.untaught = stream.untaught;
    stream.fill_at.store(@intCast(at), .monotonic);
    return .{ .at = at, .media = media, .kept = if (shared) |x| .{ .gpa = self.gpa, .marks = x.plan.marks } else null };
}

/// The pass's next cut; true once the prompt is in. The hint keeps the cold grid, where a cap would cut differently.
pub fn chunk(self: anytype, stream: *lanes.Stream, f: *Fill, _: u32) !bool {
    const ids = stream.prompt();
    // cancel between cuts and retain the last complete prefix before releasing the cache (#1228)
    if (stream.isCancelled()) {
        stream.keepInterruptedPrefill(@intCast(f.at));
        return error.Cancelled;
    }
    const cache = try self.cacheOf(stream);
    var images: ?E.Images = if (f.media) |m| .{ .features = m.features.ptr, .src = m.src, .pos = m.pos } else null;
    const end = nextCut(ids.len, stream.chunks, f.at, chunkLimit(self));
    try self.e.forward(&.{.{ .cache = cache, .rows = end - f.at, .images = if (images) |*im| im else null }}, ids[f.at..end], .last);
    lanes.prefill_widths.completed(self.gpa, stream, end - f.at);
    f.at = end;
    stream.fill_at.store(@intCast(end), .monotonic);
    // a mark the store asked for, where this pass cut anyway: the lane host copies the state here
    if (f.kept) |k| {
        if (end < ids.len and std.mem.indexOfScalar(u32, k.marks, @intCast(end)) != null)
            self.prefix.?.keep(cache, ids, end) catch |err| {
                if (err == error.QwenLaneTable) return err; // its table maps pages it may not hold
                std.log.warn("qwen3_5: prompt pages not kept: {s}", .{@errorName(err)});
            };
    } else if (end < ids.len and std.mem.indexOfScalar(u32, stream.reuse.marks, @intCast(end)) != null) {
        if (stream.reuse.hook) |k| k.at(k.ptr, stream, @intCast(end));
    }
    if (end < ids.len) return false;
    // the reply's rows rotate past the image prompt's positions (rope_delta)
    if (f.media) |m| cache.delta = m.delta;
    if (self.pages) |p| p.log(ids.len, stream.cached);
    return true;
}

test "explicit prefill chunks preserve a bounded cache-off grid" {
    try std.testing.expectEqual(@as(u32, @intCast(E.max_rows)), try chunkRows(null));
    try std.testing.expectEqual(@as(u32, 1024), try chunkRows("1024"));
    var limit_text: [24]u8 = undefined;
    try std.testing.expectEqual(@as(u32, @intCast(E.max_rows)), try chunkRows(try std.fmt.bufPrint(&limit_text, "{d}", .{E.max_rows})));
    try std.testing.expectError(error.InvalidPrefillChunk, chunkRows(try std.fmt.bufPrint(&limit_text, "{d}", .{E.max_rows + 64})));
    for ([_][]const u8{ "0", "63", "65", "4097", "bad" }) |v|
        try std.testing.expectError(error.InvalidPrefillChunk, chunkRows(v));
    try std.testing.expect(coldCut(7680, &.{}, 2048, 1024));
    try std.testing.expect(!coldCut(7680, &.{}, 2049, 1024));
}

const RestoreTest = struct {
    const be = lanes.backend;
    const dr = lanes.drafter;
    const Span = struct { start: u64, rows: u32 };
    const prompt = @as([8192]u32, @splat(3));
    gpa: std.mem.Allocator = std.testing.allocator,
    fills: std.AutoHashMapUnmanaged(*lanes.Stream, Fill) = .empty,
    restored_features: bool = true,
    restored: u32 = 7680,
    forwards: usize = 0,
    opens: usize = 0,
    head_releases: usize = 0,
    target_releases: usize = 0,
    pending_rows: u32 = 0,
    history_before_suffix: u32 = 0,
    absorbed: [4]Span = undefined,
    raw_rows: [4]u32 = undefined,
    n_absorbed: usize = 0,
    cancelled: bool = false,

    fn cast(ptr: *anyopaque) *RestoreTest {
        return @ptrCast(@alignCast(ptr));
    }
    fn deinit(self: *RestoreTest) void {
        var it = self.fills.valueIterator();
        while (it.next()) |f| f.deinit();
        self.fills.deinit(self.gpa);
    }
    const Ops = struct {
        fn open(self: *RestoreTest, s: *lanes.Stream) !Fill {
            s.cached = self.restored;
            s.fill_at.store(self.restored, .monotonic);
            return .{ .at = self.restored, .media = null };
        }
        fn advance(self: *RestoreTest, s: *lanes.Stream, f: *Fill, _: u32) !bool {
            if (s.isCancelled()) return error.Cancelled;
            if (f.at == 7680) self.history_before_suffix = self.pending_rows;
            self.pending_rows = 0; // the existing target forward's head readback drains its stream
            self.forwards += 1;
            f.at = @min(s.prompt().len, f.at + 7680);
            s.fill_at.store(@intCast(f.at), .monotonic);
            return f.at == s.prompt().len;
        }
        const finish = endFill;
    };
    fn targetStep(ptr: *anyopaque, s: *lanes.Stream, _: u32) !bool {
        return stepWith(Ops, cast(ptr), s);
    }
    fn prefill(ptr: *anyopaque, s: *lanes.Stream) !void {
        while (!try targetStep(ptr, s, 0)) {}
    }
    fn first(_: *anyopaque, _: *lanes.Stream, _: u64) !u64 {
        return 0;
    }
    fn queue(_: *anyopaque, _: *lanes.Stream, _: be.Feed, _: u64) !u64 {
        return 0;
    }
    fn read(_: *anyopaque, _: u64) !u32 {
        return 0;
    }
    fn verify(_: *anyopaque, _: []const be.Window, _: []be.Verified) !void {
        return error.UnexpectedVerify;
    }
    fn keep(_: *anyopaque, _: []const be.Window, _: []const []const u32) !void {}
    fn draft(_: *anyopaque, _: []const be.DraftRequest) !void {}
    fn features(_: *anyopaque, _: *lanes.Stream, _: []const u32, from: u64, rows: u32) !be.Features {
        return .{ .buffer = 1, .offset = from, .rows = rows, .row_bytes = 2, .space = .device, .dtype = .bf16 };
    }
    fn release(ptr: *anyopaque, s: *lanes.Stream) void {
        const self = cast(ptr);
        self.target_releases += 1;
        self.pending_rows = 0; // the existing paged lane release drains before returning its run
        endFill(self, s);
    }
    fn backend(self: *RestoreTest) be.Backend {
        return .{ .ptr = self, .restored_features = self.restored_features, .vtable = &.{
            .prefill = prefill,
            .prefill_step = targetStep,
            .reports_fill = true,
            .first = first,
            .queue = queue,
            .read = read,
            .verify = verify,
            .keep = keep,
            .draft = draft,
            .features = features,
            .release = release,
        } };
    }
    fn taps(_: *anyopaque) []const u32 {
        return &.{0};
    }
    fn facts(_: *anyopaque) dr.Facts {
        return .{ .history_rows = 2048, .depth = 7, .step_ms = 1 };
    }
    fn headOpen(ptr: *anyopaque, _: *lanes.Stream) !void {
        cast(ptr).opens += 1;
    }
    fn absorb(ptr: *anyopaque, items: []const dr.Absorb) !void {
        const self = cast(ptr);
        for (items) |item| {
            const n = @min(item.features.rows, 2048);
            self.raw_rows[self.n_absorbed] = item.features.rows;
            self.absorbed[self.n_absorbed] = .{ .start = item.features.offset + item.features.rows - n, .rows = n };
            self.n_absorbed += 1;
            self.pending_rows += n; // CUDA absorption enqueues kernels; it does not wait
        }
    }
    fn hold(_: *anyopaque, _: []const dr.Hold) !void {}
    fn held(_: *anyopaque, _: []const *lanes.Stream, _: []const []u32) !void {}
    fn headRelease(ptr: *anyopaque, _: *lanes.Stream) void {
        cast(ptr).head_releases += 1;
    }
    fn drafter(self: *RestoreTest) dr.Drafter {
        return .{ .ptr = self, .vtable = &.{ .taps = taps, .facts = facts, .open = headOpen, .absorb = absorb, .hold = hold, .held = held, .release = headRelease } };
    }
    fn isCancelled(ptr: *anyopaque) bool {
        return cast(ptr).cancelled;
    }
};

test "Qwen restored history: one boundary queues history before the cold suffix forward" {
    const t = std.testing;
    var state: RestoreTest = .{};
    defer state.deinit();
    var wrapped = try lanes.drafted.Drafted.init(t.allocator, state.backend(), state.drafter());
    defer wrapped.deinit();
    var s = try lanes.Stream.init(t.allocator, .{ .id = "restored", .prompt = &RestoreTest.prompt, .max_new = 96 });
    defer s.deinit(t.allocator);
    const b = wrapped.backend();
    defer b.release(&s);
    try t.expect(!try b.vtable.prefill_step.?(b.ptr, &s, 0));
    try t.expectEqual(@as(usize, 0), state.forwards);
    try t.expectEqual(@as(u32, 7680), s.fill_at.load(.monotonic));
    try t.expectEqual(@as(usize, 1), state.opens);
    try t.expectEqualDeep(RestoreTest.Span{ .start = 5632, .rows = 2048 }, state.absorbed[0]);
    try t.expect(try b.vtable.prefill_step.?(b.ptr, &s, 0));
    try t.expectEqual(@as(usize, 1), state.forwards);
    try t.expectEqual(@as(u32, 2048), state.history_before_suffix);
    try t.expectEqualDeep(RestoreTest.Span{ .start = 7680, .rows = 511 }, state.absorbed[1]);
    try t.expectEqual(@as(u32, 511), state.pending_rows);
    try t.expectEqual(@as(usize, 0), state.fills.count());
}

test "Qwen restored history: cold plain and unavailable history keep their existing step counts" {
    const t = std.testing;
    for ([_]bool{ false, true }) |restored_features| for ([_]bool{ false, true }) |drafts| for ([_]bool{ false, true }) |cached| {
        if (restored_features and drafts and cached) continue; // the history boundary is covered above
        var state: RestoreTest = .{ .restored = if (cached) 7680 else 0, .restored_features = restored_features };
        defer state.deinit();
        var wrapped = try lanes.drafted.Drafted.init(t.allocator, state.backend(), state.drafter());
        defer wrapped.deinit();
        var s = try lanes.Stream.init(t.allocator, .{ .id = "unchanged", .prompt = &RestoreTest.prompt, .max_new = 96, .drafts = drafts });
        defer s.deinit(t.allocator);
        const b = wrapped.backend();
        defer b.release(&s);
        var steps: usize = 1;
        while (!try b.vtable.prefill_step.?(b.ptr, &s, 0)) steps += 1;
        try t.expectEqual(@as(usize, if (cached) 1 else 2), steps);
        try t.expectEqual(steps, state.forwards);
        try t.expectEqual(@as(usize, @intFromBool(drafts)), state.opens);
    };
}

test "Qwen restored history: cancellation and release after the history boundary clean up once" {
    const t = std.testing;
    for ([_]bool{ false, true }) |want_cancel| {
        var state: RestoreTest = .{};
        defer state.deinit();
        var wrapped = try lanes.drafted.Drafted.init(t.allocator, state.backend(), state.drafter());
        defer wrapped.deinit();
        var s = try lanes.Stream.init(t.allocator, .{ .id = "cancel", .prompt = &RestoreTest.prompt, .max_new = 96, .cancel_check = .{ .ptr = &state, .check = RestoreTest.isCancelled } });
        defer s.deinit(t.allocator);
        const b = wrapped.backend();
        defer if (wrapped.lanes.contains(&s)) b.release(&s);
        try t.expect(!try b.vtable.prefill_step.?(b.ptr, &s, 0));
        if (want_cancel) {
            state.cancelled = true;
            try t.expectError(error.Cancelled, b.vtable.prefill_step.?(b.ptr, &s, 0));
        } else b.release(&s);
        try t.expectEqual(@as(usize, 0), state.forwards);
        try t.expectEqual(@as(usize, 0), state.fills.count());
        try t.expectEqual(@as(usize, 1), state.head_releases);
        try t.expectEqual(@as(usize, 1), state.target_releases);
        try t.expectEqual(@as(u32, 0), state.pending_rows);
    }
}
