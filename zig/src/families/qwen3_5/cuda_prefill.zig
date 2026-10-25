//! Qwen CUDA prompt cuts and interrupted-prefill lifetime, shared by whole and stepped host calls.
const std = @import("std");
const lanes = @import("lanes");
const E = @import("cuda_engine.zig");
const vision_prompt = @import("cuda_vision_prompt.zig");
const reuse = @import("cuda_reuse.zig");

/// Where a prompt pass from `at` cuts its chunk: max_rows rows on, or the request's next chunk start before that.
fn nextCut(len: usize, chunks: []const u32, at: usize) usize {
    var end: usize = @min(len, at + E.max_rows);
    for (chunks) |boundary| if (boundary > at) {
        end = @min(end, boundary);
        break;
    };
    return end;
}

/// Whether a cold pass over `len` rows (from 0) cuts a chunk at `pos`.
fn coldCut(len: usize, chunks: []const u32, pos: usize) bool {
    var at: usize = 0;
    while (at < pos and at < len) at = nextCut(len, chunks, at);
    return at == pos;
}

test "a pass resumes only where a cold pass cuts: every max_rows rows from 0 or from the request's chunk starts" {
    const m = E.max_rows;
    const len = 10 * m;
    const start = m + m / 2; // a request chunk start between the first two cuts
    try std.testing.expect(coldCut(len, &.{}, m * 3));
    try std.testing.expect(!coldCut(len, &.{}, m * 3 + 1));
    try std.testing.expect(coldCut(len, &.{start}, start));
    try std.testing.expect(coldCut(len, &.{start}, start + m));
    try std.testing.expect(!coldCut(len, &.{start}, 2 * m)); // cut at the start instead, then max_rows on
    try std.testing.expectEqual(@as(usize, start), nextCut(len, &.{start}, m));
}

pub fn run(self: anytype, stream: *lanes.Stream) !void {
    var f = try opening(self, stream);
    defer f.deinit();
    while (!try chunk(self, stream, &f)) {}
}

/// One cut of the prompt pass a call (prefillFn's forwards, in order): the lane host runs rounds between them.
pub fn step(self: anytype, stream: *lanes.Stream) !bool {
    const got = try self.fills.getOrPut(self.gpa, stream);
    if (!got.found_existing) {
        got.value_ptr.* = opening(self, stream) catch |err| {
            self.fills.removeByPtr(got.key_ptr);
            return err;
        };
    }
    const f = self.fills.getPtr(stream).?;
    const done = chunk(self, stream, f) catch |err| {
        endFill(self, stream);
        return err;
    };
    if (done) endFill(self, stream);
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

/// A prompt pass under way: where its next cut starts, and an image prompt's encoded images.
pub const Fill = struct {
    at: usize,
    media: ?vision_prompt.Media,

    pub fn deinit(f: *Fill) void {
        if (f.media) |*m| m.deinit();
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
    const cache = try self.take(ids.len, stream.max_new);
    errdefer self.give(cache);
    // the stream's cache is found by a save at a mark (cuda_reuse.zig) while its pass runs
    try self.caches.put(self.gpa, stream, cache);
    errdefer _ = self.caches.remove(stream);
    // a kept state resumes only where a cold pass cuts a chunk, so the chunks after it keep a cold pass's bits
    var at: usize = 0;
    stream.cached = 0;
    if (stream.reuse.saved) |saved| {
        if (stream.reuse.at > 0 and stream.reuse.at < ids.len and coldCut(ids.len, stream.chunks, stream.reuse.at)) {
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
    return .{ .at = at, .media = media };
}

/// The pass's next cut; true once the prompt is in.
fn chunk(self: anytype, stream: *lanes.Stream, f: *Fill) !bool {
    const ids = stream.prompt();
    // cancel between cuts and retain the last complete prefix before releasing the cache (#1228)
    if (stream.isCancelled()) {
        stream.keepInterruptedPrefill(@intCast(f.at));
        return error.Cancelled;
    }
    const cache = try self.cacheOf(stream);
    var images: ?E.Images = if (f.media) |m| .{ .features = m.features.ptr, .src = m.src, .pos = m.pos } else null;
    const end = nextCut(ids.len, stream.chunks, f.at);
    try self.e.forward(&.{.{ .cache = cache, .rows = end - f.at, .images = if (images) |*im| im else null }}, ids[f.at..end], .last);
    f.at = end;
    stream.fill_at.store(@intCast(end), .monotonic);
    // a mark the store asked for, where this pass cut anyway: the lane host copies the state here
    if (end < ids.len and std.mem.indexOfScalar(u32, stream.reuse.marks, @intCast(end)) != null) {
        if (stream.reuse.hook) |k| k.at(k.ptr, stream, @intCast(end));
    }
    if (end < ids.len) return false;
    // the reply's rows rotate past the image prompt's positions (rope_delta)
    if (f.media) |m| cache.delta = m.delta;
    return true;
}
