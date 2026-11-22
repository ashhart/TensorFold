//! Qwen CUDA prompt cuts and interrupted-prefill lifetime, shared by whole and stepped host calls.
const std = @import("std");
const lanes = @import("lanes");
const E = @import("cuda_engine.zig");
const prompt_tail = @import("cuda_prompt_tail.zig");
const tap_window = @import("cuda_cache.zig").tap_window;
const vision_prompt = @import("cuda_vision_prompt.zig");
const readout = @import("cuda_readout.zig");
const cuda_solo = @import("cuda_solo.zig");

/// Where a prompt pass from `at` cuts its chunk: grid_rows rows on, or the request's next chunk start before that.
fn nextCut(len: usize, chunks: []const u32, at: usize, max_rows: u32) usize {
    const bound = if (max_rows > 0) @min(E.grid_rows, max_rows) else E.grid_rows;
    var end: usize = @min(len, at + bound);
    for (chunks) |boundary| if (boundary > at) {
        end = @min(end, boundary);
        break;
    };
    return end;
}

/// Whether a cold pass over `len` rows (from 0) cuts a chunk at `pos`.
fn coldCut(len: usize, chunks: []const u32, pos: usize) bool {
    var at: usize = 0;
    while (at < pos and at < len) at = nextCut(len, chunks, at, 0);
    return at == pos;
}

test "a pass resumes only where a cold pass cuts: every grid_rows rows from 0 or from the request's chunk starts" {
    const m = E.grid_rows;
    const len = 10 * m;
    const start = m + m / 2; // a request chunk start between the first two cuts
    try std.testing.expect(coldCut(len, &.{}, m * 3));
    try std.testing.expect(!coldCut(len, &.{}, m * 3 + 1));
    try std.testing.expect(coldCut(len, &.{start}, start));
    try std.testing.expect(coldCut(len, &.{start}, start + m));
    try std.testing.expect(!coldCut(len, &.{start}, 2 * m)); // cut at the start instead, then grid_rows on
    try std.testing.expectEqual(@as(usize, start), nextCut(len, &.{start}, m, 0));
}

/// The rows a capped step's cuts keep: the prompt tile's rows, a multiple of the delta rule's 64-row chunks.
pub const cap_grid = 128;

/// MTP absorbs this many ordinary prompt rows; the final prompt row is absorbed separately.
pub fn mtpPromptTapFrom(prompt_len: usize, held_rows: usize) ?usize {
    const needed = held_rows + 1;
    return if (prompt_len > needed) prompt_len - needed else null;
}

fn initialMtpPromptRows(direct_mtp_norm: bool, configured_rows: usize) usize {
    return if (direct_mtp_norm) configured_rows else tap_window;
}

fn stepEndAtMtpSuffix(at: usize, end: usize, from: ?usize) usize {
    const boundary = from orelse return end;
    return if (at < boundary and boundary < end) boundary else end;
}

fn targetForwardEnd(at: usize, end: usize, from: ?usize, stepped: bool, direct_mtp_norm: bool) usize {
    return if (stepped and !direct_mtp_norm) stepEndAtMtpSuffix(at, end, from) else end;
}

pub fn onlySkippedMtpPrefix(start: u64, count: u32, from: ?usize) bool {
    const boundary = from orelse return false;
    const limit: u64 = @intCast(boundary);
    return start < limit and count <= limit - start;
}

test "MTP prompt tap boundary includes 2048 ordinary rows and the separately absorbed final row" {
    try std.testing.expectEqual(@as(?usize, null), mtpPromptTapFrom(2048, 2048));
    try std.testing.expectEqual(@as(?usize, null), mtpPromptTapFrom(2049, 2048));
    const from4k = mtpPromptTapFrom(4096, 2048).?;
    try std.testing.expectEqual(@as(usize, 2047), from4k);
    try std.testing.expectEqual(@as(usize, 2048), (4096 - 1) - from4k);
    try std.testing.expectEqual(@as(usize, 1), 4096 - (4096 - 1));
    const from8k = mtpPromptTapFrom(8192, 2048).?;
    try std.testing.expectEqual(@as(usize, 6143), from8k);
    try std.testing.expectEqual(@as(usize, 2048), (8192 - 1) - from8k);
    try std.testing.expectEqual(@as(usize, 1), 8192 - (8192 - 1));
    try std.testing.expectEqual(@as(?usize, 4095), mtpPromptTapFrom(6144, 2048));
    try std.testing.expectEqual(@as(?usize, 4096), mtpPromptTapFrom(6145, 2048));
}

test "qualified 256-row MTP history applies only to direct-normalized input" {
    const rows: usize = @intCast(@import("cuda_mtp_types.zig").default_initial_prompt_rows);
    try std.testing.expectEqual(rows, initialMtpPromptRows(true, rows));
    const needed = mtpPromptTapFrom(8192, rows).?;
    try std.testing.expectEqual(@as(usize, 8192 - rows - 1), needed);
    try std.testing.expectEqual(tap_window, initialMtpPromptRows(false, rows));
    try std.testing.expectEqual(@as(?usize, 7935), mtpPromptTapFrom(8192, rows));
}

test "stepped MTP prefill cuts at the suffix and identifies discarded prefix rows" {
    try std.testing.expectEqual(@as(usize, 6143), stepEndAtMtpSuffix(4096, 8192, 6143));
    try std.testing.expectEqual(@as(usize, 6143), stepEndAtMtpSuffix(4096, 6143, 6143));
    try std.testing.expectEqual(@as(usize, 8192), stepEndAtMtpSuffix(6143, 8192, 6143));
    try std.testing.expectEqual(@as(usize, 8192), stepEndAtMtpSuffix(4096, 8192, null));
    try std.testing.expect(onlySkippedMtpPrefix(4096, 2047, 6143));
    try std.testing.expect(!onlySkippedMtpPrefix(4096, 2048, 6143));
    try std.testing.expect(!onlySkippedMtpPrefix(6143, 1024, 6143));
}

test "direct-normalized MTP keeps the target's original stepped chunk boundary" {
    try std.testing.expectEqual(@as(usize, 6143), targetForwardEnd(4096, 8192, 6143, true, false));
    try std.testing.expectEqual(@as(usize, 8192), targetForwardEnd(4096, 8192, 6143, true, true));
    try std.testing.expectEqual(@as(usize, 8192), targetForwardEnd(4096, 8192, null, true, true));
    try std.testing.expectEqual(@as(usize, 8192), targetForwardEnd(4096, 8192, 6143, false, false));
}

/// The end of `f`'s next step under `max_rows` (0: the cold cut), its cold cut renewed once the last one landed.
fn capped(len: usize, chunks: []const u32, f: *Fill, max_rows: u32) usize {
    if (f.cold <= f.at) f.cold = nextCut(len, chunks, f.at, 0);
    return lanes.step_cap.end(f.at, f.cold, max_rows, cap_grid);
}

test "a yielded pass keeps its completed prefix and stops before the next chunk, on the grid chunk though solo" {
    const t = std.testing;
    const len: usize = 2 * @as(usize, E.grid_rows) + 1; // a solo pass would take both chunks at once
    const ids = try t.allocator.alloc(u32, len);
    defer t.allocator.free(ids);
    @memset(ids, 3);
    const Probe = struct {
        stream: ?*lanes.Stream = null,
        saves: usize = 0,
        saved_at: u32 = 0,
        fn cast(ctx: *anyopaque) *@This() {
            return @ptrCast(@alignCast(ctx));
        }
        fn yielded(ctx: *anyopaque) bool {
            const p = cast(ctx);
            return p.stream.?.fill_at.load(.monotonic) >= E.grid_rows;
        }
        fn kept(ctx: *anyopaque, _: *lanes.Stream, at: u32) void {
            const p = cast(ctx);
            p.saves += 1;
            p.saved_at = at;
        }
    };
    const FakeEngine = struct {
        scratch_rows: usize = 2 * E.grid_rows, // TF_QWEN35_CHUNK_SOLO=8192
        forwards: usize = 0,
        layers: [2]void = .{ {}, {} },
        taps: [1]u32 = undefined,
        n_taps: usize = 0,
        learning: ?*const anyopaque = null,
        mlp: ?*const anyopaque = null,
        pockets: struct {
            pub fn any(_: *const @This()) bool {
                return false;
            }
        } = .{},
        int8: struct { on: bool = false } = .{},
        fn forward(self: *@This(), parts: []const E.Part, ids_: []const u32, _: E.Keep) !void {
            self.forwards += 1;
            parts[0].cache.len += ids_.len;
            parts[0].cache.written += ids_.len;
        }
    };
    const FakePrefix = struct {
        fn keep(_: *@This(), _: *E.Cache, _: []const u32, _: usize) !void {}
    };
    const FakePages = struct {
        fn log(_: *@This(), _: usize, _: usize) void {}
    };
    const FakeLanes = struct {
        gpa: std.mem.Allocator,
        e: FakeEngine = .{},
        cache: *E.Cache,
        prefix: ?*FakePrefix = null,
        pages: ?*FakePages = null,
        caches: std.AutoHashMapUnmanaged(*lanes.Stream, *E.Cache) = .empty, // no stream beside it decodes
        fills: std.AutoHashMapUnmanaged(*lanes.Stream, void) = .empty,
        fn cacheOf(self: *@This(), _: *lanes.Stream) !*E.Cache {
            return self.cache;
        }
    };

    var probe: Probe = .{};
    var cache: E.Cache = undefined;
    cache.len = 0;
    cache.written = 0;
    var fake: FakeLanes = .{ .gpa = t.allocator, .cache = &cache };
    var stream = try lanes.Stream.init(t.allocator, .{
        .id = "yielded-prefix",
        .prompt = ids,
        .max_new = 1,
        .yield_check = .{ .ptr = &probe, .check = Probe.yielded },
        .reuse = .{
            .marks = &.{},
            .hook = .{ .ptr = &probe, .at = Probe.kept },
        },
    });
    defer stream.deinit(t.allocator);
    probe.stream = &stream;
    var fill: Fill = .{ .at = 0, .media = null };

    try t.expect(!try chunk(&fake, &stream, &fill, 0));
    try t.expectEqual(@as(usize, 1), fake.e.forwards);
    try t.expectEqual(E.grid_rows, fill.at); // one grid chunk: the yield check keeps the pass off the solo rows
    try t.expectEqual(@as(usize, 0), probe.saves);
    try t.expectError(error.Yielded, chunk(&fake, &stream, &fill, 0));
    try t.expectEqual(@as(usize, 1), fake.e.forwards);
    try t.expectEqual(@as(usize, 1), probe.saves);
    try t.expectEqual(E.grid_rows, probe.saved_at);
    // the same prompt in the foreground (no yield check): both grid chunks in one solo pass
    var fore = try lanes.Stream.init(t.allocator, .{ .id = "foreground", .prompt = ids, .max_new = 1 });
    defer fore.deinit(t.allocator);
    var whole_fill: Fill = .{ .at = 0, .media = null };
    try t.expect(!try chunk(&fake, &fore, &whole_fill, 0));
    try t.expectEqual(2 * E.grid_rows, whole_fill.at);
}

/// The fewest rows a cold chunk needs to join a solo pass: fewer take the projections' small-row tiles.
pub const solo_min_rows = 2048;

/// A solo pass's end (TF_QWEN35_CHUNK_SOLO): cold chunks of solo_min_rows or more, up to `rows`, a kept mark ends it.
fn soloEnd(len: usize, chunks: []const u32, f: *Fill, rows: usize, marks: []const u32) usize {
    var end = capped(len, chunks, f, 0);
    if (end - f.at < solo_min_rows) return end; // a short chunk runs alone, as in a cold pass
    while (end < len and std.mem.indexOfScalar(u32, marks, @intCast(end)) == null) {
        const next = nextCut(len, chunks, end, 0);
        if (next - end < solo_min_rows or next - f.at > rows) break;
        end = next;
    }
    f.cold = end; // the merged cuts' last: the pass lands on it, as on any cold cut
    return end;
}

test "a solo pass merges cold chunks of 2048 rows or more up to its rows; marks and short chunks keep their cuts" {
    const t = std.testing;
    const m = E.grid_rows;
    var f: Fill = .{ .at = 0, .media = null };
    try t.expectEqual(2 * m, soloEnd(10 * m, &.{}, &f, 2 * m, &.{}));
    try t.expectEqual(2 * m, f.cold);
    f.at = 2 * m;
    try t.expectEqual(3 * m, soloEnd(10 * m, &.{}, &f, 2 * m, &.{3 * m})); // the mark keeps its cut
    f = .{ .at = 0, .media = null };
    try t.expectEqual(m, soloEnd(m + 12, &.{}, &f, 2 * m, &.{})); // a 12-row tail runs alone
    f = .{ .at = 0, .media = null };
    try t.expectEqual(m + 2048, soloEnd(m + 2048, &.{}, &f, 2 * m, &.{})); // a 2048-row tail joins
    f = .{ .at = 0, .media = null };
    try t.expectEqual(m, soloEnd(10 * m, &.{m + 1000}, &f, 2 * m, &.{})); // a request start's 1000-row chunk alone
    f = .{ .at = m, .media = null };
    try t.expectEqual(m + 1000, soloEnd(10 * m, &.{m + 1000}, &f, 2 * m, &.{}));
    f = .{ .at = 512, .cold = m, .media = null }; // a capped step's 3584-row rest, then the next chunk
    try t.expectEqual(2 * m, soloEnd(10 * m, &.{}, &f, 2 * m, &.{}));
    f = .{ .at = 0, .media = null };
    try t.expectEqual(2 * m, soloEnd(10 * m, &.{}, &f, 3 * m - 1, &.{})); // whole chunks only, never past the rows
}

pub fn run(self: anytype, stream: *lanes.Stream) !void {
    var f = try opening(self, stream, false);
    defer f.deinit();
    while (!try chunk(self, stream, &f, 0)) {}
}

/// One whole prompt fill: a readout's pieces to the end, else the text pass, each exactly once.
pub fn whole(self: anytype, stream: *lanes.Stream) !void {
    if (stream.readout != null) {
        while (!try readout.step(self, stream, 0)) {}
        return;
    }
    return run(self, stream);
}

/// One cut of the prompt pass a call (prefillFn's forwards, in order): the lane host runs rounds between them.
pub fn step(self: anytype, stream: *lanes.Stream, max_rows: u32) !bool {
    const got = try self.fills.getOrPut(self.gpa, stream);
    if (!got.found_existing) {
        got.value_ptr.* = opening(self, stream, true) catch |err| {
            self.fills.removeByPtr(got.key_ptr);
            return err;
        };
    }
    const f = self.fills.getPtr(stream).?;
    const done = chunk(self, stream, f, max_rows) catch |err| {
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

/// A prompt pass under way: where its next cut starts, an image prompt's encoded images, TF_QWEN_KV_SHARE's marks.
pub const Fill = struct {
    at: usize,
    stepped: bool = false,
    cold: usize = 0, // the cold pass's cut that ends the chunk in flight; a capped step cuts inside it (lanes.step_cap)
    media: ?vision_prompt.Media,
    mtp_tap_from: ?usize = null,
    mtp_prompt_taps: bool = false,
    direct_mtp_norm: bool = false,
    kept: ?struct { gpa: std.mem.Allocator, marks: []const u32 } = null,

    pub fn deinit(f: *Fill) void {
        if (f.media) |*m| m.deinit();
        if (f.kept) |k| k.gpa.free(k.marks);
    }
};

/// A prompt pass begins: its images encoded, a cache, a kept state restored where a cold pass cuts.
fn opening(self: anytype, stream: *lanes.Stream, stepped: bool) !Fill {
    const ids = stream.prompt();
    if (ids.len == 0 or ids.len + stream.max_new > self.capacity or self.caches.contains(stream)) return error.InvalidQwenPrompt;
    // an image prompt: its images through the tower once the resume is known, then every chunk at its t, h, w positions
    if (stream.images.len > 0 and self.vision == null) return error.QwenServedWithoutVision;
    const seen = try self.seenOf(stream.images);
    self.seen.put(self.gpa, stream, seen) catch |err| {
        self.gpa.free(seen);
        return err;
    };
    errdefer if (self.seen.fetchRemove(stream)) |kv| self.gpa.free(kv.value);
    // TF_QWEN_KV_SHARE: the kept pages this prompt resumes, adopted; images and request cuts keep their own grid
    const share = self.prefix != null and stream.images.len == 0 and stream.chunks.len == 0;
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
        if (stream.reuse.at > 0 and stream.reuse.at < ids.len and coldCut(ids.len, stream.chunks, stream.reuse.at)) {
            if (self.restoreKept(cache, saved, seen)) {
                at = stream.reuse.at;
                stream.cached = stream.reuse.at;
            } else |_| {
                try cache.reset(self.e);
                stream.reuse_failed = true;
            }
        } else stream.reuse_failed = true;
    }
    cache.untaught = stream.untaught;
    // the images a kept state holds skip the tower (their rows are never spliced); positions still come from all
    const media: ?vision_prompt.Media = if (stream.images.len > 0) try vision_prompt.Media.init(self.gpa, self.vision.?, ids, stream.images, at) else null;
    stream.fill_at.store(@intCast(at), .monotonic);
    const mtp_prompt_taps = self.mtp != null;
    // MTP reuses normalized rows only before the next target forward overwrites the shared buffer.
    const direct_mtp_norm = if (comptime @TypeOf(self.e) == *E.Engine)
        stepped and mtp_prompt_taps and stream.drafts and !self.e.g.bf16 and self.e.learning == null and
            self.e.mlp == null and !self.e.pockets.any() and !self.e.int8.on and at == 0 and media == null and
            stream.readout == null and stream.chunks.len == 0 and stream.reuse.saved == null and stream.reuse.marks.len == 0 and
            stream.reuse.hook == null and self.prefix == null
    else
        false;
    const optimize_mtp_prefix = mtp_prompt_taps and at == 0 and media == null and
        stream.chunks.len == 0 and stream.reuse.saved == null and stream.reuse.marks.len == 0 and stream.reuse.hook == null and self.prefix == null;
    const configured_mtp_rows = if (comptime @TypeOf(self.e) == *E.Engine) blk: {
        const configured = if (self.mtp) |mtp| mtp.initial_prompt_rows else tap_window;
        break :blk @as(usize, configured);
    } else tap_window;
    const initial_mtp_rows = initialMtpPromptRows(direct_mtp_norm, configured_mtp_rows);
    const mtp_tap_from = if (optimize_mtp_prefix)
        (if (stream.drafts) mtpPromptTapFrom(ids.len, initial_mtp_rows) else ids.len)
    else
        null;
    return .{ .at = at, .stepped = stepped, .media = media, .mtp_tap_from = mtp_tap_from, .mtp_prompt_taps = mtp_prompt_taps, .direct_mtp_norm = direct_mtp_norm, .kept = if (shared) |x| .{ .gpa = self.gpa, .marks = x.plan.marks } else null };
}

/// The pass's next cut; true once the prompt is in. A cap cuts inside the cold chunk (on cap_grid): cold cuts all land.
pub fn chunk(self: anytype, stream: *lanes.Stream, f: *Fill, max_rows: u32) !bool {
    const ids = stream.prompt();
    const EngineType = if (@typeInfo(@TypeOf(self.e)) == .pointer)
        @typeInfo(@TypeOf(self.e)).pointer.child
    else
        @TypeOf(self.e);
    // cancel between cuts and retain the last complete prefix before releasing the cache (#1228)
    if (stream.isCancelled()) {
        stream.keepInterruptedPrefill(@intCast(f.at));
        return error.Cancelled;
    }
    if (stream.yields()) {
        stream.keepInterruptedPrefill(@intCast(f.at));
        return error.Yielded;
    }
    const cache = try self.cacheOf(stream);
    if (comptime @TypeOf(self.e) == *E.Engine and @hasField(@TypeOf(self.*), "last_mtp_normed")) {
        self.last_mtp_normed = null;
        if (comptime @hasField(@TypeOf(self.*), "last_mtp_direct_expected")) self.last_mtp_direct_expected = null;
    }
    var images: ?E.Images = if (f.media) |m| .{ .features = m.features.ptr, .src = m.src, .pos = m.pos } else null;
    // TF_QWEN35_CHUNK_SOLO: a lone uncapped text pass merges cold chunks; a yield-checked (warm) pass keeps the grid
    const rows = self.e.scratch_rows;
    const solo = max_rows == 0 and rows > E.grid_rows and stream.yield_check == null and f.media == null and
        !cuda_solo.othersDecode(self, stream);
    const marks = if (f.kept) |k| k.marks else stream.reuse.marks;
    const end = if (solo) soloEnd(ids.len, stream.chunks, f, rows, marks) else capped(ids.len, stream.chunks, f, max_rows);
    const forward_end = targetForwardEnd(f.at, end, f.mtp_tap_from, f.stepped, f.direct_mtp_norm);
    if (comptime @hasField(EngineType, "mtp_prompt")) {
        const previous_tap = self.e.mtp_prompt.setForPrompt(f.mtp_tap_from, forward_end, f.direct_mtp_norm, f.mtp_prompt_taps);
        defer self.e.mtp_prompt.restoreTapConfig(previous_tap);
        // A chunk without readers stops its last layer at the state writes.
        const unread = forward_end < ids.len and prompt_tail.unread(self.e, images != null);
        try self.e.forward(&.{.{ .cache = cache, .rows = forward_end - f.at, .images = if (images) |*im| im else null }}, ids[f.at..forward_end], if (unread) .none else .last);
    } else {
        const unread = forward_end < ids.len and prompt_tail.unread(self.e, images != null);
        try self.e.forward(&.{.{ .cache = cache, .rows = forward_end - f.at, .images = if (images) |*im| im else null }}, ids[f.at..forward_end], if (unread) .none else .last);
    }
    if (comptime @TypeOf(self.e) == *E.Engine and @hasField(@TypeOf(self.*), "last_mtp_normed")) {
        if (f.direct_mtp_norm) {
            const valid_from = @min(forward_end, @max(f.at, f.mtp_tap_from orelse f.at));
            self.last_mtp_normed = .{ .stream = stream, .chunk_start = f.at, .end = forward_end, .valid_from = valid_from, .epoch = self.e.mtp_prompt.epoch };
            if (comptime @hasField(@TypeOf(self.*), "last_mtp_direct_expected")) {
                self.last_mtp_direct_expected = .{ .stream = stream, .chunk_start = f.at, .end = forward_end, .valid_from = valid_from, .epoch = self.e.mtp_prompt.epoch };
            }
        }
    }
    lanes.prefill_widths.completed(self.gpa, stream, forward_end - f.at);
    f.at = forward_end;
    stream.fill_at.store(@intCast(forward_end), .monotonic);
    // a mark the store asked for, where the cold pass cuts (a resume's only place): the lane host copies the state here
    const cold = forward_end == f.cold; // a capped step's end inside a cold chunk keeps nothing: no pass resumes there
    if (f.kept) |k| {
        if (cold and forward_end < ids.len and std.mem.indexOfScalar(u32, k.marks, @intCast(forward_end)) != null)
            self.prefix.?.keep(cache, ids, forward_end) catch |err| {
                if (err == error.QwenLaneTable) return err; // its table maps pages it may not hold
                std.log.warn("qwen3_5: prompt pages not kept: {s}", .{@errorName(err)});
            };
    } else if (cold and forward_end < ids.len and std.mem.indexOfScalar(u32, stream.reuse.marks, @intCast(forward_end)) != null) {
        if (stream.reuse.hook) |k| k.at(k.ptr, stream, @intCast(forward_end));
    }
    if (forward_end < ids.len) return false;
    // the reply's rows rotate past the image prompt's positions (rope_delta)
    if (f.media) |m| cache.delta = m.delta;
    if (self.pages) |p| p.log(ids.len, stream.cached);
    return true;
}
