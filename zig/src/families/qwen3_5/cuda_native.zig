//! The Qwen hybrids' CUDA lane backend, with live windows and reserved streams.
const std = @import("std");
const cuda = @import("cuda");
const lanes = @import("lanes");
const c = @import("config.zig");
const E = @import("cuda_engine.zig");
pub const warm = @import("cuda_warm.zig");
pub const reservation = @import("cuda_reserve.zig");
pub const window_costs = @import("cuda_window_costs.zig");
pub const cost_adapter = @import("cuda_cost_adapter.zig");
const draws = @import("draw.zig");
const Draft = @import("cuda_draft.zig").Draft;
const vision_prompt = @import("cuda_vision_prompt.zig");
const reuse = @import("cuda_reuse.zig");
const Mtp = @import("cuda_mtp.zig").Mtp;
const be = lanes.backend;
pub const vision = @import("cuda_vision.zig");
pub const slide = @import("cuda_slide.zig");
pub const linked = slide.linked;

pub const model_type = "qwen3_5";
pub const formats: []const []const u8 = &.{"mlx-q4g64"};
pub const default_context: i64 = 32768;
pub const max_segments: u32 = 1;
pub const prompt_rows: u32 = E.max_rows;
/// --drafter loads the DFlash2 draft model (openDrafter); the checkpoints have no head of their own.
pub const takes_drafter = true;

/// Kept prompt states for the shared cache (cuda_reuse.zig), in pinned host memory, on the zero-anchored chunk grid.
pub const snap_bytes = reuse.bytes;
pub const snap_save = reuse.save;
pub const snap_restore = reuse.restore;
pub const snap_drop = reuse.drop;
pub const snap_host = true;
pub const snap_lookahead: u32 = 0;
pub const snap_grid: u32 = E.max_rows;

/// explain() words open refusals too (--vision on a checkpoint without a tower it reads).
pub const explains_open = true;

pub const Options = struct { context: usize, drafts: bool, segments: usize = 1, slide: bool = false, vision: ?vision.Limits = null, kv8: bool = false };

/// What the native server drives: the lane backend, the facts its round loop reads, and how to free it.
pub const Loaded = struct {
    learns: ?*E.Engine = null,
    learn_bytes: usize = 0,
    backend: be.Backend,
    facts: lanes.Model,
    rows: u32,
    /// Device bytes each admitted stream allocates (recurrences, convolution tails, keys and values).
    stream_bytes: usize,
    free_streams: u32 = 0,
    ctx: *anyopaque,
    deinit: *const fn (*anyopaque) void,
    /// Called with the admitted stream count: that many caches allocated and touched now, as memory commits on touch.
    reserve: *const fn (*anyopaque, u32) anyerror!void,
    /// --vision: the tower the server's image requests reach (native/cuda_vision_offer.zig).
    vision: ?*vision.Vision = null,
    /// What the prompt cache's saves copy: the engine and the streams' caches.
    target: *reuse.Target,
};

pub fn open(gpa: std.mem.Allocator, io: std.Io, ctx: *const cuda.Context, dir: []const u8, kernels: ?[]const u8, o: Options) !Loaded {
    if (kernels != null) return error.QwenHasNoCapturedKernels;
    if (o.slide and o.kv8) return error.UnsupportedLearningFormat;
    const e = try E.Engine.init(gpa, io, ctx, dir, o.kv8);
    errdefer e.deinit();
    if (e.g.vocab != c.vocab) return error.UnsupportedQwenVocab; // draw.zig's masks are sized by the 2B's vocabulary
    if (o.slide) _ = try @import("cuda_learn_shape.zig").from(e.g);
    // --vision: the checkpoint's tower, its weights and one call's scratch loaded beside the model's
    const v: ?*vision.Vision = if (o.vision) |lim| try vision.Vision.open(gpa, io, ctx.d, e.stream, dir, e.g.hidden, lim) else null;
    errdefer if (v) |x| x.deinit();
    const l = try gpa.create(Lanes);
    l.* = .{ .gpa = gpa, .io = io, .e = e, .capacity = o.context, .cost_context = ctx, .startup = e.startup, .vision = v, .target = undefined };
    l.target = .{ .e = e, .caches = &l.caches, .seen = &l.seen };
    return .{
        .learns = if (o.slide) e else null,
        .learn_bytes = if (o.slide) @import("cuda_train.zig").memory(try @import("cuda_learn_shape.zig").from(e.g)) + e.streamBytes(o.context) else 0,
        .backend = l.backend(),
        .facts = blk: {
            var facts = laneFacts(@intCast(e.window_rows), @intCast(e.verify_rows));
            facts.root_copies = e.g.hidden == 5120 and e.g.layers == 64 and e.g.experts == 0;
            break :blk facts;
        },
        .rows = @intCast(e.window_rows),
        .stream_bytes = e.streamBytes(o.context),
        .ctx = l,
        .deinit = release,
        .reserve = reserve,
        .vision = v,
        .target = &l.target,
    };
}

pub fn learnModel(p: *anyopaque, dir: []const u8) slide.Model {
    const l: *Lanes = @ptrCast(@alignCast(p));
    return .{ .e = l.e, .dir = dir, .lay = slide.layout(l.e.g.hidden, l.e.g.layers, l.e.g.intermediate, 0) catch unreachable };
}

/// The CUDA hybrids use the shared lane planner for five-token context copies, alone and without a draft model.
pub fn laneFacts(window: u32, verify: u32) lanes.Model {
    return .{ .exact_width = window, .hidden_rows = true, .batch_rows = verify, .max_streams = E.max_segs, .streams_exact = true, .copy_match = 5, .context_copies = true, .lone_cost_tile = 16 };
}

fn reserve(p: *anyopaque, streams: u32) anyerror!void {
    const l: *Lanes = @ptrCast(@alignCast(p));
    try reservation.target(E.Cache, E.Part, l, streams);
}
/// The DFlash2 drafter in `dir` over engine `p`, 4-bit only, and the target's windows timed for the round loop.
pub fn openDrafter(gpa: std.mem.Allocator, io: std.Io, p: *anyopaque, dir: []const u8, bits: u8) !struct { drafter: lanes.drafter.Drafter, ctx: *anyopaque, close: *const fn (*anyopaque) void, reserve: *const fn (*anyopaque, u32) anyerror!void, prepare: struct { ctx: *anyopaque, run: *const fn (*anyopaque) anyerror!lanes.drafter.Facts, bytes: usize } } {
    if (bits != 4) return error.DrafterBitsUnsupported;
    const l: *Lanes = @ptrCast(@alignCast(p));
    if (l.caches.count() != 0 or l.pool.items.len != 0) return error.TapsAfterStreams;
    const d = try Draft.init(gpa, io, l.e, dir);
    errdefer d.deinit();
    try l.e.setTaps(d.taps());
    d.target_taps = l.e.tapBytes();
    cost_adapter.identify(l, dir, .dflash);
    l.calibration = d;
    return .{ .drafter = d.drafter(), .ctx = d, .close = closeDrafter, .reserve = reserveDrafter, .prepare = .{ .ctx = l, .run = calibrateDrafter, .bytes = if (l.capacity < l.e.window_rows) l.e.streamBytes(l.e.window_rows) else 0 } };
}

/// The MTP layer beside the checkpoint in `dir` as the drafter over engine `p` (qwen3_5_moe), its ring the window.
pub fn openMtp(gpa: std.mem.Allocator, io: std.Io, p: *anyopaque, dir: []const u8, bits: u8) !struct { drafter: lanes.drafter.Drafter, ctx: *anyopaque, close: *const fn (*anyopaque) void, reserve: *const fn (*anyopaque, u32) anyerror!void } {
    if (bits != 4) return error.DrafterBitsUnsupported;
    const l: *Lanes = @ptrCast(@alignCast(p));
    if (l.caches.count() != 0 or l.pool.items.len != 0) return error.TapsAfterStreams;
    const d = try Mtp.init(gpa, io, l.e, dir, l.capacity);
    errdefer d.deinit();
    try l.e.setTaps(d.taps());
    cost_adapter.identify(l, dir, .mtp);
    try l.timeWindows(d, null);
    return .{ .drafter = d.drafter(), .ctx = d, .close = closeMtp, .reserve = reserveMtp };
}

fn reserveMtp(p: *anyopaque, streams: u32) anyerror!void {
    const d: *Mtp = @ptrCast(@alignCast(p));
    try d.reserve(streams);
}

fn closeMtp(p: *anyopaque) void {
    const d: *Mtp = @ptrCast(@alignCast(p));
    d.deinit();
}

fn calibrateDrafter(p: *anyopaque) !lanes.drafter.Facts {
    const l: *Lanes = @ptrCast(@alignCast(p));
    const d = l.calibration orelse return error.MissingCalibrationDrafter;
    if (d.n_costs == 0) {
        if (l.pool.items.len == 0) return error.MissingReservedQwenCache;
        try l.timeWindows(d, l.pool.items[0]);
    }
    if (d.n_costs != l.e.window_rows) return error.IncompleteWindowMeasurements;
    l.calibration = null;
    var phases = l.startup;
    phases.add(d.startup);
    phases.print();
    return d.drafter().facts();
}

fn reserveDrafter(p: *anyopaque, streams: u32) anyerror!void {
    const d: *Draft = @ptrCast(@alignCast(p));
    try d.reserve(streams);
}

fn closeDrafter(p: *anyopaque) void {
    const d: *Draft = @ptrCast(@alignCast(p));
    d.deinit();
}

/// A request this engine refuses, in words, or with no engine (null) its open; null: none of its own.
pub fn explain(ctx: ?*anyopaque, err: anyerror) ?[]const u8 {
    if (ctx == null) return switch (err) {
        error.NoQwenVision => "--vision: this checkpoint has no image tower (no vision_config): serve it without --vision",
        error.UnsupportedQwenVision => "--vision: this checkpoint's image tower or text M-RoPE is not one the CUDA engine reads",
        else => null,
    };
    return switch (err) {
        error.InvalidQwenPrompt => "the prompt and its reply exceed this server's context window: shorten it or lower max_tokens",
        error.BadImagePlaceholders => "the image placeholders do not match the prepared images",
        error.BadImageGrid, error.BadImagePatches => "an image's patches do not fit the vision tower",
        error.QwenServedWithoutVision => "image inputs need the server started with --vision",
        error.OutOfMemory => "the GPU had no memory left for this request's caches: retry once another request ends",
        else => null,
    };
}

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
    try std.testing.expect(coldCut(1000, &.{}, E.max_rows * 3));
    try std.testing.expect(!coldCut(1000, &.{}, E.max_rows * 3 + 1));
    try std.testing.expect(coldCut(1000, &.{200}, 200));
    try std.testing.expect(coldCut(1000, &.{200}, 200 + E.max_rows));
    try std.testing.expect(!coldCut(1000, &.{200}, 2 * E.max_rows)); // cut at 200 instead, then 328
    try std.testing.expectEqual(@as(usize, 200), nextCut(1000, &.{200}, E.max_rows));
}

fn release(p: *anyopaque) void {
    const l: *Lanes = @ptrCast(@alignCast(p));
    l.deinit();
}

/// The device draw of a stream's rule at `position`: sample.cu's keyed rule, greedy at temperature 0 or without one.
fn pick(sampling: ?lanes.Sampling, position: u64) E.Draw {
    const s = sampling orelse return .{ .rule = null, .position = position };
    if (s.temperature <= 0) return .{ .rule = null, .position = position };
    return .{ .rule = cuda.sample.Rule.from(lanes.gpu_rule.rule(s)), .position = position };
}

const Lanes = struct {
    gpa: std.mem.Allocator,
    io: std.Io,
    e: *E.Engine,
    capacity: usize,
    cost_context: ?*const cuda.Context = null,
    cost_identity: ?[32]u8 = null,
    cost_kind: cost_adapter.Kind = .dflash,
    startup: warm.profile.Times = .{},
    /// --vision's tower: an image prompt's images encoded at its prefill.
    vision: ?*vision.Vision = null,
    target: reuse.Target,
    caches: std.AutoHashMapUnmanaged(*lanes.Stream, *E.Cache) = .empty,
    /// An image prompt's images as its kept states name them (cuda_reuse.zig), from its prefill to its release.
    seen: std.AutoHashMapUnmanaged(*lanes.Stream, []reuse.Seen) = .empty,
    /// Caches at the full window, kept between streams: allocated and touched once (`reserve`), reset on reuse.
    pool: std.ArrayList(*E.Cache) = .empty,
    warmed: warm.Seen = .{},
    calibration: ?*Draft = null,

    fn deinit(self: *Lanes) void {
        var it = self.caches.valueIterator();
        while (it.next()) |cache| self.drop(cache.*);
        self.caches.deinit(self.gpa);
        var seen = self.seen.valueIterator();
        while (seen.next()) |v| self.gpa.free(v.*);
        self.seen.deinit(self.gpa);
        for (self.pool.items) |cache| self.drop(cache);
        self.pool.deinit(self.gpa);
        if (self.vision) |v| v.deinit();
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

    /// Each width timed on a reset cache after its captured launch shape is warm.
    fn timeWindows(self: *Lanes, d: anytype, reserved: ?*E.Cache) !void {
        self.startup.add(try cost_adapter.measure(self, d, reserved));
    }

    fn backend(self: *Lanes) be.Backend {
        return .{ .ptr = self, .first_before_draft = true, .vtable = &.{ .prefill = prefillFn, .first = firstFn, .first_masked = firstMaskedFn, .queue = queueFn, .read = readFn, .verify = verifyFn, .keep = keepFn, .draft = draftFn, .prepare_features = prepareFeaturesFn, .features = featuresFn, .release = releaseFn } };
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
        // an image prompt: its images through the tower first, then every chunk at its t, h, w positions
        var media: ?vision_prompt.Media = null;
        if (stream.images.len > 0) {
            const v = self.vision orelse return error.QwenServedWithoutVision;
            media = try vision_prompt.Media.init(self.gpa, v, ids, stream.images);
        }
        defer if (media) |*m| m.deinit();
        const images: ?E.Images = if (media) |m| .{ .features = m.features.ptr, .src = m.src, .pos = m.pos } else null;
        const seen = try self.seenOf(stream.images);
        self.seen.put(self.gpa, stream, seen) catch |err| {
            self.gpa.free(seen);
            return err;
        };
        errdefer if (self.seen.fetchRemove(stream)) |kv| self.gpa.free(kv.value);
        const cache = try self.take();
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
        while (at < ids.len) {
            const end = nextCut(ids.len, stream.chunks, at);
            try self.e.forward(&.{.{ .cache = cache, .rows = end - at, .images = if (images) |*im| im else null }}, ids[at..end], .last);
            at = end;
            // a mark the store asked for, where this pass cut anyway: the lane host copies the state here
            if (end < ids.len and std.mem.indexOfScalar(u32, stream.reuse.marks, @intCast(end)) != null) {
                if (stream.reuse.hook) |k| k.at(k.ptr, stream, @intCast(end));
            }
        }
        // the reply's rows rotate past the image prompt's positions (rope_delta)
        if (media) |m| cache.delta = m.delta;
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

    /// Each window (a pending token and drafts) checked, then as many as fit a forward; a tree's rows keyed by depth.
    fn verifyFn(ptr: *anyopaque, windows: []const be.Window, out: []be.Verified) !void {
        const self = cast(ptr);
        const t0 = std.Io.Clock.awake.now(self.io);
        defer @import("cuda_draft.zig").timing("verify", self.io, t0);
        if (windows.len != out.len) return error.InvalidQwenWindows;
        for (windows, out) |w, o| {
            const rows = w.rows();
            if (w.held != 0 or rows > self.e.window_rows or w.positions.len != rows) return error.UnsupportedQwenWindow;
            if (w.masks.len != 0 and w.masks.len != rows * draws.mask_words) return error.InvalidQwenMasks;
            if (o.sampled.len != rows or o.drafts.len != w.tokens.len) return error.InvalidQwenOutputs;
            const cache = try self.cacheOf(w.stream);
            var depth: [E.most_window]u64 = undefined;
            for (w.positions, 0..) |p, r| {
                const parent: i32 = if (w.parents) |ps| ps[r] else if (r == 0) -1 else @intCast(r - 1);
                if (r == 0 and parent != -1) return error.UnsupportedQwenWindow;
                if (r > 0 and (parent < 0 or parent >= @as(i32, @intCast(r)))) return error.UnsupportedQwenWindow;
                depth[r] = if (parent < 0) 0 else depth[@intCast(parent)] + 1;
                if (p != cache.len + depth[r] + 1) return error.InvalidQwenPosition;
            }
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
                parts[n] = .{ .cache = self.caches.get(w.stream).?, .rows = w.rows(), .parents = w.parents };
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

    /// The accepted path of each window (a chain's prefix or a tree's branch): the state as that chain leaves it.
    fn keepFn(ptr: *anyopaque, windows: []const be.Window, paths: []const []const u32) !void {
        const self = cast(ptr);
        const t0 = std.Io.Clock.awake.now(self.io);
        defer @import("cuda_draft.zig").timing("keep", self.io, t0);
        for (windows, paths) |w, path| {
            const cache = try self.cacheOf(w.stream);
            if (cache.last == null) continue; // already kept: nothing to roll back
            try cache.keep(self.e, path);
        }
    }

    fn draftFn(_: *anyopaque, requests: []const be.DraftRequest) !void {
        for (requests) |r| if (r.depth != 0) return error.QwenCheckpointHasNoDraftHead;
    }

    /// A drafter's tapped layers, before any stream's caches exist: each forward fills their rings after.
    fn prepareFeaturesFn(ptr: *anyopaque, taps: []const u32) !void {
        const self = cast(ptr);
        if (self.caches.count() != 0 or self.pool.items.len != 0) return error.TapsAfterStreams;
        try self.e.setTaps(taps);
    }

    /// Rows [start, start + count) of a stream's tap ring (slot position % tap_rows), the newest tap_rows rows kept.
    fn featuresFn(ptr: *anyopaque, stream: *lanes.Stream, taps: []const u32, start: u64, count: u32) !be.Features {
        const self = cast(ptr);
        if (taps.len != self.e.n_taps) return error.TapsMismatch;
        const cache = try self.cacheOf(stream);
        const ring = cache.tap orelse return error.NoTaps;
        // a long prompt asks for more rows than the ring keeps; the drafter reads only its window of the newest
        const held_from = cache.written -| E.tap_rows;
        if (start + count > cache.written or start + count <= held_from) return error.RowsNotHeld;
        return .{ .buffer = ring.ptr, .offset = start, .rows = count, .row_bytes = @intCast(self.e.n_taps * self.e.g.hidden * 2), .space = .device, .dtype = .bf16 };
    }

    fn releaseFn(ptr: *anyopaque, stream: *lanes.Stream) void {
        const self = cast(ptr);
        if (self.caches.fetchRemove(stream)) |entry| self.give(entry.value);
        if (self.seen.fetchRemove(stream)) |entry| self.gpa.free(entry.value);
    }

    /// Each image's first placeholder row and identity: a kept state resumes only the same images.
    fn seenOf(self: *Lanes, images: []const lanes.stream.Image) ![]reuse.Seen {
        const seen = try self.gpa.alloc(reuse.Seen, images.len);
        for (seen, images) |*s, im| s.* = .{ .at = im.at, .id = im.hash };
        return seen;
    }
};
