//! LFM2 conv state, windows and keyed draws behind the existing lane engine's backend interface.
const std = @import("std");
const platform = @import("sys");
const mtl = @import("metal");
const lanes = @import("lanes");
const c = @import("config.zig");
const st = @import("state.zig");
const fwd = @import("forward.zig");
const snapshot = @import("snapshot.zig");
const topk = @import("draw_topk.zig");
const draw_timing = @import("draw_timing.zig");
const Model = @import("model.zig").Model;
const be = lanes.backend;

/// Where a drafter holds a stream's drafts on the device, for verify windows that take them there (Window.held).
pub const HeldSource = struct {
    ptr: *anyopaque,
    get: *const fn (ptr: *anyopaque, s: *lanes.Stream) ?Device,
    host: *const fn (ptr: *anyopaque, s: *lanes.Stream, out: []u32) anyerror!void, // where `get` gives none
};
pub const Device = struct { src: mtl.Buffer, off: usize, stride: u32, wait: fwd.Wait };

/// Where a drafter owns a stream's tapped rows on the device, for forwards to capture into (Metal.stage_source).
pub const StageSource = struct {
    ptr: *anyopaque,
    /// Readies the stream's stage for rows [start, start + rows) of a forward (null: none kept for them).
    bind: *const fn (ptr: *anyopaque, s: *lanes.Stream, start: u64, rows: usize) anyerror!?Stage,
    /// The stream's stage buffer now (null: released).
    buffer: *const fn (ptr: *anyopaque, s: *lanes.Stream) ?mtl.Buffer,
};
/// Row `start`'s place in the stage, and the hold still reading the stage that the forward waits for.
pub const Stage = struct { place: fwd.Place, wait: ?fwd.Wait };

pub const Options = struct { capacity: usize, chunk: usize = 128, streams: u32 = 8 };

pub const Metal = struct {
    residency: ?mtl.ResidencySet.Follow = null,
    gpa: std.mem.Allocator,
    model: *Model,
    options: Options,
    scratch: st.Scratch,
    width: u32 = 1,
    caches: std.AutoHashMapUnmanaged(*lanes.Stream, *st.Cache) = .empty,
    /// Forward ms by window width, timed at load: the lane core weighs drafts against them.
    costs: [st.window_rows]lanes.config.Cost = undefined,
    timed: usize = 0,
    /// Shared-row increments past the qualified window, anchored at that window's measured cost.
    shared: [st.batch_rows - st.window_rows]lanes.config.Cost = undefined,
    shared_timed: usize = 0,
    /// Forwards run so far: a stream's tapped rows in the scratch are current while their gen matches.
    gen: u64 = 0,
    /// A drafter whose drafts verify windows read on the device (null: windows carry host tokens only).
    held_source: ?HeldSource = null,
    /// A drafter whose staged rows all serving forwards capture into (install before preparing taps).
    stage_source: ?StageSource = null,

    pub fn init(gpa: std.mem.Allocator, model: *Model, options: Options) !*Metal {
        const out = try gpa.create(Metal);
        errdefer gpa.destroy(out);
        out.* = .{ .gpa = gpa, .model = model, .options = options, .scratch = try st.Scratch.init(gpa, model.device, &model.config.attention, @max(options.chunk, st.batch_rows)) };
        errdefer out.scratch.deinit();
        @import("qualify.zig").check(model, &out.scratch) catch |err| switch (err) {
            error.LfmCacheLengthMismatch, error.LfmConvStateMismatch, error.LfmAttentionCacheMismatch, error.LfmWindowLogitsMismatch, error.LfmCommittedLogitsMismatch => {
                std.log.warn("LFM2 wider lanes failed the local exactness check ({s}); keeping one-row lane rounds", .{@errorName(err)});
                return out;
            },
            else => return err,
        };
        out.width = st.window_rows;
        return out;
    }

    /// Price before preparing tap destinations, after committing the admitted working set.
    pub fn measure(self: *Metal, flush: ?mtl.keepalive.Target.Flush) !void {
        if (self.width == 1) return;
        self.timed = try self.time(flush);
    }

    pub fn deinit(self: *Metal) void {
        mtl.resident.streams.set(self, null);
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
        var out = factsFor(self.width, self.options.streams, self.costs[0..self.timed]);
        out.shared_costs = self.shared[0..self.shared_timed];
        return out;
    }

    /// Each window width's forward ms on a 256-token history (the fastest of three, after the GPU's clocks ramp up).
    fn time(self: *Metal, flush: ?mtl.keepalive.Target.Flush) !usize {
        var cache = try st.Cache.init(self.gpa, self.model.device, &self.model.config.attention, 512);
        defer cache.deinit();
        cache.memory.setResident(self.residency);
        if (flush) |f| f.run(f.ctx);
        var ids: [256]u32 = undefined;
        for (&ids, 0..) |*id, i| id.* = @intCast((i * 7919 + 13) % c.vocab);
        try fwd.run(self.model, &self.scratch, &.{.{ .cache = &cache, .rows = 256 }}, &ids, false, .last);
        const base = cache.len;
        for (0..8) |_| {
            try fwd.run(self.model, &self.scratch, &.{.{ .cache = &cache, .rows = 1 }}, ids[0..1], true, .all);
            cache.len = base;
        }
        for (&self.costs, 1..) |*cost, w| {
            var best: f64 = std.math.floatMax(f64);
            for (0..3) |_| {
                var t0: platform.c.timespec = undefined;
                var t1: platform.c.timespec = undefined;
                _ = platform.c.clock_gettime(platform.c.CLOCK.MONOTONIC, &t0);
                try fwd.run(self.model, &self.scratch, &.{.{ .cache = &cache, .rows = w }}, ids[0..w], true, .all);
                _ = platform.c.clock_gettime(platform.c.CLOCK.MONOTONIC, &t1);
                cache.len = base;
                const ms = @as(f64, @floatFromInt(t1.sec - t0.sec)) * 1e3 + @as(f64, @floatFromInt(t1.nsec - t0.nsec)) / 1e6;
                best = @min(best, ms);
            }
            cost.* = .{ .width = @intCast(w), .ms = best };
        }
        std.log.info("LFM2 window ms by width: 1 {d:.1}, 2 {d:.1}, 4 {d:.1}, 8 {d:.1}, 16 {d:.1}", .{ self.costs[0].ms, self.costs[1].ms, self.costs[3].ms, self.costs[7].ms, self.costs[15].ms });
        // Two qualified windows measure shared-row increments against the same shared shape at 16 rows.
        var other = try st.Cache.init(self.gpa, self.model.device, &self.model.config.attention, 512);
        defer other.deinit();
        other.memory.setResident(self.residency);
        if (flush) |f| f.run(f.ctx);
        try fwd.run(self.model, &self.scratch, &.{.{ .cache = &other, .rows = 256 }}, &ids, false, .last);
        var shared_base: f64 = 0;
        for (st.window_rows..st.batch_rows + 1) |rows| {
            const first = @min(st.window_rows, rows - 1);
            var best: f64 = std.math.floatMax(f64);
            for (0..4) |trial| {
                var t0: platform.c.timespec = undefined;
                var t1: platform.c.timespec = undefined;
                _ = platform.c.clock_gettime(platform.c.CLOCK.MONOTONIC, &t0);
                try fwd.run(self.model, &self.scratch, &.{ .{ .cache = &cache, .rows = first }, .{ .cache = &other, .rows = rows - first } }, ids[0..rows], true, .all);
                _ = platform.c.clock_gettime(platform.c.CLOCK.MONOTONIC, &t1);
                const ms = @as(f64, @floatFromInt(t1.sec - t0.sec)) * 1e3 + @as(f64, @floatFromInt(t1.nsec - t0.nsec)) / 1e6;
                cache.len = base;
                other.len = base;
                if (trial > 0) best = @min(best, ms);
            }
            if (rows == st.window_rows) {
                shared_base = best;
            } else self.shared[rows - st.window_rows - 1] = .{ .width = @intCast(rows), .ms = self.costs[st.window_rows - 1].ms + best - shared_base };
        }
        self.shared_timed = self.shared.len;
        std.log.info("LFM2 shared calibration: window16 {d:.2} ms, shared16 {d:.2} ms", .{ self.costs[st.window_rows - 1].ms, shared_base });
        std.log.info("LFM2 shared marginal prices by rows: 17 {d:.1}, 24 {d:.1}, 32 {d:.1}", .{ self.shared[0].ms, self.shared[7].ms, self.shared[15].ms });
        return self.costs.len;
    }

    pub fn backend(self: *Metal) be.Backend {
        return .{ .ptr = self, .vtable = &.{ .prefill = prefillFn, .prepare_features = prepareFn, .features = featuresFn, .prefill_features = prefillFeaturesFn, .first = firstFn, .first_masked = firstMaskedFn, .queue = queueFn, .read = readFn, .verify = verifyFn, .keep = keepFn, .draft = draftFn, .release = releaseFn } };
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
        cache.* = try st.Cache.init(self.gpa, self.model.device, &self.model.config.attention, capacity);
        cache.memory.setResident(self.residency);
        return cache;
    }

    /// One forward on the scratch; a stream's tapped rows from earlier forwards are stale after it.
    fn run(self: *Metal, segments: []const fwd.Segment, tokens: []const u32, record: bool, head: fwd.Head, wait: ?fwd.Wait) !void {
        self.gen += 1;
        try fwd.runHeld(self.model, &self.scratch, segments, tokens, record, head, &.{}, wait);
    }

    /// Where a stream's forward rows from `start` are captured, the hold it follows folded into `wait`.
    fn tapFor(self: *Metal, stream: *lanes.Stream, start: usize, rows: usize, wait: *?fwd.Wait) !fwd.Tap {
        if (stream.head_context == false) return .none;
        const src = self.stage_source orelse return .scratch;
        if (self.scratch.taps == null) return .scratch;
        const stage = (try src.bind(src.ptr, stream, start, rows)) orelse
            return if (stream.headDrafts()) error.LfmTapsNotHeld else .none;
        if (stage.wait) |w| wait.* = if (wait.*) |x| (if (x.value >= w.value) x else w) else w;
        return .{ .stage = stage.place };
    }

    /// The stream's latest rows after a forward that captured them by `tap` (none: not held).
    fn tapped(self: *const Metal, cache: *const st.Cache, rows: usize, tap: fwd.Tap, row: usize) ?st.TapRows {
        if (self.scratch.taps == null or tap == .none) return null;
        const stage: ?usize = switch (tap) {
            .stage => |p| p.off,
            else => null,
        };
        return .{ .start = cache.len - rows, .row = row, .rows = rows, .gen = self.gen, .stage = stage };
    }

    fn prefillFn(ptr: *anyopaque, stream: *lanes.Stream) !void {
        return cast(ptr).prefillWith(stream, null);
    }

    /// The prompt pass, `sink` taking each chunk's tapped rows before the chunk's end is kept for reuse.
    fn prefillFeaturesFn(ptr: *anyopaque, stream: *lanes.Stream, layers: []const u32, sink: be.FeatureSink) !void {
        const self = cast(ptr);
        const t = self.scratch.taps orelse return error.UnpreparedLfmTaps;
        if (layers.len != t.count) return error.UnpreparedLfmTaps;
        return self.prefillWith(stream, sink);
    }

    /// A tapped row's bytes: every tap's hidden row, bf16.
    fn tapBytes(self: *const Metal) usize {
        return if (self.scratch.taps) |t| t.count * c.hidden * 2 else 0;
    }

    /// The drafter's layers: every forward from now on keeps their outputs for each row.
    fn prepareFn(ptr: *anyopaque, layers: []const u32) !void {
        const self = cast(ptr);
        if (layers.len == 0 or layers.len > st.max_taps) return error.BadLfmTaps;
        for (layers) |l| if (l >= c.layers) return error.BadLfmTaps;
        if (self.scratch.taps) |t| return if (std.mem.eql(u32, t.layers[0..t.count], layers)) {} else error.BadLfmTaps;
        const table_at = if (self.stage_source == null) self.scratch.rows * layers.len * c.hidden * 2 else 0;
        const buf = try self.scratch.memory.alloc(table_at + self.scratch.rows * @sizeOf(u64));
        var t: st.Taps = .{ .count = @intCast(layers.len), .buf = buf, .gpu_base = buf.gpuAddress(), .table_at = table_at };
        @memcpy(t.layers[0..layers.len], layers);
        self.scratch.taps = t;
    }

    /// Cache rows [start, start + count) of the stream's latest tapped rows, on the device.
    fn featuresFn(ptr: *anyopaque, stream: *lanes.Stream, layers: []const u32, start: u64, count: u32) !be.Features {
        const self = cast(ptr);
        const t = self.scratch.taps orelse return error.UnpreparedLfmTaps;
        if (layers.len != t.count) return error.UnpreparedLfmTaps;
        const cache = try self.cacheOf(stream);
        const held = cache.tap orelse return error.LfmTapsNotHeld;
        if (count == 0 or start < held.start or start + count > held.start + held.rows) return error.LfmTapsNotHeld;
        const width = self.tapBytes();
        var at = (held.row + @as(usize, @intCast(start - held.start))) * width;
        if (held.gen != 0 and held.gen != self.gen) return error.StaleLfmTaps;
        const buf = if (held.stage) |off| blk: {
            const src = self.stage_source orelse return error.LfmTapsNotHeld;
            at = off + @as(usize, @intCast(start - held.start)) * width;
            break :blk src.buffer(src.ptr, stream) orelse return error.LfmTapsNotHeld;
        } else if (held.gen == 0) cache.last_tap orelse return error.LfmTapsNotHeld else t.buf;
        return .{ .buffer = @intFromPtr(buf.id), .offset = at, .rows = count, .row_bytes = @intCast(width), .space = .device, .dtype = .bf16 };
    }

    /// The prompt's last row's taps, copied beside the stream's caches for its first draft (gen 0: kept there).
    fn keepLastTap(self: *Metal, cache: *st.Cache, row: usize, position: usize) !void {
        const width = self.tapBytes();
        if (width == 0) return;
        if (cache.last_tap == null) cache.last_tap = try cache.memory.alloc(width);
        @memcpy(cache.last_tap.?.contents()[0..width], (self.scratch.taps.?.buf.contents() + row * width)[0..width]);
        cache.tap = .{ .start = position, .row = 0, .rows = 1, .gen = 0 };
    }

    fn prefillWith(self: *Metal, stream: *lanes.Stream, sink: ?be.FeatureSink) !void {
        const ids = stream.prompt();
        if (ids.len == 0 or ids.len + stream.max_new > self.options.capacity or self.caches.contains(stream)) return error.InvalidLfmPrompt;
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
            // a drafting stream resumes only a state kept with its drafter's context
            const drafter_less = sink != null and snap.draft == null;
            if (snap.at < ids.len and !drafter_less) if (snapshot.restore(cache, snap)) |_| {
                at = snap.at;
            } else |_| {};
            stream.cached = @intCast(at);
            stream.reuse_failed = at == 0;
        }
        var mark: usize = 0;
        while (at < ids.len) {
            var end: usize = @min(ids.len, at + self.options.chunk);
            for (stream.chunks) |boundary| if (boundary > at) {
                end = @min(end, boundary);
                break;
            };
            while (mark < stream.reuse.marks.len and stream.reuse.marks[mark] <= at) mark += 1;
            if (mark < stream.reuse.marks.len) end = @min(end, stream.reuse.marks[mark]);
            var wait: ?fwd.Wait = null;
            const tap = try self.tapFor(stream, at, end - at, &wait);
            if (sink != null and tap == .none) return error.LfmTapsNotHeld;
            try self.run(&.{.{ .cache = cache, .rows = end - at, .tap = tap }}, ids[at..end], false, .last, wait);
            if (sink) |k| {
                cache.tap = self.tapped(cache, end - at, tap, 0);
                const f = try featuresFn(self, stream, self.scratch.taps.?.layers[0..self.scratch.taps.?.count], at, @intCast(end - at));
                try k.take(k.ptr, stream, f, at);
                if (end == ids.len) {
                    if (tap == .stage) {
                        // The lane owns this anchor across other streams' forwards; no scratch copy.
                        cache.tap = .{ .start = end - 1, .row = 0, .rows = 1, .gen = 0, .stage = tap.stage.off + (end - at - 1) * self.tapBytes() };
                    } else try self.keepLastTap(cache, end - 1 - at, end - 1);
                }
            }
            lanes.prefill_widths.completed(self.gpa, stream, end - at);
            at = end;
            if (stream.reuse.hook) |hk| if (std.mem.indexOfScalar(u32, stream.reuse.marks, @intCast(at)) != null)
                hk.at(hk.ptr, stream, @intCast(at));
        }
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
            .handle => |h| @intCast(h),
        };
        var wait: ?fwd.Wait = null;
        const tap = try self.tapFor(stream, cache.len, 1, &wait);
        try self.run(&.{.{ .cache = cache, .rows = 1, .tap = tap }}, &.{token}, false, .all, wait);
        cache.tap = self.tapped(cache, 1, tap, 0);
        return try draw(self.gpa, cache.logits.slice(u16, c.vocab), stream.sampling, position, &.{});
    }

    fn readFn(_: *anyopaque, handle: u64) !u32 {
        if (handle >= c.vocab) return error.InvalidLfmToken;
        return @intCast(handle);
    }

    fn verifyFn(ptr: *anyopaque, windows: []const be.Window, out: []be.Verified) !void {
        const self = cast(ptr);
        draw_timing.reset(windows);
        errdefer draw_timing.reset(windows);
        var segments: [st.batch_rows]fwd.Segment = undefined;
        var ids: [st.batch_rows]u32 = undefined;
        var held: [st.batch_rows]fwd.Held = undefined;
        var count: usize = 0;
        var wait: ?fwd.Wait = null;
        var total: usize = 0;
        if (windows.len > segments.len or windows.len != out.len) return error.InvalidLfmWindows;
        for (windows, 0..) |w, i| {
            const rows = w.rows();
            if (w.held != 0) { // the drafter's picks, copied into the window's rows on the GPU
                const source = self.held_source orelse return error.UnsupportedLfmWindow;
                if (total + rows > st.batch_rows) return error.UnsupportedLfmWindow;
                if (source.get(source.ptr, w.stream)) |dev| {
                    held[count] = .{ .row = total + 1, .src = dev.src, .off = dev.off, .stride = dev.stride, .count = w.held };
                    count += 1;
                    wait = if (wait) |x| (if (x.value >= dev.wait.value) x else dev.wait) else dev.wait;
                    @memset(ids[total + 1 ..][0..w.held], 0);
                } else try source.host(source.ptr, w.stream, ids[total + 1 ..][0..w.held]);
            }
            if (w.parents != null or rows > st.window_rows or total + rows > st.batch_rows or w.positions.len != rows) return error.UnsupportedLfmWindow;
            if (w.masks.len != 0 and w.masks.len != rows * mask_words) return error.InvalidLfmMasks;
            const cache = try self.cacheOf(w.stream);
            for (w.positions, 0..) |p, r| if (p != cache.len + r + 1) return error.InvalidLfmPosition;
            segments[i] = .{ .cache = cache, .rows = rows, .tap = try self.tapFor(w.stream, cache.len, rows, &wait) };
            ids[total] = w.pending;
            @memcpy(ids[total + 1 + w.held ..][0..w.tokens.len], w.tokens);
            total += rows;
        }
        self.gen += 1;
        try fwd.runHeld(self.model, &self.scratch, segments[0..windows.len], ids[0..total], true, .all, held[0..count], wait);
        for (segments[0..windows.len]) |seg| {
            const last = &seg.cache.last.?;
            last.committed = &self.scratch;
            seg.cache.tap = self.tapped(seg.cache, last.rows, seg.tap, last.base);
        }
        const logits = self.scratch.logits.slice(u16, total * c.vocab);
        const best = self.scratch.best.slice(u32, 2 * total);
        var base: usize = 0;
        for (windows, out) |w, o| {
            if (o.sampled.len != w.rows() or o.drafts.len != w.rows() - 1) return error.InvalidLfmOutputs;
            try draw_timing.drawWindow(drawRow, self.gpa, w, o.sampled, logits[base * c.vocab ..][0 .. w.rows() * c.vocab], best[2 * base ..][0 .. 2 * w.rows()], draw_timing.Monotonic{});
            @memcpy(o.drafts, self.scratch.ids.slice(u32, base + w.rows())[base + 1 ..][0 .. w.rows() - 1]);
            base += w.rows();
        }
    }
    fn keepFn(ptr: *anyopaque, windows: []const be.Window, paths: []const []const u32) !void {
        const self = cast(ptr);
        for (windows, paths) |w, path| try (try self.cacheOf(w.stream)).keep(&self.scratch, path);
    }

    fn draftFn(_: *anyopaque, requests: []const be.DraftRequest) !void {
        for (requests) |r| if (r.depth != 0) return error.LfmCheckpointHasNoDraftHead;
    }

    fn releaseFn(ptr: *anyopaque, stream: *lanes.Stream) void {
        const self = cast(ptr);
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
        return snapshot.bytes(&metal(ptr).model.config.attention, at);
    }
    pub fn save(ptr: *anyopaque, owner: ?*anyopaque, at: u32) anyerror!*anyopaque {
        const self = metal(ptr);
        const cache = try self.cacheOf(@ptrCast(@alignCast(owner orelse return error.NoStream)));
        return @ptrCast(try snapshot.save(self.gpa, &self.model.config.attention, cache, at));
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

/// The GPU has checked every logit; its flag is authoritative even when a mask hides the bad token.
inline fn drawRow(gpa: std.mem.Allocator, logits: []const u16, settings: ?lanes.Sampling, position: u64, mask: []const u32, picked: [2]u32) !u32 {
    if (picked[1] != 0) return error.NonfiniteLfmLogits;
    if (mask.len > 0) return draw(gpa, logits, settings, position, mask);
    const sampling = settings orelse return picked[0];
    if (sampling.temperature <= 0) return picked[0];
    if (try topk.draw(logits, sampling, position, mask)) |token| return token;
    return sampleFinite(gpa, logits, sampling, position, mask, logits.len);
}

/// The unchanged top-k candidates for the shared fp64 keyed sampler; a `mask` keeps its allowed ones (-inf's draw).
pub fn draw(gpa: std.mem.Allocator, logits: []const u16, settings: ?lanes.Sampling, position: u64, mask: []const u32) !u32 {
    if (settings) |s| if (try topk.draw(logits, s, position, mask)) |token| return token;
    var best: ?usize = null;
    var allowed: usize = 0;
    for (logits, 0..) |word, id| {
        const x = value(word);
        if (!std.math.isFinite(x)) return error.NonfiniteLfmLogits;
        if (mask.len > 0 and !lanes.grammar.allows(mask, id)) continue;
        allowed += 1;
        if (best == null or x > value(logits[best.?])) best = id;
    }
    if (best == null) return error.NoAllowedLfmToken;
    const sampling = settings orelse return @intCast(best.?);
    if (sampling.temperature <= 0) return @intCast(best.?);
    return sampleFinite(gpa, logits, sampling, position, mask, allowed);
}

/// Finite logits and a nonempty allowed set; both checked by `draw` or the GPU row's pick.
fn sampleFinite(gpa: std.mem.Allocator, logits: []const u16, sampling: lanes.Sampling, position: u64, mask: []const u32, allowed: usize) !u32 {
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

/// The lane facts at `width` exact rows: at one row (the wider check failed) no forward is shared between streams.
fn factsFor(width: u32, streams: u32, costs: []const lanes.config.Cost) lanes.Model {
    return .{
        .limit_shared_drafts = true,
        .exact_width = width,
        .hidden_rows = width > 1,
        .batch_rows = st.batch_rows,
        .max_streams = streams,
        .streams_exact = true,
        .window_costs = costs,
        .plain_guard = true,
    };
}

test {
    _ = @import("segments.zig"); // a shared round's table, with the family's tests
    try @import("draw_timing_test.zig").check(drawRow);
}

test "after a failed wider-lanes check every forward is one stream's one row, so no window or by-expert twin runs" {
    const gpa = std.testing.allocator;
    var one = try lanes.Config.init(gpa, factsFor(1, 8, &.{}), st.window_rows, st.window_rows - 1);
    defer one.deinit(gpa);
    try std.testing.expect(!one.family_streams and one.batch_streams == 1 and one.most_drafts == 0);
    var wide = try lanes.Config.init(gpa, factsFor(st.window_rows, 8, &.{}), st.window_rows, st.window_rows - 1);
    defer wide.deinit(gpa);
    try std.testing.expect(wide.family_streams and wide.batch_streams == 8);
}

test "candidate selection preserves full-vocabulary keyed draws and tie order" {
    try @import("draw_timing_test.zig").checkCandidates(drawRow, value);
}

test "a masked draw is the full-vocabulary draw with the other tokens at -inf" {
    try @import("draw_timing_test.zig").checkMasked(value);
}

test "measured shared rows price the batch without widening a stream's qualified window" {
    var backend_: Metal = .{ .gpa = std.testing.allocator, .model = undefined, .options = .{ .capacity = 512 }, .scratch = undefined, .width = st.window_rows, .timed = 1, .shared_timed = 1 };
    backend_.costs[0] = .{ .width = 1, .ms = 2 };
    backend_.shared[0] = .{ .width = 17, .ms = 9 };
    var cfg = try lanes.Config.init(std.testing.allocator, backend_.facts(), st.window_rows, st.window_rows - 1);
    defer cfg.deinit(std.testing.allocator);
    try std.testing.expectEqual(@as(u32, st.window_rows), cfg.family_width);
    try std.testing.expectEqual(@as(?f64, 2), cfg.family_costs.get(1));
    try std.testing.expectEqual(@as(?f64, null), cfg.family_costs.get(17));
    try std.testing.expectEqual(@as(?f64, 9), cfg.shared_costs.get(17));
}

test "zero grant: an attached LFM2 drafter retains committed taps and resumes after a long shared skip" {
    try lanes.shared_draft_check.check(factsFor(st.window_rows, 16, &.{}), true);
}

test "shared head costs: an attached LFM2 drafter retains costed copy grants" {
    for ([_]bool{ false, true }) |guard| for ([_]bool{ false, true }) |batched| {
        const m = try lanes.shared_head_cost_check.attached(factsFor(st.window_rows, 16, &.{}), .{
            .depth = 6,
            .step_ms = 2,
            .plain_guard = guard,
            .batched = batched,
        });
        try lanes.shared_head_cost_check.check(m);
    };
}
