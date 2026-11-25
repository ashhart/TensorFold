//! Inkling 2 Small EXL3 behind the lane engine: exact prompt reuse, multi-stream batching and window rollback.
const std = @import("std");
const builtin = @import("builtin");
const mtl = @import("metal");
const lanes = @import("lanes");
const st = @import("small_state.zig");
const fwd = @import("small_forward.zig");
const Model = @import("small_model.zig").Model;
const snapshot = @import("small_snapshot.zig");
const small_dspark = @import("small_dspark.zig");
const small_dspark_state = @import("small_dspark_state.zig");
const cfg = @import("../config.zig");
const draws = @import("draw.zig");
const be = lanes.backend;

const SavedSnap = struct {
    target: snapshot.Snap,
    dspark: ?small_dspark_state.Snap = null,
};

pub const Options = struct { capacity: usize, streams: u32 = 4, budget: u64 };
const shared_widths = [_]usize{ 24, 32 };
const ring = 4096;

/// The prompt cache gets the bytes left by live streams and never evicts the state being restored.
pub const Room = struct { ctx: *anyopaque, fit: *const fn (ctx: *anyopaque, bytes: u64, restoring: ?*anyopaque, prompt: []const u32) void };

/// Small's DSpark ABI is fixed: eight target taps in checkpoint order.
pub fn validateTaps(c: *const cfg.Config, taps: []const u32) !void {
    const mtp = c.mtp orelse return error.UnsupportedInkling2Taps;
    const expected = [_]u32{ 5, 11, 17, 23, 29, 35, 41, 48 };
    if (mtp.taps_n != expected.len or !std.mem.eql(u32, mtp.taps[0..mtp.taps_n], &expected) or !std.mem.eql(u32, taps, &expected))
        return error.UnsupportedInkling2Taps;
}

pub const Metal = struct {
    gpa: std.mem.Allocator,
    model: *Model,
    options: Options,
    batch_rows: usize,
    prompt_rows: usize,
    scratch: st.Scratch,
    prompt: st.Scratch,
    caches: std.AutoHashMapUnmanaged(*lanes.Stream, *st.Cache) = .empty,
    held: u64 = 0,
    room: ?Room = null,
    tokens: []u32,
    next: u64 = 0,
    values: []f32,
    prepared: bool = false,
    sink: ?be.FeatureSink = null,
    costs: [st.window_rows]lanes.config.Cost = undefined,
    timed: usize = 0,
    shared: [shared_widths.len]lanes.config.Cost = undefined,
    shared_timed: usize = 0,
    tap_gen: u64 = 0,
    drafter: ?*small_dspark.Drafter = null,

    pub fn init(gpa: std.mem.Allocator, model: *Model, options: Options) !*Metal {
        return initFor(gpa, model, options, st.batch_rows, st.prompt_rows);
    }

    pub fn initFor(gpa: std.mem.Allocator, model: *Model, options: Options, batch_rows: usize, prompt_rows: usize) !*Metal {
        const out = try gpa.create(Metal);
        errdefer gpa.destroy(out);
        const vocab_n = model.config.text.unpadded_vocab;
        var scratch = try st.Scratch.initFor(model.device, batch_rows, vocab_n);
        errdefer scratch.deinit();
        var prompt = try st.Scratch.initFor(model.device, prompt_rows, vocab_n);
        errdefer prompt.deinit();
        const tokens = try gpa.alloc(u32, ring);
        errdefer gpa.free(tokens);
        const values = try gpa.alloc(f32, vocab_n);
        out.* = .{
            .gpa = gpa,
            .model = model,
            .options = options,
            .batch_rows = batch_rows,
            .prompt_rows = prompt_rows,
            .scratch = scratch,
            .prompt = prompt,
            .tokens = tokens,
            .values = values,
        };
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
        self.prompt.deinit();
        self.gpa.free(self.tokens);
        self.gpa.free(self.values);
        self.gpa.destroy(self);
    }

    pub fn createCache(self: *Metal, capacity: usize) !*st.Cache {
        const cache = try self.gpa.create(st.Cache);
        errdefer self.gpa.destroy(cache);
        cache.* = try st.Cache.initFor(self.model.device, capacity, &self.model.weight_plan, self.prompt_rows, self.vocab());
        return cache;
    }

    pub fn facts(self: *const Metal) lanes.Model {
        return .{
            .exact_width = st.window_rows,
            .gpu_tokens = false,
            .hidden_rows = true,
            .batch_rows = @intCast(self.batch_rows),
            .mtp = false,
            .speculate = false,
            .speculate_early = false,
            .drafts = 1,
            .max_streams = self.options.streams,
            .streams_exact = true,
            .window_costs = self.costs[0..self.timed],
            .shared_costs = self.shared[0..self.shared_timed],
        };
    }

    fn prepareFn(ptr: *anyopaque, taps: []const u32) !void {
        const self = cast(ptr);
        try validateTaps(&self.model.config, taps);
        if (self.prepared) return;
        const device = self.model.device;
        const decode = self.scratch.taps orelse try device.buffer(st.captureBytes(self.batch_rows), mtl.ResourceOptions.shared);
        errdefer if (self.scratch.taps == null) decode.deinit();
        const prompt = self.prompt.taps orelse try device.buffer(st.captureBytes(self.prompt_rows), mtl.ResourceOptions.shared);
        errdefer if (self.prompt.taps == null) prompt.deinit();
        self.scratch.taps = decode;
        self.prompt.taps = prompt;
        self.prepared = true;
    }

    fn featuresFn(ptr: *anyopaque, stream: *lanes.Stream, taps: []const u32, start: u64, count: u32) !be.Features {
        const self = cast(ptr);
        try validateTaps(&self.model.config, taps);
        if (!self.prepared) return error.UnpreparedInkling2Taps;
        const tp = (try self.cacheOf(stream)).tap;
        const buf = tp.buf orelse return error.Inkling2TapsNotHeld;
        const off = try st.featureOffset(tp, start, count, self.tap_gen);
        return .{ .buffer = @intFromPtr(buf.id), .offset = off, .rows = count, .row_bytes = @intCast(st.feature_row_bytes), .space = .device, .dtype = .bf16 };
    }

    fn prefillFeaturesFn(ptr: *anyopaque, stream: *lanes.Stream, taps: []const u32, sink: be.FeatureSink) !void {
        const self = cast(ptr);
        try validateTaps(&self.model.config, taps);
        if (!self.prepared) return error.UnpreparedInkling2Taps;
        self.sink = sink;
        defer self.sink = null;
        return prefillFn(ptr, stream);
    }

    fn nextTapGeneration(self: *Metal) u64 {
        self.tap_gen +%= 1;
        return self.tap_gen;
    }

    pub fn backend(self: *Metal) be.Backend {
        return .{
            .keep_full_lone = true,
            .ptr = self,
            .vtable = &.{
                .prefill = prefillFn,
                .prepare_features = prepareFn,
                .features = featuresFn,
                .prefill_features = prefillFeaturesFn,
                .first = firstFn,
                .first_masked = firstMaskedFn,
                .queue = queueFn,
                .read = readFn,
                .verify = verifyFn,
                .keep = keepFn,
                .draft = draftFn,
                .release = releaseFn,
                .first_row = firstRowFn,
                .held = heldFn,
            },
        };
    }

    fn cast(ptr: *anyopaque) *Metal {
        return @ptrCast(@alignCast(ptr));
    }

    fn cacheOf(self: *Metal, stream: *lanes.Stream) !*st.Cache {
        return self.caches.get(stream) orelse error.UnknownInklingStream;
    }

    fn vocab(self: *const Metal) usize {
        return self.model.config.text.unpadded_vocab;
    }

    fn logits(self: *Metal, cache: *st.Cache) []const f32 {
        return cache.logits.slice(f32, self.vocab());
    }

    fn draw(self: *Metal, l: []const f32, sampling: ?lanes.Sampling, position: u64, mask: []const u32) !u32 {
        return draws.draw(self.gpa, l, self.values, sampling, position, mask);
    }

    fn put(self: *Metal, token: u32) u64 {
        const h = self.next;
        self.next += 1;
        self.tokens[h % ring] = token;
        return h;
    }

    fn cacheCost(self: *const Metal, capacity: usize) u64 {
        if (self.model.weight_plan.n_visits == st.layer_slots and self.prompt_rows == st.prompt_rows) {
            return st.cacheBytes(capacity);
        }
        return @as(u64, @intCast(self.model.weight_plan.n_visits)) * (st.globalSlots(capacity) * st.kv_width * 4 + (6 + self.prompt_rows) * st.conv_row_values * 2) + self.vocab() * 4;
    }

    fn feed(self: *Metal, cache: *st.Cache, ids: []const u32, stream: ?*lanes.Stream) !void {
        const base = cache.len;
        var at: usize = 0;
        while (at < ids.len) {
            var end = @min(ids.len, at + self.prompt_rows);
            if (stream) |sm| {
                if (sm.isCancelled()) return error.Cancelled;
                for (sm.chunks) |x| if (x > base + at and x < base + end) {
                    end = x - base;
                };
                for (sm.reuse.marks) |x| if (x > base + at and x < base + end) {
                    end = x - base;
                };
            }
            const s = if (end - at > self.batch_rows) &self.prompt else &self.scratch;
            const capture = self.prepared and stream != null;
            if (capture) _ = self.nextTapGeneration();
            const taps: ?st.Ref = if (capture) .{ .buf = s.taps.? } else null;
            try fwd.run(self.model, s, cache, ids[at..end], .{ .taps = taps });
            if (capture) {
                const rows = end - at;
                if (self.sink) |k| try k.take(k.ptr, stream.?, .{
                    .buffer = @intFromPtr(s.taps.?.id),
                    .offset = 0,
                    .rows = @intCast(rows),
                    .row_bytes = @intCast(st.feature_row_bytes),
                    .space = .device,
                    .dtype = .bf16,
                }, @intCast(cache.len));
                if (base + end == stream.?.prompt().len) {
                    if (cache.last_tap == null) {
                        cache.last_tap = try self.model.device.buffer(st.feature_row_bytes, mtl.ResourceOptions.shared);
                        if (cache.follow) |f| f.join(f.ctx, cache.last_tap.?);
                    }
                    const bytes = s.taps.?.slice(u8, rows * st.feature_row_bytes);
                    @memcpy(cache.last_tap.?.slice(u8, st.feature_row_bytes), bytes[(rows - 1) * st.feature_row_bytes ..][0..st.feature_row_bytes]);
                    cache.tap = .{ .buf = cache.last_tap, .start = @intCast(cache.len + rows - 1), .rows = 1 };
                }
            }
            cache.len += end - at;
            if (stream) |sm| lanes.prefill_widths.completed(self.gpa, sm, end - at);
            at = end;
            if (stream) |sm| if (std.mem.indexOfScalar(u32, sm.reuse.marks, @intCast(cache.len)) != null) if (sm.reuse.hook) |k| k.at(k.ptr, sm, @intCast(cache.len));
        }
    }

    fn prefillFn(ptr: *anyopaque, stream: *lanes.Stream) !void {
        const self = cast(ptr);
        const ids = stream.prompt();
        if (ids.len == 0 or ids.len + stream.max_new > self.options.capacity or self.caches.contains(stream)) return error.InvalidInklingPrompt;
        const need = self.cacheCost(ids.len + stream.max_new);
        if (self.held + need > self.options.budget) {
            std.log.warn("inkling2: a stream needs {d:.1} GiB of state; {d:.1} of {d:.1} GiB are held", .{ gib(need), gib(self.held), gib(self.options.budget) });
            return error.InklingStateFull;
        }
        self.held += need;
        errdefer self.held -= need;
        if (self.room) |r| r.fit(r.ctx, self.options.budget - self.held, stream.reuse.saved, ids);
        const cache = try self.createCache(ids.len + stream.max_new);
        errdefer self.gpa.destroy(cache);
        errdefer cache.deinit();
        try self.caches.put(self.gpa, stream, cache);
        errdefer _ = self.caches.remove(stream);
        stream.cached = 0;
        if (stream.reuse.saved) |saved| {
            const snap: *const SavedSnap = @ptrCast(@alignCast(saved));
            if (snap.target.at < ids.len) {
                if (self.restoreSnap(cache, stream, snap)) {
                    stream.cached = @intCast(snap.target.at);
                } else |_| {
                    cache.len = 0;
                    if (self.drafter) |d| d.resetLane(stream);
                    stream.reuse_failed = true;
                }
            } else stream.reuse_failed = true;
        }
        try self.feed(cache, ids[cache.len..], stream);
    }

    fn restoreSnap(self: *Metal, cache: *st.Cache, stream: *lanes.Stream, snap: *const SavedSnap) !void {
        if (self.drafter) |d| try d.restoreSnap(stream, if (snap.dspark) |*ds| ds else null, snap.target.at);
        try snapshot.restore(cache, self.model.queue, &snap.target);
    }

    /// Times 1..window_rows (if not yet timed) and 4-stream shared batches up to batch_rows.
    pub fn measure(self: *Metal) !void {
        const ids = try self.gpa.alloc(u32, self.batch_rows);
        defer self.gpa.free(ids);
        for (ids, 0..) |*x, i| x.* = @intCast((i * 7919 + 1) % self.vocab());
        const cap: usize = if (self.options.streams > 1) 4 else 1;
        var caches: [4]st.Cache = undefined;
        var made: usize = 0;
        defer for (caches[0..made]) |*c| c.deinit();
        for (caches[0..cap]) |*c| {
            c.* = try st.Cache.initFor(self.model.device, 64, &self.model.weight_plan, self.prompt_rows, self.vocab());
            made += 1;
        }
        if (self.timed == 0) {
            const max_w = @min(@min(st.window_rows, self.batch_rows), self.prompt_rows);
            for (1..max_w + 1) |rows| {
                var ms: f64 = 0;
                for (0..2) |attempt| {
                    caches[0].len = 0;
                    const t0 = now();
                    try fwd.run(self.model, &self.scratch, &caches[0], ids[0..rows], .{ .all = true });
                    if (attempt == 1) ms = now() - t0;
                }
                self.costs[rows - 1] = .{ .width = @intCast(rows), .ms = ms };
            }
            self.timed = max_w;
        }
        self.shared_timed = 0;
        if (self.options.streams > 1) {
            for (shared_widths) |rows| {
                if (rows > self.batch_rows) continue;
                const seg_rows = rows / 4;
                const segs = [_]fwd.Segment{
                    .{ .cache = &caches[0], .rows = seg_rows },
                    .{ .cache = &caches[1], .rows = seg_rows },
                    .{ .cache = &caches[2], .rows = seg_rows },
                    .{ .cache = &caches[3], .rows = seg_rows },
                };
                var ms: f64 = 0;
                for (0..2) |attempt| {
                    for (caches[0..4]) |*c| c.len = 0;
                    const t0 = now();
                    try fwd.runShared(self.model, &self.scratch, &segs, ids[0..rows], .{ .all = true });
                    if (attempt == 1) ms = now() - t0;
                }
                self.shared[self.shared_timed] = .{ .width = @intCast(rows), .ms = ms };
                self.shared_timed += 1;
            }
        }
    }

    pub fn warm(self: *Metal) !void {
        const cap = self.prompt_rows + st.window_rows;
        var cache = try st.Cache.initFor(self.model.device, cap, &self.model.weight_plan, self.prompt_rows, self.vocab());
        defer cache.deinit();
        const ids = try self.gpa.alloc(u32, self.prompt_rows);
        defer self.gpa.free(ids);
        for (ids, 0..) |*t, i| t.* = @intCast((i * 37 + 1) % self.vocab());
        try self.feed(&cache, ids, null);
        const max_w = @min(@min(st.window_rows, self.batch_rows), self.prompt_rows);
        try fwd.run(self.model, &self.scratch, &cache, ids[0..max_w], .{ .all = true });
        for (1..max_w + 1) |rows| {
            var ms: f64 = 0;
            for (0..2) |attempt| {
                const t0 = now();
                try fwd.run(self.model, &self.scratch, &cache, ids[0..rows], .{ .all = true });
                if (attempt == 1) ms = now() - t0;
            }
            self.costs[rows - 1] = .{ .width = @intCast(rows), .ms = ms };
        }
        self.timed = max_w;
    }

    fn now() f64 {
        if (comptime !builtin.os.tag.isDarwin()) return 0;
        var ts: std.c.timespec = undefined;
        _ = std.c.clock_gettime(.MONOTONIC, &ts);
        return @as(f64, @floatFromInt(ts.sec)) * 1000.0 + @as(f64, @floatFromInt(ts.nsec)) * 1e-6;
    }

    fn firstFn(ptr: *anyopaque, stream: *lanes.Stream, position: u64) !u64 {
        return cast(ptr).firstSlot(stream, position, &.{});
    }

    fn firstMaskedFn(ptr: *anyopaque, stream: *lanes.Stream, position: u64, mask: []const u32) !u64 {
        return cast(ptr).firstSlot(stream, position, mask);
    }

    fn firstSlot(self: *Metal, stream: *lanes.Stream, position: u64, mask: []const u32) !u64 {
        const cache = try self.cacheOf(stream);
        if (position != cache.len or cache.pending != null) return error.InvalidInklingPosition;
        return self.put(try self.draw(self.logits(cache), stream.sampling, position, mask));
    }

    fn queueFn(ptr: *anyopaque, stream: *lanes.Stream, what: be.Feed, position: u64) !u64 {
        const self = cast(ptr);
        const cache = try self.cacheOf(stream);
        if (position != cache.len + 1 or cache.pending != null) return error.InvalidInklingPosition;
        const token = switch (what) {
            .handle => |h| try self.read(h),
            .value => |v| v,
        };
        if (token >= self.vocab()) return error.InvalidInklingToken;
        const generation = if (self.prepared) self.nextTapGeneration() else 0;
        const taps: ?st.Ref = if (self.prepared) .{ .buf = self.scratch.taps.? } else null;
        try fwd.run(self.model, &self.scratch, cache, &.{token}, .{ .taps = taps });
        if (self.prepared) cache.tap = .{ .buf = self.scratch.taps, .start = @intCast(cache.len), .rows = 1, .gen = generation };
        cache.len += 1;
        return self.put(try self.draw(self.logits(cache), stream.sampling, position, &.{}));
    }

    fn read(self: *Metal, handle: u64) !u32 {
        const token = try ringToken(self.tokens, self.next, handle);
        if (token >= self.vocab()) return error.InvalidInklingToken;
        return token;
    }

    fn readFn(ptr: *anyopaque, handle: u64) !u32 {
        return cast(ptr).read(handle);
    }

    fn firstRowFn(ptr: *anyopaque, stream: *lanes.Stream) !lanes.LogprobRow {
        const self = cast(ptr);
        const l = self.logits(try self.cacheOf(stream));
        std.debug.assert(self.next > 0);
        return draws.logprobRow(l, self.tokens[(self.next - 1) % ring], stream.logprobs orelse 0);
    }

    fn verifyFn(ptr: *anyopaque, windows: []const be.Window, out: []be.Verified) !void {
        const self = cast(ptr);
        const words = (@as(usize, self.model.config.text.vocab) + 31) / 32;
        if (windows.len != out.len or windows.len > self.batch_rows) return error.InvalidInklingWindows;
        var segments: [st.batch_rows]fwd.Segment = undefined;
        var ids: [st.batch_rows]u32 = undefined;
        var total: usize = 0;
        var begun: usize = 0;
        errdefer for (segments[0..begun]) |seg| seg.cache.rollback(self.model.queue) catch {};
        for (windows, out, 0..) |w, o, i| {
            const rows = w.rows();
            if (rows == 0 or rows > st.window_rows or total + rows > self.batch_rows or w.parents != null or w.held != 0 or o.sampled.len != rows or o.drafts.len != rows - 1 or w.positions.len != rows) return error.UnsupportedInklingWindow;
            if (w.masks.len != 0 and w.masks.len != rows * words) return error.InvalidInklingMasks;
            const cache = try self.cacheOf(w.stream);
            if (cache.pending != null) return error.InklingWindowAlreadyOpen;
            for (w.positions, 0..) |p, r| if (p != cache.len + r + 1) return error.InvalidInklingPosition;
            segments[i] = .{ .cache = cache, .rows = rows };
            ids[total] = w.pending;
            @memcpy(ids[total + 1 ..][0..w.tokens.len], w.tokens);
            total += rows;
        }
        for (segments[0..windows.len]) |seg| {
            try seg.cache.begin(self.model.queue, seg.rows);
            begun += 1;
        }
        const generation = if (self.prepared) self.nextTapGeneration() else 0;
        const taps: ?st.Ref = if (self.prepared) .{ .buf = self.scratch.taps.? } else null;
        try fwd.runShared(self.model, &self.scratch, segments[0..windows.len], ids[0..total], .{ .all = true, .taps = taps });
        const all = self.scratch.logits_all.slice(f32, total * self.vocab());
        var base: usize = 0;
        for (windows, out, segments[0..windows.len]) |w, o, seg| {
            if (self.prepared) seg.cache.tap = .{ .buf = self.scratch.taps, .off = base * st.feature_row_bytes, .start = @intCast(seg.cache.len), .rows = seg.rows, .gen = generation };
            @memcpy(o.drafts, w.tokens);
            for (0..seg.rows) |r| {
                const l = all[(base + r) * self.vocab() ..][0..self.vocab()];
                const mask = if (w.masks.len == 0) &.{} else w.masks[r * words ..][0..words];
                o.sampled[r] = try self.draw(l, w.stream.sampling, w.positions[r], mask);
                if (o.rows.len == seg.rows) o.rows[r] = draws.logprobRow(l, o.sampled[r], w.stream.logprobs orelse 0);
            }
            base += seg.rows;
        }
    }

    fn keepFn(ptr: *anyopaque, windows: []const be.Window, paths: []const []const u32) !void {
        const self = cast(ptr);
        if (windows.len != paths.len) return error.InvalidInklingWindows;
        for (windows, paths) |w, path| {
            const cache = try self.cacheOf(w.stream);
            for (path, 0..) |r, i| if (r != i) return error.UnsupportedInklingTree;
            if (path.len == 0 or path.len > w.rows() or cache.pending == null or cache.pending.? != w.rows()) return error.UnsupportedInklingWindow;
            try cache.commit(self.model.queue, path.len);
        }
    }

    fn draftFn(_: *anyopaque, requests: []const be.DraftRequest) !void {
        for (requests) |r| if (r.depth != 0 or r.lanes != null or r.plan != null) return error.InklingHasNoDraftHead;
    }

    fn heldFn(_: *anyopaque, _: *lanes.Stream, out: []u32) !void {
        if (out.len > 0) return error.InklingDraftsNotHeld;
    }

    fn releaseFn(ptr: *anyopaque, stream: *lanes.Stream) void {
        const self = cast(ptr);
        if (self.caches.fetchRemove(stream)) |entry| {
            const cache = entry.value;
            if (cache.pending != null) cache.rollback(self.model.queue) catch |err| std.log.err("inkling2 rollback at release: {s}", .{@errorName(err)});
            self.held -= self.cacheCost(cache.capacity);
            cache.deinit();
            self.gpa.destroy(cache);
            if (self.room) |r| r.fit(r.ctx, self.options.budget - self.held, null, &.{});
        }
    }

    pub fn snapBytes(ptr: *anyopaque, at: u32) u64 {
        const self = cast(ptr);
        const draft: u64 = if (self.drafter != null) small_dspark_state.Snap.bytes(at) else 0;
        return snapshot.bytes(at) + draft;
    }

    pub fn snapSave(ptr: *anyopaque, owner: ?*anyopaque, at: u32) anyerror!*anyopaque {
        const self = cast(ptr);
        const stream: *lanes.Stream = @ptrCast(@alignCast(owner orelse return error.NoStream));
        const cache = try self.cacheOf(stream);
        if (cache.len != at or cache.pending != null) return error.SnapshotOutOfStep;
        var ds = if (self.drafter) |d| try d.saveSnap(stream, at, cache.tap, self.tap_gen) else null;
        errdefer if (ds) |*s| s.free();
        var target = try snapshot.save(self.model.device, self.model.queue, cache);
        errdefer target.free();
        const snap = try self.gpa.create(SavedSnap);
        snap.* = .{ .target = target, .dspark = ds };
        return snap;
    }

    pub fn snapRestore(_: *anyopaque, _: ?*anyopaque, _: *anyopaque) anyerror!void {
        return error.BackendRestores;
    }

    pub fn snapDrop(ptr: *anyopaque, saved: *anyopaque) void {
        const snap: *SavedSnap = @ptrCast(@alignCast(saved));
        if (snap.dspark) |*ds| ds.free();
        snap.target.free();
        cast(ptr).gpa.destroy(snap);
    }
};

fn gib(bytes: u64) f64 {
    return @as(f64, @floatFromInt(bytes)) / (1 << 30);
}

fn ringToken(tokens: []const u32, next: u64, handle: u64) !u32 {
    if (handle >= next or next - handle > ring) return error.InvalidInklingTokenHandle;
    return tokens[handle % ring];
}
