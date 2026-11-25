//! Inkling 2 Mini behind the lane engine: independent visit state and exact copy-window rollback.
const std = @import("std");
const mtl = @import("metal");
const lanes = @import("lanes");
const st = @import("state.zig");
const fwd = @import("forward.zig");
const Model = @import("model.zig").Model;
const snapshot = @import("snapshot.zig");
const img = @import("images.zig");
const dspark = @import("dspark.zig");
const dspark_state = @import("dspark_state.zig");
const cfg = @import("../config.zig");
const be = lanes.backend;

const SavedSnap = struct {
    target: snapshot.Snap,
    dspark: ?dspark_state.Snap = null,
};

pub const Options = struct { capacity: usize, streams: u32 = 4, budget: u64, images: ?img.Encoder = null };
const ring = 4096;

/// The prompt cache gets the bytes left by live streams and never evicts the state being restored.
pub const Room = struct { ctx: *anyopaque, fit: *const fn (ctx: *anyopaque, bytes: u64, restoring: ?*anyopaque, prompt: []const u32) void };

/// Mini's DSpark ABI is fixed: four visit taps in checkpoint order.
pub fn validateTaps(c: *const cfg.Config, taps: []const u32) !void {
    const mtp = c.mtp orelse return error.UnsupportedInkling2Taps;
    const expected = [_]u32{ 5, 23, 41, 48 };
    if (mtp.taps_n != expected.len or !std.mem.eql(u32, mtp.taps[0..mtp.taps_n], &expected) or !std.mem.eql(u32, taps, &expected))
        return error.UnsupportedInkling2Taps;
}

pub const Metal = struct {
    gpa: std.mem.Allocator,
    model: *Model,
    options: Options,
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
    tap_gen: u64 = 0,
    drafter: ?*dspark.Drafter = null,

    pub fn init(gpa: std.mem.Allocator, model: *Model, options: Options) !*Metal {
        const out = try gpa.create(Metal);
        errdefer gpa.destroy(out);
        var scratch = try st.Scratch.init(model.device, st.batch_rows);
        errdefer scratch.deinit();
        var prompt = try st.Scratch.init(model.device, st.prompt_rows);
        errdefer prompt.deinit();
        const tokens = try gpa.alloc(u32, ring);
        errdefer gpa.free(tokens);
        const values = try gpa.alloc(f32, model.config.text.unpadded_vocab);
        out.* = .{ .gpa = gpa, .model = model, .options = options, .scratch = scratch, .prompt = prompt, .tokens = tokens, .values = values };
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
        cache.* = try st.Cache.init(self.model.device, capacity, &self.model.weight_plan);
        return cache;
    }

    pub fn facts(self: *const Metal) lanes.Model {
        return .{ .exact_width = st.window_rows, .gpu_tokens = false, .hidden_rows = true, .batch_rows = st.batch_rows, .mtp = false, .speculate = false, .speculate_early = false, .drafts = 1, .max_streams = self.options.streams, .streams_exact = true };
    }

    fn prepareFn(ptr: *anyopaque, taps: []const u32) !void {
        const self = cast(ptr);
        try validateTaps(&self.model.config, taps);
        if (self.prepared) return;
        const device = self.model.device;
        const decode = self.scratch.taps orelse try device.buffer(st.captureBytes(st.batch_rows), mtl.ResourceOptions.shared);
        errdefer if (self.scratch.taps == null) decode.deinit();
        const prompt = self.prompt.taps orelse try device.buffer(st.captureBytes(st.prompt_rows), mtl.ResourceOptions.shared);
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
        return .{ .keep_full_lone = true, .ptr = self, .vtable = &.{ .prefill = prefillFn, .prepare_features = prepareFn, .features = featuresFn, .prefill_features = prefillFeaturesFn, .first = firstFn, .first_masked = firstMaskedFn, .queue = queueFn, .read = readFn, .verify = verifyFn, .keep = keepFn, .draft = draftFn, .release = releaseFn, .first_row = firstRowFn, .held = heldFn } };
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
        const t = sampled(sampling) orelse {
            var best: ?usize = null;
            for (l, 0..) |w, id| {
                if (mask.len > 0 and !lanes.grammar.allows(mask, id)) continue;
                if (best == null or w > l[best.?]) best = id;
            }
            return @intCast(best orelse return error.InklingEmptyTokenMask);
        };
        const values = if (mask.len == 0) l else blk: {
            for (l, self.values, 0..) |w, *v, id| v.* = if (lanes.grammar.allows(mask, id)) w else -std.math.inf(f32);
            break :blk self.values;
        };
        return lanes.gpu_full.draw(self.gpa, values, t, @intCast(position), null);
    }

    fn sampled(sampling: ?lanes.Sampling) ?lanes.Sampling {
        const t = sampling orelse return null;
        return if (t.temperature > 0) t else null;
    }

    fn take(self: *Metal) u64 {
        const h = self.next;
        self.next += 1;
        return h;
    }

    fn put(self: *Metal, token: u32) u64 {
        const h = self.take();
        self.tokens[h % ring] = token;
        return h;
    }

    fn feed(self: *Metal, cache: *st.Cache, ids: []const u32, stream: ?*lanes.Stream, pictures: ?*img.Rows) !void {
        const base = cache.len;
        var at: usize = 0;
        while (at < ids.len) {
            var end = @min(ids.len, at + st.prompt_rows);
            if (stream) |sm| {
                if (sm.isCancelled()) return error.Cancelled;
                for (sm.chunks) |x| if (x > base + at and x < base + end) {
                    end = x - base;
                };
                for (sm.reuse.marks) |x| if (x > base + at and x < base + end) {
                    end = x - base;
                };
            }
            const s = if (end - at > st.batch_rows) &self.prompt else &self.scratch;
            const capture = self.prepared and stream != null;
            if (capture) _ = self.nextTapGeneration();
            const taps: ?st.Ref = if (capture) .{ .buf = s.taps.? } else null;
            const rows_of: []const fwd.ImageRows = if (pictures) |p| p.chunk(base + at, base + end) else &.{};
            try fwd.run(self.model, s, cache, ids[at..end], .{ .taps = taps, .images = rows_of });
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
        const need = st.cacheBytes(ids.len + stream.max_new);
        if (self.held + need > self.options.budget) {
            std.log.warn("inkling2_loop: a stream needs {d:.1} GiB of state; {d:.1} of {d:.1} GiB are held", .{ gib(need), gib(self.held), gib(self.options.budget) });
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
        var pictures = try img.Rows.init(self.gpa, self.options.images, stream, cache.len);
        defer pictures.deinit();
        try self.feed(cache, ids[cache.len..], stream, &pictures);
    }

    fn restoreSnap(self: *Metal, cache: *st.Cache, stream: *lanes.Stream, snap: *const SavedSnap) !void {
        if (self.drafter) |d| try d.restoreSnap(stream, if (snap.dspark) |*ds| ds else null, snap.target.at);
        try snapshot.restore(cache, self.model.queue, &snap.target);
    }

    pub fn warm(self: *Metal) !void {
        var cache = try st.Cache.init(self.model.device, st.prompt_rows + st.window_rows, &self.model.weight_plan);
        defer cache.deinit();
        var ids: [st.prompt_rows]u32 = undefined;
        for (&ids, 0..) |*t, i| t.* = @intCast(1000 + i * 37 % 20000);
        try self.feed(&cache, &ids, null, null);
        try fwd.run(self.model, &self.scratch, &cache, ids[0..st.window_rows], .{ .all = true });
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
        return firstLogprobRow(l, self.tokens, self.next, stream.logprobs orelse 0);
    }

    fn verifyFn(ptr: *anyopaque, windows: []const be.Window, out: []be.Verified) !void {
        const self = cast(ptr);
        const words = maskWords(self.model.config.text.vocab);
        if (windows.len != out.len or windows.len > st.batch_rows) return error.InvalidInklingWindows;
        var segments: [st.batch_rows]fwd.Segment = undefined;
        var ids: [st.batch_rows]u32 = undefined;
        var total: usize = 0;
        var begun: usize = 0;
        errdefer for (segments[0..begun]) |seg| seg.cache.rollback(self.model.queue) catch {};
        for (windows, out, 0..) |w, o, i| {
            const rows = w.rows();
            if (rows == 0 or rows > st.window_rows or total + rows > st.batch_rows or w.parents != null or w.held != 0 or o.sampled.len != rows or o.drafts.len != rows - 1 or w.positions.len != rows) return error.UnsupportedInklingWindow;
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
                const mask = maskRow(w.masks, words, r);
                o.sampled[r] = try self.draw(l, w.stream.sampling, w.positions[r], mask);
                if (o.rows.len == seg.rows) o.rows[r] = logprobRow(l, o.sampled[r], w.stream.logprobs orelse 0);
            }
            @memcpy(seg.cache.logits.slice(f32, self.vocab()), all[(base + seg.rows - 1) * self.vocab() ..][0..self.vocab()]);
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

    fn draftFn(ptr: *anyopaque, requests: []const be.DraftRequest) !void {
        _ = ptr;
        for (requests) |r| if (r.depth != 0 or r.lanes != null or r.plan != null) return error.InklingHasNoDraftHead;
    }

    fn heldFn(_: *anyopaque, _: *lanes.Stream, out: []u32) !void {
        if (out.len > 0) return error.InklingDraftsNotHeld;
    }

    fn releaseFn(ptr: *anyopaque, stream: *lanes.Stream) void {
        const self = cast(ptr);
        if (self.caches.fetchRemove(stream)) |entry| {
            const cache = entry.value;
            if (cache.pending != null) cache.rollback(self.model.queue) catch |err| std.log.err("inkling2_loop rollback at release: {s}", .{@errorName(err)});
            self.held -= st.cacheBytes(cache.capacity);
            cache.deinit();
            self.gpa.destroy(cache);
            if (self.room) |r| r.fit(r.ctx, self.options.budget - self.held, null, &.{});
        }
    }

    pub fn snapBytes(ptr: *anyopaque, at: u32) u64 {
        const self = cast(ptr);
        return snapshot.bytes(at) + if (self.drafter != null) dspark_state.Snap.bytes(at) else 0;
    }

    pub fn snapSave(ptr: *anyopaque, owner: ?*anyopaque, at: u32) anyerror!*anyopaque {
        const self = cast(ptr);
        const stream: *lanes.Stream = @ptrCast(@alignCast(owner orelse return error.NoStream));
        const cache = try self.cacheOf(stream);
        if (cache.len != at or cache.pending != null) return error.SnapshotOutOfStep;
        const snap = try self.gpa.create(SavedSnap);
        errdefer self.gpa.destroy(snap);
        snap.* = .{ .target = try snapshot.save(self.model.device, self.model.queue, cache) };
        errdefer snap.target.free();
        if (self.drafter) |d| {
            snap.dspark = try d.saveSnap(stream, at, cache.tap, self.tap_gen);
        }
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

/// A grammar mask's words a row: a bit a token of the tokenizer's vocabulary, padding included.
fn maskWords(vocab: u32) usize {
    return (@as(usize, vocab) + 31) / 32;
}

/// Row `r` of a window's masks, or none where the window has none.
fn maskRow(masks: []const u32, words: usize, r: usize) []const u32 {
    return if (masks.len == 0) &.{} else masks[r * words ..][0..words];
}

test "a window's masks step by the padded vocabulary's words" {
    const words = maskWords(201024);
    try std.testing.expectEqual(@as(usize, 6282), words);
    var masks: [3 * 6282]u32 = @splat(0);
    masks[2 * 6282] = 1;
    try std.testing.expectEqual(@as(usize, 0), maskRow(&.{}, words, 2).len);
    try std.testing.expectEqual(@as(u32, 0), maskRow(&masks, words, 1)[0]);
    try std.testing.expect(lanes.grammar.allows(maskRow(&masks, words, 2), 0));
    try std.testing.expect(!lanes.grammar.allows(maskRow(&masks, words, 1), 0));
}

fn gib(bytes: u64) f64 {
    return @as(f64, @floatFromInt(bytes)) / (1 << 30);
}

/// The token behind `handle`: only the last `ring` handles drawn are still held.
fn ringToken(tokens: []const u32, next: u64, handle: u64) !u32 {
    if (handle >= next or next - handle > ring) return error.InvalidInklingTokenHandle;
    return tokens[handle % ring];
}

fn firstLogprobRow(l: []const f32, tokens: []const u32, next: usize, count: u8) lanes.LogprobRow {
    std.debug.assert(tokens.len == ring and next > 0);
    return logprobRow(l, tokens[(next - 1) % ring], count);
}

fn logprobRow(l: []const f32, pick: u32, count: u8) lanes.LogprobRow {
    const top = lanes.logprob.max_top;
    var most: f64 = -std.math.inf(f64);
    for (l) |w| most = @max(most, w);
    var sum: f64 = 0;
    for (l) |w| sum += @exp(@as(f64, w) - most);
    const lse: f32 = @floatCast(most + @log(sum));
    var r: lanes.LogprobRow = .{ .token = pick, .logprob = l[pick] - lse, .count = 0 };
    var filled: usize = 0;
    for (l, 0..) |w, id| {
        if (filled == top and w <= r.logprobs[top - 1] + lse) continue;
        var at = @min(filled, top - 1);
        while (at > 0 and w > r.logprobs[at - 1] + lse) : (at -= 1) {
            r.ids[at] = r.ids[at - 1];
            r.logprobs[at] = r.logprobs[at - 1];
        }
        r.ids[at] = @intCast(id);
        r.logprobs[at] = w - lse;
        filled = @min(filled + 1, top);
    }
    r.count = @min(count, top);
    return r;
}

test "sampled first-token logprob uses the last written handle, not argmax" {
    const logits = [_]f32{ 0, -1, -2, -3 };
    var tokens: [ring]u32 = @splat(0);
    tokens[2] = 3;
    const row = firstLogprobRow(&logits, &tokens, 3, 0);
    try std.testing.expectEqual(@as(u32, 3), row.token);
    try std.testing.expect(std.math.isFinite(row.logprob));
    const lse = @log(@as(f64, 1) + @exp(-@as(f64, 1)) + @exp(-@as(f64, 2)) + @exp(-@as(f64, 3)));
    try std.testing.expectApproxEqAbs(@as(f32, -3) - @as(f32, @floatCast(lse)), row.logprob, 1e-6);
}

test "a token handle expires once the ring has wrapped past it" {
    var tokens: [ring]u32 = @splat(0);
    tokens[0] = 7;
    try std.testing.expectEqual(@as(u32, 7), try ringToken(&tokens, ring, 0));
    try std.testing.expectError(error.InvalidInklingTokenHandle, ringToken(&tokens, ring + 1, 0));
    try std.testing.expectEqual(@as(u32, 7), try ringToken(&tokens, ring + 1, ring));
    try std.testing.expectError(error.InvalidInklingTokenHandle, ringToken(&tokens, ring, ring));
}
