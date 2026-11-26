//! Inkling 2 Mini behind the lane engine: independent visit state and exact copy-window rollback.
const std = @import("std");
const builtin = @import("builtin");
const mtl = @import("metal");
const lanes = @import("lanes");
const st = @import("state.zig");
const fwd = @import("forward.zig");
const fwd_ops = @import("forward_ops.zig");
const Model = @import("model.zig").Model;
const snapshot = @import("snapshot.zig");
const img = @import("images.zig");
const dspark = @import("dspark.zig");
const dspark_state = @import("dspark_state.zig");
const cfg = @import("../config.zig");
const draws = @import("draw.zig");
const warm_ops = @import("backend_warm.zig");
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
    /// A forward's ms by window width, timed by `warm` / `measure`: the depth rule prices rounds with them.
    costs: [st.window_rows]lanes.config.Cost = undefined,
    timed: usize = 0,
    shared: [warm_ops.shared_widths.len]lanes.config.Cost = undefined,
    shared_timed: usize = 0,
    tap_gen: u64 = 0,
    drafter: ?*dspark.Drafter = null,
    verify_gen: u64 = 0,
    commit_cb: ?mtl.CommandBuffer = null,
    commit_seq: u64 = 0,
    commit_active_seq: u64 = 0,
    test_fail_commit: if (builtin.is_test) bool else void = if (builtin.is_test) false else {},

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
        self.settleCommit() catch {};
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
        return .{ .exact_width = st.window_rows, .gpu_tokens = false, .hidden_rows = true, .batch_rows = st.batch_rows, .mtp = false, .speculate = false, .speculate_early = false, .drafts = 1, .max_streams = self.options.streams, .streams_exact = true, .window_costs = self.costs[0..self.timed], .shared_costs = self.shared[0..self.shared_timed] };
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
        if (self.commit_cb) |cb| if (cb.status() == .@"error") try self.settleCommit();
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

    pub fn nextVerifyGeneration(self: *Metal) u64 {
        self.verify_gen +%= 1;
        return self.verify_gen;
    }

    pub fn backend(self: *Metal) be.Backend {
        return .{ .keep_full_lone = true, .ptr = self, .vtable = &.{ .prefill = prefillFn, .prepare_features = prepareFn, .features = featuresFn, .prefill_features = prefillFeaturesFn, .first = firstFn, .first_masked = firstMaskedFn, .queue = queueFn, .read = readFn, .verify = verifyFn, .keep = keepFn, .draft = draftFn, .release = releaseFn, .first_row = firstRowFn, .held = heldFn } };
    }

    fn cast(ptr: *anyopaque) *Metal {
        return @ptrCast(@alignCast(ptr));
    }

    fn cacheOf(self: *Metal, stream: *lanes.Stream) !*st.Cache {
        const cache = self.caches.get(stream) orelse return error.UnknownInklingStream;
        try cache.ensureUsable();
        return cache;
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

    pub fn feed(self: *Metal, cache: *st.Cache, ids: []const u32, stream: ?*lanes.Stream, pictures: ?*img.Rows) !void {
        try self.settleCommit();
        try cache.ensureUsable();
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
            if (s == &self.scratch) _ = self.nextVerifyGeneration();
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

    fn nextCommitSeq(self: *Metal) u64 {
        self.commit_seq +%= 1;
        if (self.commit_seq == 0) self.commit_seq = 1;
        return self.commit_seq;
    }

    fn latchCommitFailure(self: *Metal, seq: u64) void {
        if (seq == 0) return;
        var it = self.caches.valueIterator();
        while (it.next()) |cache_ptr| {
            const cache = cache_ptr.*;
            if (cache.commit_seq == seq) cache.markCommitFailed();
        }
    }

    pub fn settleCommit(self: *Metal) !void {
        const seq = self.commit_active_seq;
        const cb = self.commit_cb;
        self.commit_cb = null;
        self.commit_active_seq = 0;
        const gpu_err = if (cb) |buf| blk: {
            defer buf.release();
            buf.wait();
            break :blk buf.failure();
        } else null;
        const injected = if (comptime builtin.is_test) blk: {
            if (seq == 0) break :blk false;
            const fail = self.test_fail_commit;
            self.test_fail_commit = false;
            break :blk fail;
        } else false;
        if (gpu_err != null or injected) {
            if (gpu_err) |msg| std.log.err("inkling2_loop keep commit: {s}", .{msg});
            self.latchCommitFailure(seq);
            return error.GpuFailed;
        }
    }

    fn prefillFn(ptr: *anyopaque, stream: *lanes.Stream) !void {
        const self = cast(ptr);
        try self.settleCommit();
        const ids = stream.prompt();
        if (self.caches.get(stream)) |existing| return if (existing.commit_failed) error.GpuFailed else error.InvalidInklingPrompt;
        if (ids.len == 0 or ids.len + stream.max_new > self.options.capacity) return error.InvalidInklingPrompt;
        const min_cap = ids.len + stream.max_new;
        const padded = @min(@as(usize, 1048576), min_cap + st.window_rows);
        const cap = if (self.held + st.cacheBytes(padded) <= self.options.budget) padded else min_cap;
        const need = st.cacheBytes(cap);
        if (self.held + need > self.options.budget) {
            std.log.warn("inkling2_loop: a stream needs {d:.1} GiB of state; {d:.1} of {d:.1} GiB are held", .{ gib(need), gib(self.held), gib(self.options.budget) });
            return error.InklingStateFull;
        }
        self.held += need;
        errdefer self.held -= need;
        if (self.room) |r| r.fit(r.ctx, self.options.budget - self.held, stream.reuse.saved, ids);
        const cache = try self.createCache(cap);
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
        return warm_ops.warm(self);
    }

    /// Times 1..window_rows if not timed yet and 4-stream shared batches up to batch_rows.
    pub fn measure(self: *Metal) !void {
        return warm_ops.measure(self);
    }

    fn firstFn(ptr: *anyopaque, stream: *lanes.Stream, position: u64) !u64 {
        return cast(ptr).firstSlot(stream, position, &.{});
    }

    fn firstMaskedFn(ptr: *anyopaque, stream: *lanes.Stream, position: u64, mask: []const u32) !u64 {
        return cast(ptr).firstSlot(stream, position, mask);
    }

    fn firstSlot(self: *Metal, stream: *lanes.Stream, position: u64, mask: []const u32) !u64 {
        const cache = try self.cacheOf(stream);
        if (self.commit_active_seq != 0 and cache.commit_seq == self.commit_active_seq) try self.settleCommit();
        if (position != cache.len or cache.pending != null) return error.InvalidInklingPosition;
        return self.put(try self.draw(self.logits(cache), stream.sampling, position, mask));
    }

    fn queueFn(ptr: *anyopaque, stream: *lanes.Stream, what: be.Feed, position: u64) !u64 {
        const self = cast(ptr);
        try self.settleCommit();
        const cache = try self.cacheOf(stream);
        if (position != cache.len + 1 or cache.pending != null) return error.InvalidInklingPosition;
        const token = switch (what) {
            .handle => |h| try self.read(h),
            .value => |v| v,
        };
        if (token >= self.vocab()) return error.InvalidInklingToken;
        const generation = if (self.prepared) self.nextTapGeneration() else 0;
        const taps: ?st.Ref = if (self.prepared) .{ .buf = self.scratch.taps.? } else null;
        _ = self.nextVerifyGeneration();
        try fwd.run(self.model, &self.scratch, cache, &.{token}, .{ .taps = taps });
        if (self.prepared) cache.tap = .{ .buf = self.scratch.taps, .start = @intCast(cache.len), .rows = 1, .gen = generation };
        cache.len += 1;
        return self.put(try self.draw(self.logits(cache), stream.sampling, position, &.{}));
    }

    fn read(self: *Metal, handle: u64) !u32 {
        const token = try draws.ringToken(self.tokens, self.next, handle, ring);
        if (token >= self.vocab()) return error.InvalidInklingToken;
        return token;
    }

    fn readFn(ptr: *anyopaque, handle: u64) !u32 {
        return cast(ptr).read(handle);
    }

    fn firstRowFn(ptr: *anyopaque, stream: *lanes.Stream) !lanes.LogprobRow {
        const self = cast(ptr);
        const cache = try self.cacheOf(stream);
        if (self.commit_active_seq != 0 and cache.commit_seq == self.commit_active_seq) try self.settleCommit();
        const l = self.logits(cache);
        return draws.firstLogprobRow(l, self.tokens, self.next, stream.logprobs orelse 0, ring);
    }

    fn verifyFn(ptr: *anyopaque, windows: []const be.Window, out: []be.Verified) !void {
        const self = cast(ptr);
        try self.settleCommit();
        const words = draws.maskWords(self.model.config.text.vocab);
        if (windows.len != out.len or windows.len > st.batch_rows) return error.InvalidInklingWindows;
        var segments: [st.batch_rows]fwd.Segment = undefined;
        var ids: [st.batch_rows]u32 = undefined;
        var depths: [st.batch_rows]u32 = undefined;
        var total: usize = 0;
        var begun: usize = 0;
        errdefer for (segments[0..begun]) |seg| seg.cache.rollback() catch {};
        for (windows, out, 0..) |w, o, i| {
            const rows = w.rows();
            if (rows == 0 or rows > st.window_rows or total + rows > st.batch_rows or w.held != 0 or o.sampled.len != rows or o.drafts.len != rows - 1 or w.positions.len != rows) return error.UnsupportedInklingWindow;
            if (w.masks.len != 0 and w.masks.len != rows * words) return error.InvalidInklingMasks;
            const cache = try self.cacheOf(w.stream);
            if (cache.pending != null) return error.InklingWindowAlreadyOpen;
            if (w.parents) |ps| {
                if (ps.len != rows) return error.UnsupportedInklingWindow;
                try st.validateParents(ps, depths[total..][0..rows]);
                for (w.positions, depths[total..][0..rows]) |p, d| if (p != cache.len + d + 1) return error.InvalidInklingPosition;
            } else {
                for (w.positions, 0..) |p, r| if (p != cache.len + r + 1) return error.InvalidInklingPosition;
            }
            segments[i] = .{ .cache = cache, .rows = rows, .parents = w.parents };
            ids[total] = w.pending;
            @memcpy(ids[total + 1 ..][0..w.tokens.len], w.tokens);
            total += rows;
        }
        for (segments[0..windows.len]) |seg| {
            try seg.cache.beginTree(seg.rows, seg.parents);
            begun += 1;
        }
        const verify_gen = self.nextVerifyGeneration();
        const generation = if (self.prepared) self.nextTapGeneration() else 0;
        const taps: ?st.Ref = if (self.prepared) .{ .buf = self.scratch.taps.? } else null;
        try fwd.runShared(self.model, &self.scratch, segments[0..windows.len], ids[0..total], .{ .all = true, .taps = taps });
        const all = self.scratch.logits_all.slice(f32, total * self.vocab());
        var base: usize = 0;
        for (windows, out, segments[0..windows.len]) |w, o, seg| {
            if (self.prepared) seg.cache.tap = .{ .buf = self.scratch.taps, .off = base * st.feature_row_bytes, .start = @intCast(seg.cache.len), .rows = seg.rows, .gen = generation };
            seg.cache.logits_base = base;
            seg.cache.logits_gen = verify_gen;
            @memcpy(o.drafts, w.tokens);
            for (0..seg.rows) |r| {
                const l = all[(base + r) * self.vocab() ..][0..self.vocab()];
                const mask = draws.maskRow(w.masks, words, r);
                o.sampled[r] = try self.draw(l, w.stream.sampling, w.positions[r], mask);
                if (o.rows.len == seg.rows) o.rows[r] = draws.logprobRow(l, o.sampled[r], w.stream.logprobs orelse 0);
            }
            base += seg.rows;
        }
    }

    fn keepFn(ptr: *anyopaque, windows: []const be.Window, paths: []const []const u32) !void {
        const self = cast(ptr);
        try self.settleCommit();
        if (windows.len != paths.len or windows.len > st.batch_rows) return error.InvalidInklingWindows;
        var segs: [st.batch_rows]fwd.Segment = undefined;
        var prefix_segs: [st.batch_rows]fwd.Segment = undefined;
        var prefix_acc: [st.batch_rows]usize = undefined;
        var prefix_n: usize = 0;
        var tree_idx: [st.batch_rows]usize = undefined;
        var tree_n: usize = 0;
        for (windows, paths, 0..) |w, path, i| {
            const cache = try self.cacheOf(w.stream);
            for (segs[0..i]) |prev| if (prev.cache == cache) return error.UnsupportedInklingWindow;
            const rows = w.rows();
            if (cache.pending == null or cache.pending.? != rows) return error.UnsupportedInklingWindow;
            if (w.parents) |ps| {
                if (ps.len != rows) return error.UnsupportedInklingWindow;
                if (cache.pending_has_tree) {
                    if (!std.mem.eql(i32, ps, cache.pending_parents[0..rows])) return error.UnsupportedInklingWindow;
                } else if (!st.isChain(ps)) return error.UnsupportedInklingWindow;
            }
            try st.validateKeepPath(rows, if (cache.pending_has_tree) cache.pending_parents[0..rows] else null, path);
            segs[i] = .{ .cache = cache, .rows = rows };
            if (st.isPrefixPath(path)) {
                prefix_segs[prefix_n] = segs[i];
                prefix_acc[prefix_n] = path.len;
                prefix_n += 1;
            } else {
                tree_idx[tree_n] = i;
                tree_n += 1;
            }
        }
        if (windows.len == 0) return;
        const seq = self.nextCommitSeq();
        {
            const pool = mtl.objc.Pool.push();
            defer pool.pop();
            const cb = self.model.queue.commandBuffer();
            const e = cb.compute(.concurrent);
            fwd_ops.commitHistories(self.model, e, prefix_segs[0..prefix_n], prefix_acc[0..prefix_n]);
            for (tree_idx[0..tree_n]) |i| segs[i].cache.encodeCommitPath(e, paths[i], &self.model.tree);
            e.end();
            cb.commit();
            self.commit_cb = cb.retain();
            self.commit_active_seq = seq;
        }
        for (segs[0..windows.len]) |seg| seg.cache.commit_seq = seq;
        for (segs[0..windows.len], paths) |seg, path| {
            try seg.cache.finishCommitPath(path);
            const last = path[path.len - 1];
            if (seg.cache.logits_gen == self.verify_gen) {
                const all = self.scratch.logits_all.slice(f32, st.batch_rows * self.vocab());
                @memcpy(seg.cache.logits.slice(f32, self.vocab()), all[(seg.cache.logits_base + last) * self.vocab() ..][0..self.vocab()]);
            }
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
        self.settleCommit() catch |err| if (!builtin.is_test) std.log.err("inkling2_loop commit at release: {s}", .{@errorName(err)});
        if (self.caches.fetchRemove(stream)) |entry| {
            const cache = entry.value;
            if (!cache.commit_failed and cache.pending != null) cache.rollback() catch |err| std.log.err("inkling2_loop rollback at release: {s}", .{@errorName(err)});
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

    fn syncQueue(self: *Metal) !void {
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        const cb = self.model.queue.commandBuffer();
        const e = cb.compute(.serial);
        e.barrier();
        e.end();
        cb.commit();
        cb.wait();
        if (cb.failure()) |msg| {
            std.log.err("inkling2_loop queue barrier: {s}", .{msg});
            return error.GpuFailed;
        }
    }

    pub fn snapSave(ptr: *anyopaque, owner: ?*anyopaque, at: u32) anyerror!*anyopaque {
        const self = cast(ptr);
        try self.settleCommit();
        const stream: *lanes.Stream = @ptrCast(@alignCast(owner orelse return error.NoStream));
        const cache = try self.cacheOf(stream);
        if (cache.len != at or cache.pending != null) return error.SnapshotOutOfStep;
        if (self.prepared) try self.syncQueue();
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

fn gib(bytes: u64) f64 {
    return @as(f64, @floatFromInt(bytes)) / (1 << 30);
}
