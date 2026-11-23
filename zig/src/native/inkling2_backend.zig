//! Inkling 2 Mini's synchronous Metal target behind shared lane rounds: one pooled state per admitted stream.
const std = @import("std");
const mtl = @import("metal");
const api = @import("engine_api");
const tf = @import("tensorfold");
const lanes = tf.lanes;
const fl = tf.inkling2_metal;
const be = lanes.backend;
const Allocator = std.mem.Allocator;
const costs = @import("inkling2_costs.zig");

pub const window_rows = 16;
pub const max_rows = 512;
const max_items = 64;
const ring_size = 4096;

const kept_states = @import("inkling2_kept.zig");
const Kept = kept_states.Kept;

pub const Images = struct {
    ctx: *anyopaque,
    encode: *const fn (*anyopaque, lanes.stream.Image) anyerror!mtl.Buffer,
};

pub const Options = struct { capacity: u32, streams: u32, rows: u32, drafts: bool = true, images: ?Images = null };

/// A stream's state, first-draw logits, and draft taps that outlive the shared forward buffers.
const Lane = struct {
    state: fl.state.State,
    logits: []f32,
    taps: ?mtl.Buffer,
    tap_start: u32 = 0,
    tap_rows: u32 = 0,
    first_ready: bool = false,
    first_pick: ?u32 = null,
    verify_start: u32 = 0,
    verify_rows: u32 = 0,

    fn clear(lane: *Lane) void {
        lane.tap_start = 0;
        lane.tap_rows = 0;
        lane.first_ready = false;
        lane.first_pick = null;
        lane.verify_start = 0;
        lane.verify_rows = 0;
    }
};

/// A span of the model's tap buffer to retain for the stream's drafter.
const Copy = struct { lane: *Lane, start: u32, rows: u32, tap_base: usize };

pub const Target = struct {
    gpa: Allocator,
    model: *fl.Model,
    options: Options,
    pool: []Lane,
    idle: []*Lane,
    idle_n: usize,
    active: std.AutoHashMapUnmanaged(*lanes.Stream, *Lane) = .empty,
    tokens: [ring_size]u32 = undefined,
    next: u64 = 0,
    sampling_scratch: []u8,
    follow: ?mtl.ResidencySet.Follow = null,
    costs: costs.Table = .{}, // timed at load for a drafted server (Target.measureCosts); the lane config reads it
    head: ?*fl.draft.Adapter = null, // a drafted server's DSpark head: a kept state holds its context beside the lane

    pub fn samplingBytes(model: *const fl.Model) usize {
        return @as(usize, model.config.text.unpadded_vocab) * 64 + 128;
    }

    pub fn laneBytes(model: *const fl.Model, drafts: bool) u64 {
        return fl.layout.laneBytes(&model.config, model.geometry(), drafts, window_rows);
    }

    pub fn init(gpa: Allocator, model: *fl.Model, options: Options) !*Target {
        if (options.streams == 0 or options.streams > max_items or options.rows < window_rows or options.rows > max_rows or
            options.capacity != model.options.capacity or options.rows != model.options.rows or (options.drafts and !model.options.drafts))
            return error.BadInklingLaneOptions;
        const out = try gpa.create(Target);
        errdefer gpa.destroy(out);
        const scratch = try gpa.alloc(u8, samplingBytes(model));
        errdefer gpa.free(scratch);
        const pool = try gpa.alloc(Lane, options.streams);
        errdefer gpa.free(pool);
        const idle = try gpa.alloc(*Lane, options.streams);
        errdefer gpa.free(idle);
        out.* = .{ .gpa = gpa, .model = model, .options = options, .pool = pool, .idle = idle, .idle_n = 0, .sampling_scratch = scratch };
        try out.active.ensureTotalCapacity(gpa, options.streams);
        errdefer out.active.deinit(gpa);
        errdefer for (pool[0..out.idle_n]) |*lane| out.freeLane(lane);
        for (pool) |*lane| {
            lane.* = try out.makeLane();
            idle[out.idle_n] = lane;
            out.idle_n += 1;
        }
        return out;
    }

    pub fn deinit(self: *Target) void {
        for (self.pool) |*lane| self.freeLane(lane);
        self.active.deinit(self.gpa);
        self.gpa.free(self.pool);
        self.gpa.free(self.idle);
        self.gpa.free(self.sampling_scratch);
        self.gpa.destroy(self);
    }

    pub fn facts(self: *const Target) lanes.Model {
        return .{ .exact_width = window_rows, .gpu_tokens = false, .hidden_rows = true, .batch_rows = self.options.rows, .mtp = false, .speculate = false, .speculate_early = false, .drafts = 1, .max_streams = self.options.streams, .streams_exact = true, .shared_costs = self.costs.slice() };
    }

    /// Times the shared forward by total rows, for the lane rule's draft pricing (a plain server never drafts).
    pub fn measureCosts(self: *Target, io: std.Io) !void {
        try costs.measure(self.model, io, &self.costs);
    }

    pub fn backend(self: *Target) be.Backend {
        return .{ .ptr = self, .chunk_invariant = true, .vtable = &.{ .prefill = prefill, .prefill_features = prefillFeatures, .first = first, .first_masked = firstMasked, .queue = queue, .read = read, .verify = verify, .keep = keep, .draft = draft, .release = release, .prepare_features = prepareFeatures, .features = features, .first_row = firstRow, .held = held } };
    }

    fn of(ptr: *anyopaque) *Target {
        return @ptrCast(@alignCast(ptr));
    }

    fn laneOf(self: *Target, stream: *lanes.Stream) !*Lane {
        return self.active.get(stream) orelse error.UnknownInklingStream;
    }

    pub fn setResident(self: *Target, hook: ?mtl.ResidencySet.Follow) void {
        for (self.pool) |*lane| {
            lane.state.setResident(null);
            if (lane.taps) |taps| if (self.follow) |f| f.leave(f.ctx, taps);
            lane.state.setResident(hook);
            if (lane.taps) |taps| if (hook) |f| f.join(f.ctx, taps);
        }
        self.follow = hook;
    }

    fn vocab(self: *const Target) usize {
        return self.model.config.text.unpadded_vocab;
    }

    fn tapBytes(self: *const Target) usize {
        if (!self.options.drafts) return 0;
        const m = self.model.config.mtp orelse return 0;
        return @as(usize, m.taps_n) * self.model.config.text.hidden * 2;
    }

    fn makeLane(self: *Target) !Lane {
        var state = try fl.state.State.init(self.gpa, self.model.device, &self.model.config, .{ .capacity = self.options.capacity, .rows = self.options.rows });
        errdefer state.deinit();
        const logits = try self.gpa.alloc(f32, self.vocab());
        errdefer self.gpa.free(logits);
        const taps = if (self.options.drafts) try self.model.device.buffer(window_rows * self.tapBytes(), mtl.ResourceOptions.shared) else null;
        return .{ .state = state, .logits = logits, .taps = taps };
    }

    fn freeLane(self: *Target, lane: *Lane) void {
        lane.state.setResident(null);
        if (lane.taps) |taps| if (self.follow) |f| f.leave(f.ctx, taps);
        lane.state.deinit();
        self.gpa.free(lane.logits);
        if (lane.taps) |taps| taps.deinit();
    }

    /// An idle lane at length 0 for `stream`; `stream.cached` starts at 0.
    fn newLane(self: *Target, stream: *lanes.Stream) !*Lane {
        const ids = stream.prompt();
        if (self.active.contains(stream) or self.idle_n == 0 or ids.len == 0 or
            ids.len > self.options.capacity or stream.max_new > self.options.capacity - ids.len)
            return error.InklingRequestCapacity;
        self.idle_n -= 1;
        const lane = self.idle[self.idle_n];
        self.active.putAssumeCapacity(stream, lane);
        stream.cached = 0;
        return lane;
    }

    /// Copy every lane's kept rows out of the model's shared buffers in one command buffer.
    fn capture(self: *Target, copies: []const Copy) !void {
        if (copies.len == 0) return;
        const tap_bytes = self.tapBytes();
        const source = self.model.taps.?;
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        const cb = self.model.queue.commandBuffer();
        const blit = cb.blit();
        for (copies) |c| blit.copy(source, c.tap_base * tap_bytes, c.lane.taps.?, 0, c.rows * tap_bytes);
        blit.end();
        cb.commit();
        cb.wait();
        if (cb.failure() != null) return error.InklingGpuFailed;
        for (copies) |c| {
            c.lane.tap_start = c.start;
            c.lane.tap_rows = c.rows;
        }
    }

    fn prefill(ptr: *anyopaque, stream: *lanes.Stream) !void {
        return of(ptr).fill(stream, null);
    }

    fn prefillFeatures(ptr: *anyopaque, stream: *lanes.Stream, taps: []const u32, sink: be.FeatureSink) !void {
        try prepareFeatures(ptr, taps);
        return of(ptr).fill(stream, sink);
    }

    /// A kept prompt state of this prompt's prefix: the lane starts there, or the pass runs from 0.
    fn restoreKept(self: *Target, lane: *Lane, stream: *lanes.Stream, prompt_len: usize, drafting: bool) void {
        const saved = stream.reuse.saved orelse return;
        const kept: *const Kept = @ptrCast(@alignCast(saved));
        const snap = &kept.snap;
        if (kept.target != self or snap.len != stream.reuse.at or snap.len >= prompt_len) {
            stream.reuse_failed = true;
            return;
        }
        self.model.restore(&lane.state, snap) catch |err| return self.refuseKept(lane, stream, err);
        // a drafting stream's head takes the context kept beside the lane; with none kept it begins past the prefix
        if (drafting) if (self.head) |h| if (kept.head) |*context| {
            h.restore(stream, context) catch |err| return self.refuseKept(lane, stream, err);
        };
        stream.cached = snap.len;
    }

    fn refuseKept(self: *Target, lane: *Lane, stream: *lanes.Stream, err: anyerror) void {
        stream.reuse_failed = true;
        stream.reuse_error = err;
        self.model.reset(&lane.state); // a failed copy may have written part of the state
    }

    fn fill(self: *Target, stream: *lanes.Stream, sink: ?be.FeatureSink) !void {
        const lane = try self.newLane(stream);
        errdefer release(self, stream);
        const ids = stream.prompt();
        self.restoreKept(lane, stream, ids.len, sink != null);
        const hidden: usize = self.model.config.text.hidden;
        const first_row = lane.state.len;
        const image_rows = try self.gpa.alloc(fl.ImageRows, stream.images.len);
        defer self.gpa.free(image_rows);
        var made: usize = 0;
        defer for (image_rows[0..made]) |im| im.buf.deinit();
        for (stream.images) |im| {
            if (im.at > ids.len or im.tokens > ids.len - im.at) return error.BadInklingImageRows;
            if (im.at + im.tokens <= first_row) continue; // inside the restored prefix: its rows are already in the state
            const encoder = self.options.images orelse return error.InklingVisionDisabled;
            const buffer = try encoder.encode(encoder.ctx, im);
            image_rows[made] = .{ .row = im.at, .rows = im.tokens, .buf = buffer, .off = 0 };
            made += 1;
            if (buffer.length() < @as(usize, im.tokens) * hidden * 2) return error.BadInklingImageRows;
        }
        const chunk_images = try self.gpa.alloc(fl.ImageRows, made);
        defer self.gpa.free(chunk_images);
        var last: struct { start: u32, rows: u32 } = .{ .start = 0, .rows = 0 };
        while (lane.state.len < ids.len) {
            if (stream.isCancelled()) return error.Cancelled;
            const start = lane.state.len;
            const end = cutAt(stream.reuse.marks, cutAt(stream.chunks, @min(ids.len, start + self.options.rows), start), start);
            var count: usize = 0;
            for (image_rows[0..made]) |im| {
                const from = @max(start, im.row);
                const to = @min(end, im.row + im.rows);
                if (from >= to) continue;
                chunk_images[count] = .{ .row = @intCast(from - start), .rows = @intCast(to - from), .buf = im.buf, .off = im.off + (from - im.row) * hidden * 2 };
                count += 1;
            }
            try self.model.forward(&.{.{
                .state = &lane.state,
                .ids = ids[start..end],
                .images = chunk_images[0..count],
                .logit_rows = @intFromBool(end == ids.len),
                .taps = sink != null,
            }});
            stream.fill_at.store(lane.state.len, .monotonic);
            if (sink) |s| {
                const f: be.Features = .{ .buffer = @intFromPtr(self.model.taps.?.id), .rows = @intCast(end - start), .row_bytes = @intCast(self.tapBytes()), .space = .device, .dtype = .bf16 };
                try s.take(s.ptr, stream, f, start);
            }
            lanes.prefill_widths.completed(self.gpa, stream, end - start);
            last = .{ .start = @intCast(end - 1), .rows = @intCast(end - start) };
            if (end < ids.len and std.mem.indexOfScalar(u32, stream.reuse.marks, @intCast(end)) != null) if (stream.reuse.hook) |k| k.at(k.ptr, stream, @intCast(end));
        }
        @memcpy(lane.logits, self.model.logits.slice(f32, self.vocab()));
        lane.first_ready = true;
        if (sink != null) try self.capture(&.{.{ .lane = lane, .start = last.start, .rows = 1, .tap_base = last.rows - 1 }});
    }

    fn token(self: *Target, id: u32) !u64 {
        if (id >= self.vocab()) return error.InvalidInklingToken;
        const handle = self.next;
        self.next = try std.math.add(u64, self.next, 1);
        self.tokens[handle % ring_size] = id;
        return handle;
    }

    fn read(ptr: *anyopaque, handle: u64) !u32 {
        const self = of(ptr);
        if (handle >= self.next or self.next - handle > ring_size) return error.StaleInklingToken;
        return self.tokens[handle % ring_size];
    }

    fn draw(self: *Target, logits: []const f32, stream: *lanes.Stream, position: u64, mask: []const u32) !u32 {
        var scratch = std.heap.FixedBufferAllocator.init(self.sampling_scratch);
        var best: ?usize = null;
        for (logits, 0..) |x, id| {
            if (!std.math.isFinite(x)) return error.NonfiniteLogits;
            if (mask.len != 0 and !lanes.grammar.allows(mask, id)) continue;
            if (best == null or x > logits[best.?]) best = id;
        }
        const greedy = best orelse return error.NoAllowedToken;
        const settings = stream.sampling orelse return @intCast(greedy);
        if (settings.temperature <= 0) return @intCast(greedy);
        const allowed = if (mask.len == 0) logits else blk: {
            const values = try scratch.allocator().alloc(f32, logits.len);
            for (logits, values, 0..) |x, *v, id| v.* = if (lanes.grammar.allows(mask, id)) x else -std.math.inf(f32);
            break :blk values;
        };
        return lanes.gpu_full.draw(scratch.allocator(), allowed, settings, @intCast(position), null);
    }

    fn lastLogits(lane: *const Lane) ![]const f32 {
        if (!lane.first_ready) return error.StaleInklingLogits;
        return lane.logits;
    }

    fn first(ptr: *anyopaque, stream: *lanes.Stream, position: u64) !u64 {
        return firstMasked(ptr, stream, position, &.{});
    }

    fn firstMasked(ptr: *anyopaque, stream: *lanes.Stream, position: u64, mask: []const u32) !u64 {
        const self = of(ptr);
        const lane = try self.laneOf(stream);
        if (position != lane.state.len) return error.InvalidInklingPosition;
        const pick = try self.draw(try lastLogits(lane), stream, position, mask);
        lane.first_pick = pick;
        return self.token(pick);
    }

    fn queue(ptr: *anyopaque, stream: *lanes.Stream, feed: be.Feed, position: u64) !u64 {
        const self = of(ptr);
        const lane = try self.laneOf(stream);
        if (position != @as(u64, lane.state.len) + 1) return error.InvalidInklingPosition;
        const id = switch (feed) {
            .value => |v| v,
            .handle => |h| try read(ptr, h),
        };
        const start = lane.state.len;
        const tapping = self.options.drafts and stream.drafts;
        try self.model.forward(&.{.{ .state = &lane.state, .ids = &.{id}, .taps = tapping }});
        lane.first_ready = false;
        if (tapping) try self.capture(&.{.{ .lane = lane, .start = start, .rows = 1, .tap_base = 0 }});
        return self.token(try self.draw(self.model.logits.slice(f32, self.vocab()), stream, position, &.{}));
    }

    fn verify(ptr: *anyopaque, windows: []const be.Window, out: []be.Verified) !void {
        const self = of(ptr);
        if (windows.len == 0 or windows.len != out.len or windows.len > self.options.streams) return error.InvalidInklingWindows;
        var items: [max_items]fl.Item = undefined;
        var ids: [max_items * window_rows]u32 = undefined;
        var total: usize = 0;
        const words = (self.vocab() + 31) / 32;
        for (windows, out, 0..) |w, result, i| {
            const n = w.rows();
            if (n == 0 or n > window_rows or total + n > self.options.rows or w.held != 0 or w.parents != null or
                w.positions.len != n or result.sampled.len != n or result.drafts.len != n - 1 or
                (w.masks.len != 0 and w.masks.len != n * words)) return error.InvalidInklingWindows;
            const lane = try self.laneOf(w.stream);
            for (windows[0..i]) |previous| if (previous.stream == w.stream) return error.DuplicateInklingStream;
            for (w.positions, 0..) |position, r| if (position != @as(u64, lane.state.len) + r + 1) return error.InvalidInklingPosition;
            ids[total] = w.pending;
            @memcpy(ids[total + 1 ..][0..w.tokens.len], w.tokens);
            const tapping = self.options.drafts and w.stream.drafts;
            items[i] = .{ .state = &lane.state, .ids = ids[total..][0..n], .logit_rows = @intCast(n), .taps = tapping, .tap_into = if (tapping) lane.taps else null };
            lane.first_ready = false;
            lane.verify_start = lane.state.len;
            lane.verify_rows = @intCast(n);
            total += n;
        }
        try self.model.forward(items[0..windows.len]);
        for (windows) |w| if (self.options.drafts and w.stream.drafts) { // the forward wrote each lane's own taps
            const lane = self.active.get(w.stream).?;
            lane.tap_start = lane.verify_start;
            lane.tap_rows = lane.verify_rows;
        };
        const logits = self.model.logits.slice(f32, total * self.vocab());
        var base: usize = 0;
        for (windows, out) |w, result| {
            const n = w.rows();
            @memcpy(result.drafts, w.tokens);
            for (0..n) |r| {
                const values = logits[(base + r) * self.vocab() ..][0..self.vocab()];
                const mask = if (w.masks.len == 0) &.{} else w.masks[r * words ..][0..words];
                result.sampled[r] = try self.draw(values, w.stream, w.positions[r], mask);
                if (result.rows.len == n) result.rows[r] = row(values, result.sampled[r], w.stream.logprobs orelse 0);
            }
            base += n;
        }
    }

    fn keep(ptr: *anyopaque, windows: []const be.Window, paths: []const []const u32) !void {
        const self = of(ptr);
        if (windows.len != paths.len) return error.InvalidInklingKeep;
        for (windows, paths) |w, path| {
            const lane = try self.laneOf(w.stream);
            if (lane.verify_rows != w.rows() or path.len == 0 or path.len > lane.verify_rows) return error.InvalidInklingKeep;
            for (path, 0..) |r, i| if (r != i) return error.InklingTreesUnsupported;
            try self.model.keep(&lane.state, lane.verify_start + @as(u32, @intCast(path.len)));
        }
    }

    fn prepareFeatures(ptr: *anyopaque, taps: []const u32) !void {
        const self = of(ptr);
        if (!self.options.drafts) return error.InklingHasNoDraftHead;
        const mtp = self.model.config.mtp orelse return error.InklingHasNoDraftHead;
        if (!std.mem.eql(u32, taps, mtp.taps[0..mtp.taps_n])) return error.InvalidInklingTaps;
    }

    fn features(ptr: *anyopaque, stream: *lanes.Stream, taps: []const u32, start: u64, count: u32) !be.Features {
        const self = of(ptr);
        try prepareFeatures(ptr, taps);
        const lane = try self.laneOf(stream);
        const buffer = lane.taps orelse return error.InklingFeaturesNotHeld;
        if (start < lane.tap_start or start - lane.tap_start > lane.tap_rows or count > lane.tap_rows - (start - lane.tap_start))
            return error.InklingFeaturesNotHeld;
        return .{ .buffer = @intFromPtr(buffer.id), .offset = (start - lane.tap_start) * self.tapBytes(), .rows = count, .row_bytes = @intCast(self.tapBytes()), .space = .device, .dtype = .bf16 };
    }

    fn firstRow(ptr: *anyopaque, stream: *lanes.Stream) !lanes.LogprobRow {
        const self = of(ptr);
        const lane = try self.laneOf(stream);
        return firstRowOf(try lastLogits(lane), lane.first_pick, stream.logprobs orelse 0);
    }

    fn draft(_: *anyopaque, requests: []const be.DraftRequest) !void {
        for (requests) |r| if (r.depth != 0) return error.InklingDrafterNotAttached;
    }

    fn held(_: *anyopaque, _: *lanes.Stream, tokens: []u32) !void {
        if (tokens.len != 0) return error.InklingDrafterNotAttached;
    }

    /// The stream left the rounds: its lane is emptied (histories zeroed, length 0) and goes back to the pool.
    fn release(ptr: *anyopaque, stream: *lanes.Stream) void {
        const self = of(ptr);
        const entry = self.active.fetchRemove(stream) orelse return;
        self.model.reset(&entry.value.state);
        entry.value.clear();
        self.idle[self.idle_n] = entry.value;
        self.idle_n += 1;
    }

    /// The prompt cache's copies of a stream's state at a chunk end; the prompt pass restores them itself.
    pub fn snapshots(self: *Target) api.prompt_cache.Snapshots {
        return kept_states.snapshots(self);
    }

    /// The state of the stream's lane (an error if it has none).
    pub fn stateOf(self: *Target, stream: *lanes.Stream) !*fl.state.State {
        return &(try self.laneOf(stream)).state;
    }
};

/// The earliest of `end` and the first cut of `cuts` inside (start, end).
fn cutAt(cuts: []const u32, end: usize, start: usize) usize {
    var out = end;
    for (cuts) |c| if (c > start and c < out) {
        out = c;
    };
    return out;
}

/// The first token's logprob row: of the token the draw selected (a masked or sampled pick), argmax only without one.
fn firstRowOf(values: []const f32, pick: ?u32, count: u8) lanes.LogprobRow {
    return row(values, pick orelse lanes.gpu_rule.argmax(values), count);
}

fn row(values: []const f32, pick: u32, count: u8) lanes.LogprobRow {
    const top = lanes.logprob.max_top;
    var best: [top]f32 = @splat(-std.math.inf(f32));
    var result: lanes.LogprobRow = .{ .token = pick, .logprob = undefined, .count = @intCast(@min(count, @min(top, values.len))) };
    var peak: f32 = -std.math.inf(f32);
    for (values, 0..) |value, id| {
        peak = @max(peak, value);
        if (value >= values[pick]) result.rank += 1;
        var slot: usize = result.count;
        while (slot > 0 and value > best[slot - 1]) : (slot -= 1) {
            if (slot < result.count) {
                best[slot] = best[slot - 1];
                result.ids[slot] = result.ids[slot - 1];
            }
        }
        if (slot < result.count) {
            best[slot] = value;
            result.ids[slot] = @intCast(id);
        }
    }
    var sum: f64 = 0;
    for (values) |value| sum += @exp(@as(f64, value) - peak);
    const lse = @as(f64, peak) + @log(sum);
    result.logprob = @floatCast(@as(f64, values[pick]) - lse);
    for (0..result.count) |i| result.logprobs[i] = @floatCast(@as(f64, best[i]) - lse);
    return result;
}

test "inkling2 target keeps float32 logits and obeys a grammar mask" {
    var memory: [4096]u8 = undefined;
    var target: Target = undefined;
    target.sampling_scratch = &memory;
    var stream: lanes.Stream = undefined;
    stream.sampling = null;
    const logits = [_]f32{ 1.003, 1.004, -2 };
    try std.testing.expectEqual(@as(u32, 1), try target.draw(&logits, &stream, 17, &.{}));
    try std.testing.expectEqual(@as(u32, 2), try target.draw(&logits, &stream, 17, &.{0b100}));
    try std.testing.expectError(error.NoAllowedToken, target.draw(&logits, &stream, 17, &.{0}));
    stream.sampling = .{ .seed = 9, .temperature = 0.7, .top_k = 0 };
    try std.testing.expectEqual(@as(u32, 2), try target.draw(&logits, &stream, 17, &.{0b100}));
}

test "inkling2 target reports raw float32 logprobs and stable tied alternatives" {
    const got = row(&.{ 1, 1, 0, 1 }, 2, 3);
    const lse: f64 = @log(3 * @exp(@as(f64, 1)) + 1);
    try std.testing.expectApproxEqAbs(@as(f32, @floatCast(-lse)), got.logprob, 1e-6);
    try std.testing.expectEqualSlices(u32, &.{ 0, 1, 3 }, got.ids[0..3]);
    try std.testing.expectEqual(@as(u32, 4), got.rank);
}

test "inkling2 first logprob row follows the selected token, not the argmax" {
    const values = [_]f32{ 0.5, 3, -1, 2 };
    const lse: f64 = @log(@exp(@as(f64, 0.5)) + @exp(@as(f64, 3)) + @exp(@as(f64, -1)) + @exp(@as(f64, 2)));
    // a masked or sampled first draw picks token 2; its row keeps token 2's logprob, rank and the true top alternatives
    const picked = firstRowOf(&values, 2, 2);
    try std.testing.expectEqual(@as(u32, 2), picked.token);
    try std.testing.expectApproxEqAbs(@as(f32, @floatCast(@as(f64, -1) - lse)), picked.logprob, 1e-6);
    try std.testing.expectEqual(@as(u32, 4), picked.rank);
    try std.testing.expectEqualSlices(u32, &.{ 1, 3 }, picked.ids[0..2]);
    // no recorded pick (a plain prompt-only read): the argmax's row
    const argmax = firstRowOf(&values, null, 0);
    try std.testing.expectEqual(@as(u32, 1), argmax.token);
    try std.testing.expectEqual(@as(u32, 1), argmax.rank);
}

test "inkling2 prompt chunks cut at the earliest of the stream's chunk starts and reuse marks" {
    try std.testing.expectEqual(@as(usize, 100), cutAt(&.{}, 100, 0));
    try std.testing.expectEqual(@as(usize, 64), cutAt(&.{ 64, 90 }, 100, 0));
    try std.testing.expectEqual(@as(usize, 40), cutAt(&.{50}, cutAt(&.{ 0, 40 }, 100, 0), 0));
    try std.testing.expectEqual(@as(usize, 100), cutAt(&.{ 0, 100, 128 }, 100, 0));
    try std.testing.expectEqual(@as(usize, 100), cutAt(&.{64}, 100, 64));
}

test "inkling2 target prices shared rounds by its timed table and leaves a lone stream's rule alone" {
    var t: Target = undefined;
    t.options = .{ .capacity = 64, .streams = 4, .rows = max_rows };
    t.costs = .{};
    for (costs.widths, 0..) |width, i| t.costs.costs[i] = .{ .width = width, .ms = 30 + 5 * @as(f64, @floatFromInt(width)) };
    t.costs.timed = costs.widths.len;
    var cfg = try lanes.Config.init(std.testing.allocator, t.facts(), window_rows, window_rows - 1);
    defer cfg.deinit(std.testing.allocator);
    try std.testing.expectEqual(@as(?f64, 55), cfg.shared_costs.get(5)); // between the timed 4 and 6 rows
    try std.testing.expectEqual(@as(?f64, 350), cfg.shared_costs.get(64));
    try std.testing.expect(cfg.family_costs.empty()); // a lone stream keeps the depth heuristic
    t.costs = .{};
    var untimed = try lanes.Config.init(std.testing.allocator, t.facts(), window_rows, window_rows - 1);
    defer untimed.deinit(std.testing.allocator);
    try std.testing.expect(untimed.shared_costs.empty()); // a plain server times nothing
}
