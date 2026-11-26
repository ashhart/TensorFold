//! Flint behind the lane engine: windows a stream, a cache each; lone steps run ahead, drawn on the GPU into a ring.
const std = @import("std");
const mtl = @import("metal");
const lanes = @import("lanes");
const st = @import("state.zig");
const fwd = @import("forward.zig");
const Model = @import("model.zig").Model;
const snapshot = @import("snapshot.zig");
const row = lanes.logprob.bf16Row;
const be = lanes.backend;

/// The longest context a stream may take, the streams at once, and the bytes every stream's caches may hold.
pub const Options = struct { capacity: usize, streams: u32 = 4, budget: u64 };

/// Token slots a stream's handles point at: drawn tokens and host-fed ones (no command in flight reads a fresh one).
const ring = 4096;

/// A committed step: the token slots below `end` are written when `cb` completes.
const Flight = struct { end: u64, cb: mtl.CommandBuffer };

/// The prompt cache's room beside the streams: its kept states fit in `bytes`, never evicting `restoring`.
pub const Room = struct { ctx: *anyopaque, fit: *const fn (ctx: *anyopaque, bytes: u64, restoring: ?*anyopaque, prompt: []const u32) void };

pub const Metal = struct {
    gpa: std.mem.Allocator,
    model: *Model,
    options: Options,
    scratch: st.Scratch, // decode rounds
    prompt: st.Scratch, // prompt chunks
    caches: std.AutoHashMapUnmanaged(*lanes.Stream, *st.Cache) = .empty,
    follow: ?mtl.ResidencySet.Follow = null,
    held: u64 = 0, // bytes the streams' caches hold
    room: ?Room = null, // the prompt cache, which takes what the streams leave of the budget
    tokens: mtl.Buffer, // u32 [ring]
    next: u64 = 0, // the next free token slot
    landed: u64 = 0, // slots below it are written
    flights: std.ArrayList(Flight) = .empty,
    tap_ids: []const u32 = &.{},
    costs: [st.batch_rows]lanes.config.Cost = undefined,
    timed: bool = false,
    values: []f32, // a logits row as f32, for the host twin of the GPU draw

    pub fn init(gpa: std.mem.Allocator, model: *Model, options: Options) !*Metal {
        const out = try gpa.create(Metal);
        errdefer gpa.destroy(out);
        var scratch = try st.Scratch.init(model.device, &model.config, st.batch_rows);
        errdefer scratch.deinit();
        var prompt = try st.Scratch.init(model.device, &model.config, st.prompt_rows);
        errdefer prompt.deinit();
        const tokens = try model.device.buffer(ring * 4, mtl.ResourceOptions.shared);
        errdefer tokens.deinit();
        out.* = .{ .gpa = gpa, .model = model, .options = options, .scratch = scratch, .prompt = prompt, .tokens = tokens, .values = try gpa.alloc(f32, model.config.vocab) };
        return out;
    }

    pub fn deinit(self: *Metal) void {
        self.drain() catch {};
        self.flights.deinit(self.gpa);
        self.tokens.deinit();
        self.gpa.free(self.values);
        self.setResident(null);
        var it = self.caches.valueIterator();
        while (it.next()) |cache| {
            cache.*.deinit();
            self.gpa.destroy(cache.*);
        }
        self.caches.deinit(self.gpa);
        self.scratch.deinit();
        self.prompt.deinit();
        self.gpa.destroy(self);
    }

    pub fn setResident(self: *Metal, hook: ?mtl.ResidencySet.Follow) void {
        if (self.follow) |f| {
            mtl.resident.fields.apply(&self.scratch, f, false);
            mtl.resident.fields.apply(&self.prompt, f, false);
        }
        self.follow = hook;
        if (hook) |f| {
            mtl.resident.fields.apply(&self.scratch, f, true);
            mtl.resident.fields.apply(&self.prompt, f, true);
        }
        var it = self.caches.valueIterator();
        while (it.next()) |cache| cache.*.setResident(hook);
    }

    pub fn createCache(self: *Metal, capacity: usize) !*st.Cache {
        const cache = try self.gpa.create(st.Cache);
        errdefer self.gpa.destroy(cache);
        cache.* = try st.Cache.init(self.model.device, &self.model.config, capacity);
        if (self.tap_ids.len > 0) {
            errdefer cache.deinit();
            cache.taps = try self.model.device.buffer(st.prompt_rows * self.model.config.hidden * 5 * 2, mtl.ResourceOptions.shared);
        }
        cache.setResident(self.follow);
        return cache;
    }

    pub fn facts(self: *const Metal) lanes.Model {
        // rows are independent in every decode kernel, so any width, alone or beside other streams, gives one-row bits
        return .{ .exact_width = st.window_rows, .gpu_tokens = true, .hidden_rows = true, .batch_rows = st.batch_rows, .mtp = false, .speculate = false, .speculate_early = false, .drafts = 1, .max_streams = self.options.streams, .streams_exact = true, .window_costs = if (self.timed) self.costs[0..st.window_rows] else &.{}, .shared_costs = if (self.timed) &self.costs else &.{} };
    }

    pub fn backend(self: *Metal) be.Backend {
        return .{ .ptr = self, .vtable = &.{ .prefill = prefillFn, .first = firstFn, .first_masked = firstMaskedFn, .queue = queueFn, .read = readFn, .verify = verifyFn, .keep = keepFn, .draft = draftFn, .release = releaseFn, .first_row = firstRowFn, .held = heldFn, .prepare_features = prepareFeatures, .features = featuresFn, .prefill_features = prefillFeatures } };
    }

    fn cast(ptr: *anyopaque) *Metal {
        return @ptrCast(@alignCast(ptr));
    }

    fn cacheOf(self: *Metal, stream: *lanes.Stream) !*st.Cache {
        return self.caches.get(stream) orelse error.UnknownFlintStream;
    }

    fn logits(self: *Metal, cache: *st.Cache) []const u16 {
        return cache.logits.slice(u16, self.model.config.vocab);
    }

    /// The GPU draw's host twin (lanes' gpu_full): argmax or the keyed race at `position`, a mask's tokens only.
    fn draw(self: *Metal, l: []const u16, sampling: ?lanes.Sampling, position: u64, mask: []const u32) !u32 {
        const t = sampled(sampling) orelse { // gpu_rule's argmax: the first largest, masked tokens at -inf
            var best: ?usize = null;
            for (l, 0..) |w, id| {
                if (mask.len > 0 and !lanes.grammar.allows(mask, id)) continue;
                if (best == null or logit(w) > logit(l[best.?])) best = id;
            }
            return @intCast(best orelse 0);
        };
        for (l, self.values, 0..) |w, *v, id| {
            v.* = logit(w);
            if (mask.len > 0 and !lanes.grammar.allows(mask, id)) v.* = -std.math.inf(f32);
        }
        return lanes.gpu_full.draw(self.gpa, self.values, t, @intCast(position), null);
    }

    fn sampled(sampling: ?lanes.Sampling) ?lanes.Sampling {
        const t = sampling orelse return null;
        return if (t.temperature > 0) t else null;
    }

    /// `n` consecutive token slots, never wrapping the ring.
    fn take(self: *Metal, n: usize) u64 {
        if (self.next % ring + n > ring) self.next += ring - self.next % ring;
        const h = self.next;
        self.next += n;
        return h;
    }

    fn slotOf(self: *Metal, h: u64) fwd.B {
        return .{ .buf = self.tokens, .off = @intCast(h % ring * 4) };
    }

    /// Wait for the oldest step in flight.
    fn land(self: *Metal) !void {
        const f = self.flights.orderedRemove(0);
        defer f.cb.release();
        f.cb.wait();
        if (f.cb.failure()) |msg| {
            std.log.err("flint step: {s}", .{msg});
            return error.GpuFailed;
        }
        self.landed = @max(self.landed, f.end);
    }

    /// Every step in flight landed: before the host reads caches or logits, or the GPU runs anything else.
    fn drain(self: *Metal) !void {
        while (self.flights.items.len > 0) try self.land();
    }

    /// Run `ids` at the cache's end in prompt chunks; a stream's pass stops at its chunk starts and reuse marks.
    fn feed(self: *Metal, cache: *st.Cache, ids: []const u32, stream: ?*lanes.Stream, sink: ?be.FeatureSink) !void {
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
            const capture = self.tap_ids.len > 0 and stream != null and stream.?.drafts;
            try fwd.run(self.model, s, cache, ids[at..end], .{ .taps = capture });
            cache.len += end - at;
            if (sink) |sink_| try sink_.take(sink_.ptr, stream.?, try self.backend().features(stream.?, self.tap_ids, base + at, @intCast(end - at)), base + at);
            at = end;
            if (stream) |sm| if (std.mem.indexOfScalar(u32, sm.reuse.marks, @intCast(cache.len)) != null) if (sm.reuse.hook) |k| k.at(k.ptr, sm, @intCast(cache.len));
        }
    }

    fn prefillFn(ptr: *anyopaque, stream: *lanes.Stream) !void {
        return cast(ptr).prefill(stream, null);
    }

    fn prefill(self: *Metal, stream: *lanes.Stream, sink: ?be.FeatureSink) !void {
        try self.drain();
        const ids = stream.prompt();
        if (ids.len == 0 or ids.len + stream.max_new > self.options.capacity or self.caches.contains(stream)) return error.InvalidFlintPrompt;
        if (stream.images.len > 0) return error.FlintHasNoImageTower;
        const capacity = ids.len + stream.max_new + 1;
        const need = st.cacheBytes(&self.model.config, capacity);
        if (self.held + need > self.options.budget) {
            std.log.warn("flint: a {d}-token stream needs {d:.1} GiB of caches; {d:.1} of {d:.1} GiB are held", .{ capacity, gib(need), gib(self.held), gib(self.options.budget) });
            return error.FlintCachesFull;
        }
        self.held += need;
        errdefer { // as releaseFn: the bytes back and the prompt cache's room fitted to them (it was fitted short below)
            self.held -= need;
            if (self.room) |r| r.fit(r.ctx, self.options.budget - self.held, null, &.{});
        }
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
        try self.feed(cache, ids[cache.len..], stream, sink);
    }

    /// A prompt chunk and a decode window once: the first request skips the GPU's first touch.
    pub fn warm(self: *Metal) !void {
        const c = &self.model.config;
        var cache = try st.Cache.init(self.model.device, c, st.prompt_rows + st.window_rows);
        defer cache.deinit();
        var ids: [st.prompt_rows]u32 = undefined;
        for (&ids, 0..) |*t, i| t.* = @intCast(1000 + i * 37 % 20000);
        try self.feed(&cache, &ids, null, null);
        try fwd.run(self.model, &self.scratch, &cache, ids[0..st.window_rows], .{ .all = true });
    }

    pub fn calibrate(self: *Metal) !void {
        var cache = try st.Cache.init(self.model.device, &self.model.config, 128 + st.batch_rows);
        defer cache.deinit();
        const ids: [128]u32 = @splat(1000);
        try self.feed(&cache, &ids, null, null);
        for (&self.costs, 1..) |*cost, width| {
            var ms: [3]f64 = undefined;
            for (0..4) |i| {
                cache.len = 128;
                const t0 = fwd.seconds();
                try fwd.run(self.model, &self.scratch, &cache, ids[0..width], .{ .all = true });
                if (i > 0) ms[i - 1] = (fwd.seconds() - t0) * 1000;
            }
            std.mem.sort(f64, &ms, {}, std.sort.asc(f64));
            cost.* = .{ .width = @intCast(width), .ms = ms[1] };
        }
        self.timed = true;
    }

    /// The first token, drawn on the host from the prompt's last row into a fresh slot.
    fn firstFn(ptr: *anyopaque, stream: *lanes.Stream, position: u64) !u64 {
        return cast(ptr).firstSlot(stream, position, &.{});
    }

    fn firstMaskedFn(ptr: *anyopaque, stream: *lanes.Stream, position: u64, mask: []const u32) !u64 {
        return cast(ptr).firstSlot(stream, position, mask);
    }

    fn firstSlot(self: *Metal, stream: *lanes.Stream, position: u64, mask: []const u32) !u64 {
        try self.drain();
        const cache = try self.cacheOf(stream);
        if (position != cache.len) return error.InvalidFlintPosition;
        const h = self.take(1);
        self.tokens.slice(u32, ring)[h % ring] = try self.draw(self.logits(cache), stream.sampling, position, mask);
        return h;
    }

    /// A step behind those in flight: its token from the ring or the host, the next drawn into a slot; read() lands it.
    fn queueFn(ptr: *anyopaque, stream: *lanes.Stream, f: be.Feed, position: u64) !u64 {
        const self = cast(ptr);
        const cache = try self.cacheOf(stream);
        if (position != cache.len + 1) return error.InvalidFlintPosition;
        const in = switch (f) {
            .handle => |h| h,
            .value => |v| blk: {
                if (v >= self.model.config.vocab) return error.InvalidFlintToken;
                const h = self.take(1);
                self.tokens.slice(u32, ring)[h % ring] = v;
                break :blk h;
            },
        };
        const out = self.take(1);
        const how: @FieldType(fwd.Draw, "how") = if (sampled(stream.sampling)) |t| .{ .sampled = .{ .sampling = t, .position = position } } else .greedy;
        var cb: mtl.CommandBuffer = undefined;
        try fwd.run(self.model, &self.scratch, cache, &.{0}, .{ .taps = self.tap_ids.len > 0 and stream.drafts, .ids = self.slotOf(in), .draw = .{ .how = how, .out = self.slotOf(out) }, .flight = &cb });
        self.flights.append(self.gpa, .{ .end = out + 1, .cb = cb }) catch |err| {
            cb.wait();
            cb.release();
            return err;
        };
        cache.len += 1;
        return out;
    }

    fn firstRowFn(ptr: *anyopaque, stream: *lanes.Stream) !lanes.LogprobRow {
        const self = cast(ptr);
        try self.drain();
        const l = self.logits(try self.cacheOf(stream));
        return row(l, @intCast(argmax(l)), stream.logprobs orelse 0);
    }

    /// A slot's token, once the step that writes it has landed (host-fed slots are written at once).
    fn readFn(ptr: *anyopaque, handle: u64) !u32 {
        const self = cast(ptr);
        while (handle >= self.landed and self.flights.items.len > 0) try self.land();
        const token = self.tokens.slice(u32, ring)[handle % ring];
        if (token >= self.model.config.vocab) return error.InvalidFlintToken;
        return token;
    }

    /// Every window's pending token and drafts in one shared forward, every row drawn and kept until `keep` rolls back.
    fn verifyFn(ptr: *anyopaque, windows: []const be.Window, out: []be.Verified) !void {
        const self = cast(ptr);
        const c = &self.model.config;
        const words = (c.vocab + 31) / 32;
        if (windows.len != out.len or windows.len > st.batch_rows) return error.InvalidFlintWindows;
        try self.drain();
        var segments: [st.batch_rows]fwd.Segment = undefined;
        var ids: [st.batch_rows]u32 = undefined;
        var draws: [st.batch_rows]?fwd.Draw = @splat(null);
        const picked = self.take(st.batch_rows);
        var total: usize = 0;
        for (windows, out, 0..) |w, o, i| {
            const rows = w.rows();
            if (rows > st.window_rows or total + rows > st.batch_rows or w.parents != null or w.held != 0 or o.sampled.len != rows or o.drafts.len != rows - 1) return error.UnsupportedFlintWindow;
            const cache = try self.cacheOf(w.stream);
            for (w.positions, 0..) |p, r| if (p != cache.len + r + 1) return error.InvalidFlintPosition;
            segments[i] = .{ .cache = cache, .rows = rows };
            ids[total] = w.pending;
            @memcpy(ids[total + 1 ..][0..w.tokens.len], w.tokens);
            for (0..rows) |r| if (w.masks.len < (r + 1) * words) {
                const how: @FieldType(fwd.Draw, "how") = if (sampled(w.stream.sampling)) |t| .{ .sampled = .{ .sampling = t, .position = w.positions[r] } } else .greedy;
                draws[total + r] = .{ .how = how, .out = self.slotOf(picked + total + r) };
            };
            total += rows;
        }
        try fwd.runShared(self.model, &self.scratch, segments[0..windows.len], ids[0..total], .{ .all = true, .taps = self.tap_ids.len > 0, .draws = draws[0..total] });
        const all = self.scratch.logits_all.slice(u16, total * c.vocab);
        var base: usize = 0;
        for (windows, out, segments[0..windows.len]) |w, o, seg| {
            const rows = seg.rows;
            seg.cache.len += rows;
            @memcpy(o.drafts, w.tokens);
            for (0..rows) |r| {
                const l = all[(base + r) * c.vocab ..][0..c.vocab];
                const mask = if (w.masks.len >= (r + 1) * words) w.masks[r * words ..][0..words] else &.{};
                o.sampled[r] = if (draws[base + r] != null) self.tokens.slice(u32, ring)[(picked + base + r) % ring] else try self.draw(l, w.stream.sampling, w.positions[r], mask);
                if (o.rows.len == rows) o.rows[r] = row(l, o.sampled[r], w.stream.logprobs orelse 0);
            }
            // the last kept row's logits are the next first draw's (the lane core keeps every row here)
            @memcpy(seg.cache.logits.slice(u16, c.vocab), all[(base + rows - 1) * c.vocab ..][0..c.vocab]);
            base += rows;
        }
    }

    fn keepFn(ptr: *anyopaque, windows: []const be.Window, paths: []const []const u32) !void {
        const self = cast(ptr);
        try self.drain();
        for (windows, paths) |w, path| {
            const cache = try self.cacheOf(w.stream);
            for (path, 0..) |r, i| if (r != i) return error.UnsupportedFlintTree;
            if (path.len == 0 or path.len > w.rows() or w.rows() > cache.len) return error.UnsupportedFlintWindow;
            cache.len = cache.len - w.rows() + path.len;
            if (cache.tap_rows > 0) cache.tap_rows = @intCast(path.len);
        }
    }

    /// No draft head: a request at depth 0 holds nothing; the lane core's copy drafts come as window tokens.
    fn draftFn(ptr: *anyopaque, requests: []const be.DraftRequest) !void {
        _ = ptr;
        for (requests) |r| if (r.depth != 0) return error.FlintHasNoDraftHead;
    }

    fn heldFn(_: *anyopaque, _: *lanes.Stream, out: []u32) !void {
        if (out.len > 0) return error.FlintDraftsNotHeld;
    }

    fn releaseFn(ptr: *anyopaque, stream: *lanes.Stream) void {
        const self = cast(ptr);
        self.drain() catch |err| std.log.err("flint: a step in flight failed at release: {s}", .{@errorName(err)});
        if (self.caches.fetchRemove(stream)) |entry| {
            self.held -= st.cacheBytes(&self.model.config, entry.value.capacity);
            entry.value.deinit();
            self.gpa.destroy(entry.value);
            if (self.room) |r| r.fit(r.ctx, self.options.budget - self.held, null, &.{});
        }
    }

    fn prepareFeatures(ptr: *anyopaque, taps: []const u32) !void {
        const self = cast(ptr);
        if (self.caches.count() != 0 or !std.mem.eql(u32, taps, @import("../qwen27/dflash/config.zig").flint_tap_ids[0..5])) return error.UnsupportedFlintTaps;
        self.tap_ids = @import("../qwen27/dflash/config.zig").flint_tap_ids[0..5];
    }

    fn featuresFn(ptr: *anyopaque, stream: *lanes.Stream, taps: []const u32, start: u64, count: u32) !be.Features {
        const self = cast(ptr);
        try self.drain();
        const cache = try self.cacheOf(stream);
        if (!std.mem.eql(u32, taps, self.tap_ids) or start < cache.tap_start or count > cache.tap_rows or start - cache.tap_start > cache.tap_rows - count) return error.StaleFlintTaps;
        const bytes = self.model.config.hidden * 5 * 2;
        return .{ .buffer = @intFromPtr((cache.taps orelse return error.FlintTapsNotPrepared).contents()), .offset = (start - cache.tap_start) * bytes, .rows = count, .row_bytes = bytes, .space = .host, .dtype = .bf16 };
    }

    fn prefillFeatures(ptr: *anyopaque, stream: *lanes.Stream, taps: []const u32, sink: be.FeatureSink) !void {
        const self = cast(ptr);
        if (!std.mem.eql(u32, taps, self.tap_ids)) return error.UnsupportedFlintTaps;
        try self.prefill(stream, sink);
    }

    /// The prompt cache's Snapshots functions: a stream's caches at a mark, copied on the GPU.
    pub fn snapBytes(ptr: *anyopaque, at: u32) u64 {
        return snapshot.bytes(&cast(ptr).model.config, at);
    }

    pub fn snapSave(ptr: *anyopaque, owner: ?*anyopaque, at: u32) anyerror!*anyopaque {
        const self = cast(ptr);
        try self.drain();
        const cache = try self.cacheOf(@ptrCast(@alignCast(owner orelse return error.NoStream)));
        if (cache.len != at) return error.SnapshotOutOfStep;
        const snap = try self.gpa.create(snapshot.Snap);
        errdefer self.gpa.destroy(snap);
        snap.* = try snapshot.save(self.model.device, self.model.queue, &self.model.config, cache);
        return snap;
    }

    pub fn snapRestore(_: *anyopaque, _: ?*anyopaque, _: *anyopaque) anyerror!void {
        return error.BackendRestores; // the prompt pass restores, before its first chunk
    }

    pub fn snapDrop(ptr: *anyopaque, saved: *anyopaque) void {
        const snap: *snapshot.Snap = @ptrCast(@alignCast(saved));
        snap.free();
        cast(ptr).gpa.destroy(snap);
    }
};

fn gib(bytes: u64) f64 {
    return @as(f64, @floatFromInt(bytes)) / (1 << 30);
}

fn logit(word: u16) f32 {
    return @bitCast(@as(u32, word) << 16);
}

fn argmax(l: []const u16) usize {
    var best: usize = 0;
    for (l, 0..) |w, i| if (logit(w) > logit(l[best])) {
        best = i;
    };
    return best;
}

test "Flint DFlash2 taps reject stale stream rows and another tap order before reading buffers" {
    const a = std.testing.allocator;
    var m: Model = undefined;
    m.config = .{};
    var b: Metal = undefined;
    b.gpa = a;
    b.model = &m;
    b.caches = .empty;
    defer b.caches.deinit(a);
    b.flights = .empty;
    b.tap_ids = @import("../qwen27/dflash/config.zig").flint_tap_ids[0..5];
    var stream: lanes.Stream = undefined;
    var cache: st.Cache = undefined;
    cache.tap_start = 100;
    cache.tap_rows = 3;
    try b.caches.put(a, &stream, &cache);
    try std.testing.expectError(error.StaleFlintTaps, Metal.featuresFn(&b, &stream, b.tap_ids, 99, 1));
    try std.testing.expectError(error.StaleFlintTaps, Metal.featuresFn(&b, &stream, b.tap_ids, 102, 2));
    try std.testing.expectError(error.StaleFlintTaps, Metal.featuresFn(&b, &stream, &.{5}, 100, 1));
    try std.testing.expectError(error.UnsupportedFlintTaps, Metal.prepareFeatures(&b, b.tap_ids));
}
