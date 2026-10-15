//! Step-5 behind the lane engine's backend: windows of the pending token and the MTP layers' drafts, a cache a stream.
const std = @import("std");
const mtl = @import("metal");
const lanes = @import("lanes");
const st = @import("state.zig");
const fwd = @import("forward.zig");
const Model = @import("model.zig").Model;
const draw = @import("../qwen3_5/backend.zig").draw;
const be = lanes.backend;

/// The longest context a stream may take, the streams at once, and the bytes every stream's caches may hold.
pub const Options = struct { capacity: usize, streams: u32 = 4, budget: u64 };

/// Drafts a window holds at most: one a MTP layer.
pub const max_drafts = 3;

pub const Metal = struct {
    gpa: std.mem.Allocator,
    model: *Model,
    options: Options,
    scratch: st.Scratch, // decode rounds
    prompt: st.Scratch, // prompt chunks
    caches: std.AutoHashMapUnmanaged(*lanes.Stream, *st.Cache) = .empty,
    held: u64 = 0, // bytes the streams' caches hold

    pub fn init(gpa: std.mem.Allocator, model: *Model, options: Options) !*Metal {
        const out = try gpa.create(Metal);
        errdefer gpa.destroy(out);
        var scratch = try st.Scratch.init(model.device, &model.config, options.capacity, st.window_rows);
        errdefer scratch.deinit();
        out.* = .{ .gpa = gpa, .model = model, .options = options, .scratch = scratch, .prompt = try st.Scratch.init(model.device, &model.config, options.capacity, st.prompt_rows) };
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
        self.gpa.destroy(self);
    }

    pub fn facts(self: *const Metal) lanes.Model {
        // rows are independent in every kernel of a decode window, so any width to 16 gives one-row steps' bits
        const mtp = self.model.config.mtp > 0;
        return .{ .exact_width = st.window_rows, .mtp = mtp, .speculate = mtp, .speculate_early = false, .drafts = if (mtp) max_drafts else 1, .max_streams = self.options.streams, .streams_exact = true };
    }

    pub fn backend(self: *Metal) be.Backend {
        return .{ .ptr = self, .vtable = &.{ .prefill = prefillFn, .first = firstFn, .first_masked = firstMaskedFn, .queue = queueFn, .read = readFn, .verify = verifyFn, .keep = keepFn, .draft = draftFn, .release = releaseFn, .first_row = firstRowFn, .held = heldFn } };
    }

    fn cast(ptr: *anyopaque) *Metal {
        return @ptrCast(@alignCast(ptr));
    }

    fn cacheOf(self: *Metal, stream: *lanes.Stream) !*st.Cache {
        return self.caches.get(stream) orelse error.UnknownStep5Stream;
    }

    fn logits(self: *Metal, cache: *st.Cache) []const u16 {
        return cache.logits.slice(u16, self.model.config.vocab);
    }

    /// Run `ids` at the cache's end and keep them; with MTP the first MTP layer absorbs each row with a known next.
    fn feed(self: *Metal, cache: *st.Cache, ids: []const u32, images: []const fwd.Span) !void {
        const c = &self.model.config;
        var at: usize = 0;
        while (at < ids.len) {
            const end = @min(ids.len, at + st.prompt_rows);
            const s = if (end - at > st.window_rows) &self.prompt else &self.scratch;
            const n = end - at;
            try fwd.run(self.model, s, cache, ids[at..end], .{ .hidden = .{ .buf = s.hid }, .images = images });
            const start = cache.len;
            cache.len += n;
            const last: [*]const u8 = s.hid.contents() + (n - 1) * c.hidden * 4;
            @memcpy(cache.hidden.contents()[0 .. c.hidden * 4], last[0 .. c.hidden * 4]);
            const known = if (end < ids.len) n else n - 1; // rows whose next token is in ids
            if (c.mtp > 0 and known > 0) try fwd.mtp(self.model, s, cache, 0, .{ .buf = s.hid }, ids[at + 1 .. at + 1 + known], start);
            at = end;
        }
    }

    fn prefillFn(ptr: *anyopaque, stream: *lanes.Stream) !void {
        const self = cast(ptr);
        const ids = stream.prompt();
        if (ids.len == 0 or ids.len + stream.max_new > self.options.capacity or self.caches.contains(stream)) return error.InvalidStep5Prompt;
        const capacity = ids.len + stream.max_new + 1;
        const need = st.cacheBytes(&self.model.config, capacity);
        if (self.held + need > self.options.budget) {
            std.log.warn("step5: a {d}-token stream needs {d:.1} GiB of caches; {d:.1} of {d:.1} GiB are held", .{ capacity, gib(need), gib(self.held), gib(self.options.budget) });
            return error.Step5CachesFull;
        }
        const cache = try self.gpa.create(st.Cache);
        errdefer self.gpa.destroy(cache);
        cache.* = try st.Cache.init(self.model.device, &self.model.config, capacity);
        errdefer cache.deinit();
        self.held += need;
        errdefer self.held -= need;
        var rows: ?mtl.Buffer = null;
        defer if (rows) |r| r.deinit();
        const spans = try self.encodeImages(stream.images, &rows);
        defer self.gpa.free(spans);
        try self.feed(cache, ids, spans);
        try self.caches.put(self.gpa, stream, cache);
    }

    /// The tower's rows for every image piece of a prompt, into one buffer, as spans over their placeholders.
    fn encodeImages(self: *Metal, images: []const lanes.stream.Image, rows: *?mtl.Buffer) ![]fwd.Span {
        const spans = try self.gpa.alloc(fwd.Span, images.len);
        errdefer self.gpa.free(spans);
        if (images.len == 0) return spans;
        const v = self.model.vision orelse return error.Step5HasNoImageTower;
        const h: usize = self.model.config.hidden;
        var total: usize = 0;
        for (images) |im| total += im.tokens;
        rows.* = try self.model.device.buffer(total * h * 2, mtl.ResourceOptions.private);
        var off: usize = 0;
        for (images, spans) |im, *sp| {
            const g2 = ((im.gh - 1) / 2 + 1 - 1) / 2 + 1; // two stride-2 convs
            if (im.gh != im.gw or im.tokens != g2 * g2) return error.Step5ImageGrid;
            sp.* = .{ .at = im.at, .n = im.tokens, .rows = .{ .buf = rows.*.?, .off = off * h * 2 } };
            const pool = mtl.objc.Pool.push();
            defer pool.pop();
            const cb = self.model.queue.commandBuffer();
            const e = cb.compute(.serial);
            try v.encode(e, im.pixels, im.gh, im.gw, .{ .buf = rows.*.?, .off = off * h * 2 });
            e.end();
            cb.commit();
            cb.wait();
            if (cb.failure()) |msg| {
                std.log.err("step5 image tower: {s}", .{msg});
                return error.GpuFailed;
            }
            off += im.tokens;
        }
        return spans;
    }

    /// A prompt chunk, a decode row, the MTP layers and the tower once: the first request skips the GPU's first touch.
    pub fn warm(self: *Metal) !void {
        const c = &self.model.config;
        var cache = try st.Cache.init(self.model.device, c, st.prompt_rows + 2);
        defer cache.deinit();
        var ids: [st.prompt_rows]u32 = undefined;
        for (&ids, 0..) |*t, i| t.* = @intCast(1000 + i * 37 % 20000);
        try self.feed(&cache, &ids, &.{});
        try fwd.run(self.model, &self.scratch, &cache, &.{ids[0]}, .{ .all = true, .hidden = .{ .buf = cache.hidden } });
        if (self.model.vision) |v| {
            const rows = try self.model.device.buffer(169 * @as(usize, c.hidden) * 2, mtl.ResourceOptions.private);
            defer rows.deinit();
            for ([_]u32{ 52, 36 }) |g| {
                const px = try self.gpa.alloc(f32, @as(usize, g) * g * 588);
                defer self.gpa.free(px);
                @memset(px, 0);
                const pool = mtl.objc.Pool.push();
                defer pool.pop();
                const cb = self.model.queue.commandBuffer();
                const e = cb.compute(.serial);
                try v.encode(e, px, g, g, .{ .buf = rows });
                e.end();
                cb.commit();
                cb.wait();
            }
        }
    }

    fn firstFn(ptr: *anyopaque, stream: *lanes.Stream, position: u64) !u64 {
        const self = cast(ptr);
        const cache = try self.cacheOf(stream);
        if (position != cache.len) return error.InvalidStep5Position;
        return try draw(self.gpa, self.logits(cache), stream.sampling, position, &.{});
    }

    fn firstMaskedFn(ptr: *anyopaque, stream: *lanes.Stream, position: u64, mask: []const u32) !u64 {
        const self = cast(ptr);
        const cache = try self.cacheOf(stream);
        if (position != cache.len) return error.InvalidStep5Position;
        return try draw(self.gpa, self.logits(cache), stream.sampling, position, mask);
    }

    fn queueFn(ptr: *anyopaque, stream: *lanes.Stream, f: be.Feed, position: u64) !u64 {
        const self = cast(ptr);
        const cache = try self.cacheOf(stream);
        if (position != cache.len + 1) return error.InvalidStep5Position;
        const token: u32 = switch (f) {
            .value => |v| v,
            .handle => |h| @intCast(h),
        };
        try fwd.run(self.model, &self.scratch, cache, &.{token}, .{ .hidden = .{ .buf = cache.hidden } });
        cache.len += 1;
        return try draw(self.gpa, self.logits(cache), stream.sampling, position, &.{});
    }

    fn firstRowFn(ptr: *anyopaque, stream: *lanes.Stream) !lanes.LogprobRow {
        const self = cast(ptr);
        const l = self.logits(try self.cacheOf(stream));
        return row(l, @intCast(argmax(l)), stream.logprobs orelse 0);
    }

    fn readFn(ptr: *anyopaque, handle: u64) !u32 {
        if (handle >= cast(ptr).model.config.vocab) return error.InvalidStep5Token;
        return @intCast(handle);
    }

    /// Each window's pending token and drafts in one forward, every row drawn and kept until `keep` rolls back.
    fn verifyFn(ptr: *anyopaque, windows: []const be.Window, out: []be.Verified) !void {
        const self = cast(ptr);
        const c = &self.model.config;
        const words = (c.vocab + 31) / 32;
        if (windows.len != out.len) return error.InvalidStep5Windows;
        for (windows, out) |w, o| {
            const rows = w.rows();
            if (rows > st.window_rows or w.parents != null or o.sampled.len != rows or o.drafts.len != rows - 1) return error.UnsupportedStep5Window;
            const cache = try self.cacheOf(w.stream);
            if (w.held > cache.nheld) return error.Step5DraftsNotHeld;
            for (w.positions, 0..) |p, r| if (p != cache.len + r + 1) return error.InvalidStep5Position;
            var ids: [st.window_rows]u32 = undefined;
            ids[0] = w.pending;
            @memcpy(ids[1..][0..w.held], cache.held[0..w.held]);
            @memcpy(ids[1 + w.held ..][0..w.tokens.len], w.tokens);
            try fwd.run(self.model, &self.scratch, cache, ids[0..rows], .{ .all = true, .hidden = .{ .buf = cache.hidden } });
            cache.len += rows;
            cache.nheld = 0;
            @memcpy(o.drafts, ids[1..rows]);
            const all = self.scratch.logits_all.slice(u16, rows * c.vocab);
            for (0..rows) |r| {
                const l = all[r * c.vocab ..][0..c.vocab];
                const mask = if (w.masks.len >= (r + 1) * words) w.masks[r * words ..][0..words] else &.{};
                o.sampled[r] = try draw(self.gpa, l, w.stream.sampling, w.positions[r], mask);
                if (o.rows.len == rows) o.rows[r] = row(l, o.sampled[r], w.stream.logprobs orelse 0);
            }
            // the last kept row's logits are the next first draw's (the lane core keeps every row here)
            @memcpy(cache.logits.slice(u16, c.vocab), all[(rows - 1) * c.vocab ..][0..c.vocab]);
        }
    }

    fn keepFn(ptr: *anyopaque, windows: []const be.Window, paths: []const []const u32) !void {
        const self = cast(ptr);
        for (windows, paths) |w, path| {
            const cache = try self.cacheOf(w.stream);
            for (path, 0..) |r, i| if (r != i) return error.UnsupportedStep5Tree;
            if (path.len == 0 or path.len > w.rows() or w.rows() > cache.len) return error.UnsupportedStep5Window;
            cache.len = cache.len - w.rows() + path.len;
        }
    }

    /// MTP layer 0 absorbs the kept rows and drafts from the last; further drafts chain the next layers, greedy.
    fn draftFn(ptr: *anyopaque, requests: []const be.DraftRequest) !void {
        const self = cast(ptr);
        const c = &self.model.config;
        const s = &self.scratch;
        for (requests) |r| {
            const cache = try self.cacheOf(r.stream);
            cache.nheld = 0;
            if (c.mtp == 0) {
                if (r.depth != 0) return error.Step5CheckpointHasNoMtp;
                continue;
            }
            var follow: [st.window_rows]u32 = undefined;
            var kept: usize = 1;
            if (r.rows) |rows| {
                kept = rows.len;
                for (rows, 0..) |x, i| if (x != i) return error.UnsupportedStep5Tree;
                if (kept == 0 or kept > st.window_rows or r.follow.len < kept) return error.UnsupportedStep5Window;
                @memcpy(follow[0..kept], r.follow[0..kept]);
            } else if (r.follow.len > 0) {
                follow[0] = r.follow[r.follow.len - 1];
            } else {
                const f = r.first orelse return error.NoFirstToken;
                follow[0] = switch (f) {
                    .value => |v| v,
                    .handle => |h| @intCast(h),
                };
            }
            // kept row i sits at position (r.position - 1 - kept) + i, its residual in cache.hidden's row i
            const first: usize = @intCast(r.position - 1 - kept);
            try fwd.mtp(self.model, s, cache, 0, .{ .buf = cache.hidden }, follow[0..kept], first);
            const depth = @min(r.depth, max_drafts);
            var pos = first + kept - 1;
            var k: usize = 0;
            while (k < depth) : (k += 1) {
                const d: u32 = @intCast(argmax(s.logits_all.slice(u16, c.vocab)));
                cache.held[k] = d;
                cache.nheld = k + 1;
                if (k + 1 == depth) break;
                pos += 1;
                @memcpy(s.chain.contents()[0 .. c.hidden * 4], (s.hid.contents() + (if (k == 0) kept - 1 else 0) * c.hidden * 4)[0 .. c.hidden * 4]);
                try fwd.mtp(self.model, s, cache, @min(k + 1, c.mtp - 1), .{ .buf = s.chain }, &.{d}, pos);
            }
        }
    }

    /// The held drafts in window order, for a grammar to walk.
    fn heldFn(ptr: *anyopaque, stream: *lanes.Stream, out: []u32) !void {
        const cache = try cast(ptr).cacheOf(stream);
        if (out.len > cache.nheld) return error.Step5DraftsNotHeld;
        @memcpy(out, cache.held[0..out.len]);
    }

    fn releaseFn(ptr: *anyopaque, stream: *lanes.Stream) void {
        const self = cast(ptr);
        if (self.caches.fetchRemove(stream)) |entry| {
            self.held -= st.cacheBytes(&self.model.config, entry.value.capacity);
            entry.value.deinit();
            self.gpa.destroy(entry.value);
        }
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

/// The logprob row of bf16 logits `l` for `pick`: an fp64 normalizer, `count` best ids, ties to the lower id.
pub fn row(l: []const u16, pick: u32, count: u8) lanes.LogprobRow {
    const top = lanes.logprob.max_top;
    var most: f64 = -std.math.inf(f64);
    for (l) |w| most = @max(most, logit(w));
    var sum: f64 = 0;
    for (l) |w| sum += @exp(@as(f64, logit(w)) - most);
    const lse: f32 = @floatCast(most + @log(sum));
    var r: lanes.LogprobRow = .{ .token = pick, .logprob = logit(l[pick]) - lse, .count = top };
    var filled: usize = 0;
    for (l, 0..) |w, id| {
        const x = logit(w);
        if (filled == top and x <= r.logprobs[top - 1] + lse) continue;
        var at = @min(filled, top - 1);
        while (at > 0 and x > r.logprobs[at - 1] + lse) : (at -= 1) {
            r.ids[at] = r.ids[at - 1];
            r.logprobs[at] = r.logprobs[at - 1];
        }
        r.ids[at] = @intCast(id);
        r.logprobs[at] = x - lse;
        filled = @min(filled + 1, top);
    }
    r.count = @min(count, top);
    return r;
}

test "a logprob row: the normalizer, the best ids in order, ties to the lower id" {
    var l: [40]u16 = undefined;
    for (&l, 0..) |*w, i| w.* = @intCast(@as(u32, @bitCast(@as(f32, @floatFromInt(i % 7)))) >> 16);
    const r = row(&l, 6, 20);
    try std.testing.expectEqual(@as(u32, 6), r.ids[0]);
    try std.testing.expectEqual(@as(u32, 13), r.ids[1]);
    try std.testing.expect(r.logprobs[0] == r.logprob and r.logprobs[19] <= r.logprobs[18]);
}
