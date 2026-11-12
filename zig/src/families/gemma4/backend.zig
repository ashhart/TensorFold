//! Gemma 4 dense behind the lane engine's backend: windows of up to 16 rows, 128 rows of streams a forward, taps.
const std = @import("std");
const lanes = @import("lanes");
const mtl = @import("metal");
const cfg = @import("config.zig");
const st = @import("state.zig");
const fwd = @import("forward.zig");
const prompt = @import("prompt.zig");
const snapshot = @import("snapshot.zig");
const Model = @import("model.zig").Model;
const steps = @import("prefill_steps.zig");
const draw = @import("../qwen3_5/backend.zig").draw;
const be = lanes.backend;

pub const Options = struct { capacity: usize, streams: u32 = 8 };

pub const Metal = struct {
    gpa: std.mem.Allocator,
    model: *Model,
    options: Options,
    scratch: st.Scratch,
    /// Prompt chunks' activations: 512 rows on core affine_mm and the tensor-unit attention.
    prompt_scratch: prompt.Scratch,
    caches: std.AutoHashMapUnmanaged(*lanes.Stream, *st.Cache) = .empty,
    /// Prompt passes a chunk a call (prefill_step) between rounds, by stream.
    fills: std.AutoHashMapUnmanaged(*lanes.Stream, Fill) = .empty,
    /// Prompt chunks run on `prompt_scratch`: a stepped chunk's tapped rows there are current while this matches.
    prompt_gen: u64 = 0,
    follow: ?mtl.ResidencySet.Follow = null,
    /// Forwards run on `scratch`: a stream's tapped rows there are current while its tap's gen matches.
    gen: u64 = 0,
    /// A window's forward ms by rows, timed by measure() (the depth rule prices drafts with them); none until then.
    costs: [st.max_rows]lanes.config.Cost = undefined,
    timed: usize = 0,
    /// Wider forwards' ms by total rows (several streams' windows), timed at load; the rule interpolates between.
    shared: [shared_widths.len]lanes.config.Cost = undefined,
    shared_timed: usize = 0,
    io: std.Io,
    trace: bool = false, // GEMMA_DRAFT_TRACE: each verify's rows and wall time on stderr
    /// Whether the drafter keeps a context with a kept state (DFlash does); the MTP assistant reads the target's alone.
    resume_needs_draft: bool = true,

    pub fn init(gpa: std.mem.Allocator, io: std.Io, model: *Model, options: Options) !*Metal {
        const out = try gpa.create(Metal);
        errdefer gpa.destroy(out);
        var scratch = try st.Scratch.init(model.device, model.config.geo, options.capacity);
        errdefer scratch.deinit();
        out.* = .{ .gpa = gpa, .io = io, .trace = std.c.getenv("GEMMA_DRAFT_TRACE") != null, .model = model, .options = options, .scratch = scratch, .prompt_scratch = try prompt.Scratch.init(model.device, model.config.geo, options.capacity) };
        return out;
    }

    pub fn deinit(self: *Metal) void {
        self.setResident(null);
        var it = self.caches.valueIterator();
        while (it.next()) |cache| {
            cache.*.deinit();
            self.gpa.destroy(cache.*);
        }
        self.caches.deinit(self.gpa);
        var fills = self.fills.valueIterator();
        while (fills.next()) |f| {
            f.spans.deinit(self.gpa);
            if (f.rows) |b| b.deinit();
        }
        self.fills.deinit(self.gpa);
        self.scratch.deinit();
        self.prompt_scratch.deinit();
        self.gpa.destroy(self);
    }

    pub fn setResident(self: *Metal, hook: ?mtl.ResidencySet.Follow) void {
        if (self.follow) |f| {
            mtl.resident.fields.apply(&self.scratch, f, false);
            mtl.resident.fields.apply(&self.prompt_scratch, f, false);
        }
        self.follow = hook;
        if (hook) |f| {
            mtl.resident.fields.apply(&self.scratch, f, true);
            mtl.resident.fields.apply(&self.prompt_scratch, f, true);
        }
        var it = self.caches.valueIterator();
        while (it.next()) |cache| cache.*.setResident(hook);
    }

    pub fn createCache(self: *Metal, capacity: usize) !*st.Cache {
        const cache = try self.gpa.create(st.Cache);
        errdefer self.gpa.destroy(cache);
        cache.* = try st.Cache.init(self.model.device, self.model.config.geo, capacity);
        cache.setResident(self.follow);
        return cache;
    }

    /// Every kernel keeps a row's bits at any window width and beside any other stream's rows (tf-gemma4-glue).
    pub fn facts(self: *const Metal) lanes.Model {
        return .{ .exact_width = st.max_rows, .hidden_rows = true, .batch_rows = st.prompt_rows, .max_streams = self.options.streams, .streams_exact = true, .window_costs = self.costs[0..self.timed], .shared_costs = self.shared[0..self.shared_timed] };
    }

    /// A window of each width 1..16 at position 0 on a throwaway cache, the second of two runs timed.
    pub fn measure(self: *Metal) !void {
        const io = self.io;
        var cache = try st.Cache.init(self.model.device, self.model.config.geo, 256);
        defer cache.deinit();
        var ids: [st.max_rows]u32 = @splat(2);
        for (1..st.max_rows + 1) |rows| {
            var ms: f64 = 0;
            for (0..2) |run| {
                cache.len = 0;
                const t0 = std.Io.Clock.awake.now(io);
                try fwd.run(self.model, &self.scratch, &.{.{ .cache = &cache, .rows = rows }}, ids[0..rows]);
                if (run == 1) ms = @as(f64, @floatFromInt(t0.durationTo(std.Io.Clock.awake.now(io)).toNanoseconds())) / 1e6;
            }
            self.costs[rows - 1] = .{ .width = @intCast(rows), .ms = ms };
        }
        self.timed = st.max_rows;
        for (&self.shared, shared_widths) |*cost, rows| cost.* = .{ .width = @intCast(rows), .ms = try self.timeRows(rows) };
        self.shared_timed = shared_widths.len;
        self.gen += 1;
    }

    /// The forward's ms at `rows` rows (up to 128) of one stream, the second of two runs.
    pub fn timeRows(self: *Metal, rows: usize) !f64 {
        const io = self.io;
        var cache = try st.Cache.init(self.model.device, self.model.config.geo, 256);
        defer cache.deinit();
        var ids: [st.prompt_rows]u32 = @splat(2);
        var ms: f64 = 0;
        for (0..2) |run| {
            cache.len = 0;
            const t0 = std.Io.Clock.awake.now(io);
            try fwd.run(self.model, &self.scratch, &.{.{ .cache = &cache, .rows = rows }}, ids[0..rows]);
            if (run == 1) ms = @as(f64, @floatFromInt(t0.durationTo(std.Io.Clock.awake.now(io)).toNanoseconds())) / 1e6;
        }
        self.gen += 1;
        return ms;
    }

    pub fn backend(self: *Metal) be.Backend {
        return .{ .ptr = self, .vtable = &.{ .prefill = prefillFn, .prefill_step = prefillStepFn, .cancel_prefill = cancelPrefillFn, .reports_fill = true, .prepare_features = prepareFn, .features = featuresFn, .prefill_features = prefillFeaturesFn, .first = firstFn, .first_masked = firstMaskedFn, .queue = queueFn, .read = readFn, .verify = verifyFn, .keep = keepFn, .draft = draftFn, .release = releaseFn } };
    }

    fn cast(ptr: *anyopaque) *Metal {
        return @ptrCast(@alignCast(ptr));
    }

    fn cacheOf(self: *Metal, stream: *lanes.Stream) !*st.Cache {
        return self.caches.get(stream) orelse error.UnknownGemmaStream;
    }

    /// The last row of the scratch logits into the stream's cache.
    fn keepLast(self: *Metal, cache: *st.Cache, row: usize) void {
        @memcpy(cache.logits.slice(u16, cfg.vocab), self.scratch.logits.slice(u16, st.prompt_rows * cfg.vocab)[row * cfg.vocab ..][0..cfg.vocab]);
    }

    fn prefillFn(ptr: *anyopaque, stream: *lanes.Stream) !void {
        return cast(ptr).prefillWith(stream, null);
    }

    fn prefillFeaturesFn(ptr: *anyopaque, stream: *lanes.Stream, taps: []const u32, sink: be.FeatureSink) !void {
        const self = cast(ptr);
        if (taps.len != self.model.taps) return error.UnpreparedGemmaTaps;
        return self.prefillWith(stream, sink);
    }

    /// A tapped row's bytes: every tap's hidden row, bf16.
    fn tapBytes(self: *const Metal) usize {
        return self.model.taps * self.model.config.geo.hidden * 2;
    }

    /// The drafter's layers: both scratches keep their rows from now on.
    fn prepareFn(ptr: *anyopaque, taps: []const u32) !void {
        const self = cast(ptr);
        if (self.scratch.taps != null) return if (taps.len == self.model.taps) self.model.setTaps(taps) else error.BadGemmaTaps;
        try self.model.setTaps(taps);
        const shared = mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked;
        const decode_taps = try self.model.device.buffer(st.prompt_rows * self.tapBytes(), shared);
        errdefer decode_taps.deinit();
        const prompt_taps = try self.model.device.buffer(st.chunk_rows * self.tapBytes(), shared);
        self.scratch.taps = decode_taps;
        self.prompt_scratch.taps = prompt_taps;
        if (self.follow) |f| {
            f.join(f.ctx, decode_taps);
            f.join(f.ctx, prompt_taps);
        }
    }

    /// Rows [start, start + count) of the stream's latest tapped rows, on the device.
    fn featuresFn(ptr: *anyopaque, stream: *lanes.Stream, taps: []const u32, start: u64, count: u32) !be.Features {
        const self = cast(ptr);
        const cache = try self.cacheOf(stream);
        const t = cache.tap;
        // a stepped prompt's latest chunk (prefillStepFn), while no other prompt chunk has rewritten the prompt scratch
        const c = cache.chunk_tap;
        const in_tap = t.buf != null and steps.holds(.{ .start = t.start, .rows = t.rows, .gen = t.gen }, self.gen, start, count);
        if (!in_tap and taps.len == self.model.taps and c.buf != null and steps.holds(.{ .start = c.start, .rows = c.rows, .gen = c.gen }, self.prompt_gen, start, count))
            return .{ .buffer = @intFromPtr(c.buf.?.id), .offset = c.off + (start - c.start) * self.tapBytes(), .rows = count, .row_bytes = @intCast(self.tapBytes()), .space = .device, .dtype = .bf16 };
        if (taps.len != self.model.taps or t.buf == null) return error.UnpreparedGemmaTaps;
        if (t.gen) |g| if (g != self.gen) return error.StaleGemmaTaps;
        if (count == 0 or start < t.start or start + count > t.start + t.rows) return error.GemmaTapsNotHeld;
        const width = self.tapBytes();
        return .{ .buffer = @intFromPtr(t.buf.?.id), .offset = t.off + (start - t.start) * width, .rows = count, .row_bytes = @intCast(width), .space = .device, .dtype = .bf16 };
    }

    /// A stream's prompt pass in progress (prefill_step): its next row, its images' rows and their spans.
    const Fill = struct { pass: steps.Fill, spans: std.ArrayList(prompt.Span) = .empty, rows: ?mtl.Buffer = null };

    /// One chunk's work for `steps.step`: the pass, the drafter's rows (a sink, or `later` via featuresFn), marks.
    const Run = struct {
        m: *Metal,
        cache: *st.Cache,
        spans: []const prompt.Span,
        sink: ?be.FeatureSink,
        later: bool,

        pub fn run(r: Run, stream: *lanes.Stream, at: usize, end: usize) !void {
            const self = r.m;
            const ids = stream.prompt();
            try prompt.run(self.model, &self.prompt_scratch, r.cache, ids[at..end], r.spans);
            self.prompt_gen += 1;
            if (r.sink != null or r.later) {
                const width = self.tapBytes();
                const rows = self.prompt_scratch.taps.?;
                if (r.sink) |k| try k.take(k.ptr, stream, .{ .buffer = @intFromPtr(rows.id), .rows = @intCast(end - at), .row_bytes = @intCast(width), .space = .device, .dtype = .bf16 }, at);
                // a stepped pass's rows stay in the prompt scratch until the next chunk runs: featuresFn serves them
                if (r.later) r.cache.chunk_tap = .{ .buf = rows, .start = at, .rows = end - at, .gen = self.prompt_gen };
                if (end == ids.len) { // the prompt's last row, kept for the first draft (another prompt may reuse the scratch)
                    const last = try self.model.device.buffer(width, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
                    r.cache.setLastTap(last);
                    @memcpy(last.slice(u8, width), rows.slice(u8, st.chunk_rows * width)[(end - 1 - at) * width ..][0..width]);
                    r.cache.tap = .{ .buf = last, .start = ids.len - 1, .rows = 1 };
                }
            }
            lanes.prefill_widths.completed(self.gpa, stream, end - at);
            if (end == ids.len) @memcpy(r.cache.logits.slice(u16, cfg.vocab), self.prompt_scratch.logits.slice(u16, cfg.vocab));
        }

        pub fn keep(_: Run, stream: *lanes.Stream, at: u32) void {
            if (stream.reuse.hook) |k| k.at(k.ptr, stream, at);
        }
    };

    fn prefillWith(self: *Metal, stream: *lanes.Stream, sink: ?be.FeatureSink) !void {
        const f = try self.opening(stream);
        errdefer self.drop(stream);
        const r: Run = .{ .m = self, .cache = try self.cacheOf(stream), .spans = f.spans.items, .sink = sink, .later = false };
        while (!try steps.step(r, stream, &f.pass, st.chunk_rows, f.spans.items, false)) {}
        self.closing(stream);
    }

    /// The stream's cache made (a kept state restored into it) and its images encoded: its pass from `cache.len`.
    fn opening(self: *Metal, stream: *lanes.Stream) !*Fill {
        const ids = stream.prompt();
        if (ids.len == 0 or ids.len + stream.max_new > self.options.capacity or self.caches.contains(stream)) return error.InvalidGemmaPrompt;
        const capacity = @min(self.options.capacity, (ids.len + stream.max_new + st.max_rows + 63) / 64 * 64);
        const cache = try self.createCache(capacity);
        errdefer self.gpa.destroy(cache);
        errdefer cache.deinit();
        try self.caches.put(self.gpa, stream, cache); // a mark's save finds it there
        errdefer _ = self.caches.remove(stream);
        stream.cached = 0;
        if (stream.reuse.saved) |saved| { // a kept state of this prompt's prefix: the pass starts there
            const snap: *const snapshot.Snap = @ptrCast(@alignCast(saved));
            var inside = false; // an image's rows see each other: a state inside one is not resumed
            for (stream.images) |im| inside = inside or (im.at < snap.at and snap.at < im.at + im.tokens);
            // a drafting stream resumes only a state kept with its drafter's context, if any (dflash.zig keepLane)
            const drafter_less = stream.drafts and self.prompt_scratch.taps != null and snap.draft == null and self.resume_needs_draft;
            if (snap.at < ids.len and !inside and !drafter_less) {
                if (snapshot.restore(self.gpa, self.model.queue, snap, cache)) stream.cached = snap.at else |_| stream.reuse_failed = true;
            } else stream.reuse_failed = true;
        }
        var f: Fill = .{ .pass = .{ .at = cache.len } };
        errdefer f.spans.deinit(self.gpa);
        f.rows = try self.encodeImages(stream, cache.len, &f.spans);
        errdefer if (f.rows) |b| b.deinit();
        try self.fills.put(self.gpa, stream, f);
        stream.fill_at.store(@intCast(cache.len), .monotonic); // what the restore kept: 0 when none
        return self.fills.getPtr(stream).?;
    }

    /// The pass is over, or the stream left mid-pass: its images' rows and spans freed.
    fn closing(self: *Metal, stream: *lanes.Stream) void {
        var entry = self.fills.fetchRemove(stream) orelse return;
        entry.value.spans.deinit(self.gpa);
        if (entry.value.rows) |b| b.deinit();
    }

    /// A pass that failed: its fill and its cache go, as the whole pass's error left them.
    fn drop(self: *Metal, stream: *lanes.Stream) void {
        self.closing(stream);
        if (self.caches.fetchRemove(stream)) |entry| {
            entry.value.deinit();
            self.gpa.destroy(entry.value);
        }
    }

    /// One chunk a call at the whole pass's cuts (`max_rows` is never set: this family declares no chunk_invariant).
    fn prefillStepFn(ptr: *anyopaque, stream: *lanes.Stream, _: u32) !bool {
        const self = cast(ptr);
        const f = self.fills.getPtr(stream) orelse try self.opening(stream);
        errdefer self.drop(stream);
        // a drafting stream's wrapper absorbs each chunk after the call (featuresFn): its marks keep at the next call
        const later = stream.drafts and self.prompt_scratch.taps != null;
        const r: Run = .{ .m = self, .cache = try self.cacheOf(stream), .spans = f.spans.items, .sink = null, .later = later };
        const done = try steps.step(r, stream, &f.pass, st.chunk_rows, f.spans.items, later);
        if (done) self.closing(stream);
        return done;
    }

    /// The host cancels between chunks: the last landed chunk's state is kept (no prompt work runs).
    fn cancelPrefillFn(ptr: *anyopaque, stream: *lanes.Stream) !void {
        const self = cast(ptr);
        if (self.fills.get(stream)) |f| stream.keepInterruptedPrefill(@intCast(f.pass.at));
    }

    /// The stream's images from `from` on through the tower into one buffer of rows (null: none), a span each.
    fn encodeImages(self: *Metal, stream: *const lanes.Stream, from: usize, spans: *std.ArrayList(prompt.Span)) !?mtl.Buffer {
        var total: usize = 0;
        for (stream.images) |im| {
            if (im.at + im.tokens > stream.prompt().len or im.tokens * 9 != im.gh * im.gw) return error.InvalidGemmaImage;
            if (im.at >= from) total += im.tokens; // one before `from` is in the restored state
        }
        if (total == 0) return null;
        const v = self.model.vision orelse return error.NoGemmaVision;
        const hidden = self.model.config.geo.hidden;
        const buf = try self.model.device.buffer(total * hidden * 2, mtl.ResourceOptions.private | mtl.ResourceOptions.untracked);
        errdefer buf.deinit();
        var row: usize = 0;
        for (stream.images) |im| { // one command buffer an image: the tower's host-written inputs are one image's
            if (im.at < from) continue;
            const rows: prompt.Span = .{ .at = im.at, .n = im.tokens, .rows = .{ .buf = buf, .off = row * hidden * 2 } };
            const cb = self.model.queue.commandBuffer();
            const e = cb.compute(.serial);
            try v.encode(e, im.pixels, im.gh, im.gw, rows.rows);
            e.end();
            cb.commit();
            cb.wait();
            if (cb.failure()) |msg| {
                std.log.err("gemma4: image tower failed: {s}", .{msg});
                return error.GpuFailed;
            }
            try spans.append(self.gpa, rows);
            row += im.tokens;
        }
        return buf;
    }

    fn firstFn(ptr: *anyopaque, stream: *lanes.Stream, position: u64) !u64 {
        const self = cast(ptr);
        const cache = try self.cacheOf(stream);
        if (position != cache.len) return error.InvalidGemmaPosition;
        return try draw(self.gpa, cache.logits.slice(u16, cfg.vocab), stream.sampling, position, &.{});
    }

    fn firstMaskedFn(ptr: *anyopaque, stream: *lanes.Stream, position: u64, mask: []const u32) !u64 {
        const self = cast(ptr);
        const cache = try self.cacheOf(stream);
        if (position != cache.len) return error.InvalidGemmaPosition;
        return try draw(self.gpa, cache.logits.slice(u16, cfg.vocab), stream.sampling, position, mask);
    }

    fn queueFn(ptr: *anyopaque, stream: *lanes.Stream, feed: be.Feed, position: u64) !u64 {
        const self = cast(ptr);
        const cache = try self.cacheOf(stream);
        if (position != cache.len + 1) return error.InvalidGemmaPosition;
        const token: u32 = switch (feed) {
            .value => |v| v,
            .handle => |h| @intCast(h),
        };
        cache.from = cache.len;
        try fwd.run(self.model, &self.scratch, &.{.{ .cache = cache, .rows = 1 }}, &.{token});
        self.gen += 1;
        self.keepLast(cache, 0);
        return try draw(self.gpa, cache.logits.slice(u16, cfg.vocab), stream.sampling, position, &.{});
    }

    fn readFn(_: *anyopaque, handle: u64) !u32 {
        if (handle >= cfg.vocab) return error.InvalidGemmaToken;
        return @intCast(handle);
    }

    fn verifyFn(ptr: *anyopaque, windows: []const be.Window, out: []be.Verified) !void {
        const self = cast(ptr);
        var segments: [st.prompt_rows]fwd.Segment = undefined;
        var ids: [st.prompt_rows]u32 = undefined;
        var total: usize = 0;
        if (windows.len > segments.len or windows.len != out.len) return error.InvalidGemmaWindows;
        for (windows, 0..) |w, i| {
            const rows = w.rows();
            if (w.parents != null or w.held != 0 or rows > st.max_rows or total + rows > st.prompt_rows or w.positions.len != rows) return error.UnsupportedGemmaWindow;
            if (w.masks.len != 0 and w.masks.len != rows * mask_words) return error.InvalidGemmaMasks;
            const cache = try self.cacheOf(w.stream);
            for (w.positions, 0..) |p, r| if (p != cache.len + r + 1) return error.InvalidGemmaPosition;
            cache.from = cache.len;
            segments[i] = .{ .cache = cache, .rows = rows };
            ids[total] = w.pending;
            @memcpy(ids[total + 1 ..][0..w.tokens.len], w.tokens);
            total += rows;
        }
        const t0 = std.Io.Clock.awake.now(self.io);
        try fwd.run(self.model, &self.scratch, segments[0..windows.len], ids[0..total]);
        if (self.trace) std.debug.print("verify of {d} streams, {d} rows: {d:.2} ms\n", .{ windows.len, total, @as(f64, @floatFromInt(t0.durationTo(std.Io.Clock.awake.now(self.io)).toNanoseconds())) / 1e6 });
        self.gen += 1;
        if (self.scratch.taps) |taps| {
            var row: usize = 0;
            for (segments[0..windows.len]) |seg| {
                seg.cache.tap = .{ .buf = taps, .off = row * self.tapBytes(), .start = seg.cache.from, .rows = seg.rows, .gen = self.gen };
                row += seg.rows;
            }
        }
        const logits = self.scratch.logits.slice(u16, total * cfg.vocab);
        const picks = self.scratch.picks.slice(Pick, total);
        var base: usize = 0;
        for (windows, out) |w, o| {
            if (o.sampled.len != w.rows() or o.drafts.len != w.tokens.len) return error.InvalidGemmaOutputs;
            const greedy = w.masks.len == 0 and (w.stream.sampling == null or w.stream.sampling.?.temperature <= 0);
            for (o.sampled, w.positions, 0..) |*t, p, r| {
                if (greedy) { // the GPU's argmax, the same token the host draw finds
                    const pick = picks[base + r];
                    if (pick.nonfinite != 0 or pick.token >= cfg.vocab) return error.NonfiniteGemmaLogits;
                    t.* = pick.token;
                    continue;
                }
                const mask = if (w.masks.len > 0) w.masks[r * mask_words ..][0..mask_words] else &.{};
                t.* = try draw(self.gpa, logits[(base + r) * cfg.vocab ..][0..cfg.vocab], w.stream.sampling, p, mask);
            }
            @memcpy(o.drafts, w.tokens);
            base += w.rows();
        }
    }

    /// A chain's kept rows: the cache rolls back to where the verify began plus them; later rows are rewritten.
    fn keepFn(ptr: *anyopaque, windows: []const be.Window, paths: []const []const u32) !void {
        const self = cast(ptr);
        for (windows, paths) |w, path| {
            const cache = try self.cacheOf(w.stream);
            for (path, 0..) |r, j| if (r != j) return error.UnsupportedGemmaTree;
            if (cache.from + path.len > cache.len) return error.InvalidGemmaKeep;
            cache.len = cache.from + path.len;
        }
    }

    fn draftFn(_: *anyopaque, requests: []const be.DraftRequest) !void {
        for (requests) |r| if (r.depth != 0) return error.GemmaCheckpointHasNoDraftHead;
    }

    /// The prompt cache's Snapshots functions: a stream's caches at a mark, copied on the GPU.
    pub fn snapBytes(ptr: *anyopaque, at: u32) u64 {
        return snapshot.bytes(cast(ptr).model.config.geo, at);
    }

    pub fn snapSave(ptr: *anyopaque, owner: ?*anyopaque, at: u32) anyerror!*anyopaque {
        const self = cast(ptr);
        const cache = try self.cacheOf(@ptrCast(@alignCast(owner orelse return error.NoStream)));
        if (cache.len != at) return error.SnapshotOutOfStep;
        const snap = try self.gpa.create(snapshot.Snap);
        errdefer self.gpa.destroy(snap);
        snap.* = try snapshot.save(self.gpa, self.model.device, self.model.queue, cache);
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

    fn releaseFn(ptr: *anyopaque, stream: *lanes.Stream) void {
        const self = cast(ptr);
        self.closing(stream); // a stream that left mid-pass
        if (self.caches.fetchRemove(stream)) |entry| {
            entry.value.deinit();
            self.gpa.destroy(entry.value);
        }
    }
};

/// Total rows a shared round is timed at beyond one window (the forward's cost grows past 16 rows: tensor-op compute).
const shared_widths = [_]usize{ 32, 48, 64, 96, 128 };

/// A row's allowed-token bits (structured output): one bit a logit.
const mask_words = (cfg.vocab + 31) / 32;

const Pick = @import("tree_round_gpu").contract.Pick;

test "Gemma Metal steps: the backend steps its prompt passes, reports their fill and keeps a cancelled one's chunks" {
    var b: Metal = undefined;
    const v = b.backend().vtable;
    // the lane host fills it a chunk a round (open.zig fills); the drafted wrapper steps too (features, reports_fill)
    try std.testing.expect(v.prefill_step != null and v.reports_fill and v.cancel_prefill != null and v.features != null);
    try std.testing.expect(!b.backend().chunk_invariant); // the host never caps its chunks (lane_host_fill.zig capRows)
}

test "zero grant: Gemma Metal assistant and DFlash retain tapped context through a long skip" {
    var b: Metal = undefined;
    b.options = .{ .capacity = 512, .streams = 16 };
    b.timed = 0;
    b.shared_timed = 0;
    try lanes.shared_draft_check.check(b.facts(), true);
}

test "shared head costs: Gemma Metal assistant and DFlash retain costed copy grants" {
    var b: Metal = undefined;
    b.options = .{ .capacity = 512, .streams = 16 };
    b.timed = 0;
    b.shared_timed = 0;
    try @import("metal_head_cost_test.zig").check(b.facts());
}
