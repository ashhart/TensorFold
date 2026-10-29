//! Gemma 4 on CUDA behind lanes.Backend: a slot a stream, a chunk a prompt step, one forward a round of windows.
const std = @import("std");
const cuda = @import("cuda");
const lanes = @import("lanes");
const cfg = @import("cuda_config.zig");
const fwd = @import("cuda_forward.zig");
const kern = @import("cuda_kernels.zig");
const snap = @import("cuda_snap.zig");
const cv = @import("cuda_vision.zig");

const be = lanes.backend;
const Stream = lanes.Stream;
const Pick = cuda.tree_round.Pick;

/// Prompt rows a step takes: the forward's chunk grid, from position 0.
pub const prompt_rows = fwd.prompt_rows;
/// Window rows a round takes over every stream.
pub const round_rows = fwd.round_rows;
/// The widest window a stream verifies: one-row steps and every wider window give a row the same bits.
pub const max_window = 16;
/// Drawn tokens a handle names (a ring; read before the round loop could wrap it).
const ring = 1024;

const Lane = struct {
    slot: usize,
    len: u64 = 0, // the cache length: positions [0, len) hold this stream's keys
    base: u64 = 0, // the cache length the last verify started from
    rows: std.ArrayList(u32) = .empty, // the last verify's row tokens
    last: u64, // the prompt's last row's soft-capped bf16 logits [V] (the first draw reads them)
    rule: u64, // sample.cu's Rule, when the stream samples
    sampled: bool = false,
    fill: ?usize = null, // a prompt pass going a chunk at a time: where its next chunk starts
    mark: ?usize = null, // a stepped pass's mark kept at its next step, after the drafter absorbed the chunk
    tap: Tap = .{},
    images: ?cuda.DeviceBuffer = null, // the prompt's images through the embedder, while its pass runs
};

/// A drafter's context beside a kept state: its bytes at a position, and its copy into the state.
pub const Keeper = struct { ctx: *anyopaque, bytes: *const fn (*anyopaque, usize) u64, keep: *const fn (*anyopaque, *const Stream, *snap.Saved) anyerror!void };

/// A stream's rows in the last forward: rows [row, row + rows) at positions [start, start + rows) while gen holds.
const Tap = struct { row: usize = 0, start: u64 = 0, rows: usize = 0, gen: u64 = 0 };

pub const Gemma = struct {
    gpa: std.mem.Allocator,
    d: *const cuda.Driver,
    m: *fwd.Model,
    k: *const kern.Kernels,
    s: cuda.Stream,
    lanes: std.AutoHashMapUnmanaged(*const Stream, Lane) = .empty,
    free: []bool, // slot i is free
    tok: cuda.DeviceBuffer, // drawn tokens (u32): the handle ring, then a round's rows
    picks: cuda.DeviceBuffer, // argmax picks (Pick), the same entries
    greedy: []bool, // entry e of the ring (and the round's rows) is a pick, not a draw
    meta: cuda.DeviceBuffer, // a draw's first position minus one, a window each
    slot_bufs: []cuda.DeviceBuffer, // a slot's last-row logits, then its rule: two a slot
    next: u64 = 0,
    gen: u64 = 1, // forwards run: a tap is current while its gen is this
    tapped: bool = false, // a drafter reads the last layer's rows (prepare_features)
    vision: ?*cv.Vision = null, // --vision: the embedder a prompt's images go through
    keeper: ?Keeper = null, // a drafter whose context a kept prompt state carries (DFlash's)

    /// The device bytes `init` allocates: the token and pick rings, metas, each slot's last row and rule.
    pub fn openBytes(slots: usize) usize {
        return (ring + round_rows) * (4 + @sizeOf(Pick)) + round_rows * 16 + slots * (cfg.vocab * 2 + @sizeOf(cuda.sample.Rule));
    }

    pub fn init(gpa: std.mem.Allocator, d: *const cuda.Driver, m: *fwd.Model, k: *const kern.Kernels, s: cuda.Stream) !Gemma {
        const free = try gpa.alloc(bool, m.slots);
        errdefer gpa.free(free);
        @memset(free, true);
        const greedy = try gpa.alloc(bool, ring + round_rows);
        errdefer gpa.free(greedy);
        @memset(greedy, true);
        var tok = try cuda.DeviceBuffer.alloc(d, (ring + round_rows) * 4);
        errdefer tok.free();
        var picks = try cuda.DeviceBuffer.alloc(d, (ring + round_rows) * @sizeOf(Pick));
        errdefer picks.free();
        var meta = try cuda.DeviceBuffer.alloc(d, round_rows * 16);
        errdefer meta.free();
        const bufs = try gpa.alloc(cuda.DeviceBuffer, 2 * m.slots);
        errdefer gpa.free(bufs);
        var made: usize = 0;
        errdefer for (bufs[0..made]) |*b| b.free();
        for (bufs, 0..) |*b, i| {
            b.* = try cuda.DeviceBuffer.alloc(d, if (i % 2 == 0) cfg.vocab * 2 else @sizeOf(cuda.sample.Rule));
            made += 1;
        }
        return .{ .gpa = gpa, .d = d, .m = m, .k = k, .s = s, .free = free, .tok = tok, .picks = picks, .greedy = greedy, .meta = meta, .slot_bufs = bufs };
    }

    pub fn deinit(x: *Gemma) void {
        var it = x.lanes.valueIterator();
        while (it.next()) |l| l.rows.deinit(x.gpa);
        x.lanes.deinit(x.gpa);
        for (x.slot_bufs) |*b| b.free();
        x.gpa.free(x.slot_bufs);
        x.gpa.free(x.free);
        x.gpa.free(x.greedy);
        x.tok.free();
        x.picks.free();
        x.meta.free();
    }

    pub fn backend(x: *Gemma) be.Backend {
        return .{ .ptr = x, .vtable = &vtable };
    }

    const vtable: be.Backend.VTable = .{ .prefill = prefill, .prefill_step = prefillStep, .first = first, .queue = queue, .read = read, .verify = verify, .keep = keep, .draft = draft, .release = release, .reports_fill = true, .prepare_features = prepareFeatures, .features = features, .prefill_features = prefillFeatures };

    /// The drafter's taps: those layers' outputs (before the next norm), kept by every forward from now on.
    fn prepareFeatures(ptr: *anyopaque, taps: []const u32) anyerror!void {
        const x = self(ptr);
        if (x.lanes.count() != 0) return error.TapsAfterStreams;
        try x.m.setTaps(taps);
        x.tapped = true;
    }

    fn tapsMatch(x: *const Gemma, taps: []const u32) bool {
        return x.tapped and std.mem.eql(u32, taps, x.m.taps[0..x.m.tap_count]);
    }

    /// Rows [start, start + count) of the stream's last forward, on the device until the next forward.
    fn features(ptr: *anyopaque, s: *Stream, taps: []const u32, start: u64, count: u32) anyerror!be.Features {
        const x = self(ptr);
        if (!x.tapsMatch(taps)) return error.GemmaTapNotServed;
        const l = x.lanes.getPtr(s) orelse return error.UnknownStream;
        return x.featuresOf(l.tap, start, count);
    }

    fn featuresOf(x: *const Gemma, t: Tap, start: u64, count: u32) !be.Features {
        if (t.gen != x.gen or start < t.start or start + count > t.start + t.rows or count == 0) return error.RowsNotHeld;
        const width = x.m.tap_count * x.m.geo.hidden * 2;
        return .{ .buffer = x.m.taps_buf.?.ptr, .offset = (t.row + start - t.start) * width, .rows = count, .row_bytes = @intCast(width), .space = .device, .dtype = .bf16 };
    }

    /// The prompt pass, `sink` taking each chunk's rows as the chunk is made.
    fn prefillFeatures(ptr: *anyopaque, s: *Stream, taps: []const u32, sink: be.FeatureSink) anyerror!void {
        const x = self(ptr);
        if (!x.tapsMatch(taps)) return error.GemmaTapNotServed;
        const l = try x.opening(s);
        errdefer x.dropLane(s);
        while (true) {
            const at = l.fill.?;
            const done = try x.chunk(s, l, true);
            try sink.take(sink.ptr, s, try x.featuresOf(l.tap, at, @intCast(l.tap.rows)), at);
            x.marked(s, at + l.tap.rows);
            if (done) return;
        }
    }

    /// One forward of the model, its rows' taps current until the next.
    pub fn run(x: *Gemma, chains: []const fwd.Chain, prompt: bool, head0: usize, heads: usize) !void {
        x.gen += 1;
        try x.m.forward(x.k, x.s, chains, prompt, head0, heads);
    }

    /// The stream's cache slot and length, for a drafter reading its caches.
    pub fn slotOf(x: *const Gemma, s: *const Stream) ?struct { slot: usize, len: u64 } {
        const l = x.lanes.getPtr(s) orelse return null;
        return .{ .slot = l.slot, .len = l.len };
    }

    /// The round loop's facts: no draft head; host copies verified in windows up to 16 rows, streams sharing a forward.
    pub fn facts(x: *const Gemma) lanes.Model {
        return .{ .exact_width = max_window, .gpu_tokens = false, .mtp = false, .speculate = false, .hidden_rows = true, .batch_rows = round_rows, .max_streams = @intCast(@min(x.m.slots, round_rows)) };
    }

    fn self(ptr: *anyopaque) *Gemma {
        return @ptrCast(@alignCast(ptr));
    }

    fn upload(x: *Gemma, dst: u64, bytes: []const u8) !void {
        try x.d.check(x.d.api.cuMemcpyHtoDAsync_v2(dst, bytes.ptr, bytes.len, x.s.handle), "cuMemcpyHtoDAsync");
    }

    /// Rows [row0, row0 + rows) of bf16 `logits` into entries [at, at + rows), row r at `position0 + r`.
    fn drawRows(x: *Gemma, l: *const Lane, logits: u64, row0: usize, rows: usize, position0: u64, at: usize, meta_at: usize) !void {
        const v = cfg.vocab;
        const src = logits + row0 * v * 2;
        @memset(x.greedy[at..][0..rows], !l.sampled);
        if (!l.sampled) return x.k.round.argmax(x.s, src, x.picks.ptr + at * @sizeOf(Pick), rows, v, v);
        const meta = [4]i32{ @intCast(position0 - 1), 0, 0, 0 };
        const mp = x.meta.ptr + meta_at * 16;
        try x.upload(mp, std.mem.asBytes(&meta));
        try x.k.draws.draw(x.s, src, v, l.rule, mp, 0, x.tok.ptr + at * 4, rows, null, null);
    }

    /// Entries [at, at + out.len) as tokens, after the stream's work; a pick over non-finite logits is an error.
    fn tokens(x: *Gemma, at: usize, out: []u32) !void {
        try x.s.synchronize();
        var picks: [round_rows]Pick = undefined;
        try x.tok.download(at * 4, std.mem.sliceAsBytes(out));
        try x.picks.download(at * @sizeOf(Pick), std.mem.sliceAsBytes(picks[0..out.len]));
        for (out, picks[0..out.len], x.greedy[at..][0..out.len]) |*t, p, g| if (g) {
            if (p.nonfinite != 0) return error.NonFiniteLogits;
            t.* = p.token;
        };
    }

    pub fn saveState(x: *Gemma, owner: ?*anyopaque, at: u32) !*snap.Saved {
        const s: *const Stream = @ptrCast(@alignCast(owner orelse return error.NoOwner));
        const l = x.lanes.getPtr(s) orelse return error.UnknownStream;
        if ((l.fill orelse l.len) != at) return error.NotAtMark; // a state is copied only where the pass stands
        const k = try snap.save(x.gpa, x.m, x.s, l.slot, at);
        errdefer snap.drop(x.gpa, x.s, k);
        if (x.keeper) |kp| try kp.keep(kp.ctx, s, k);
        return k;
    }

    pub fn restoreState(x: *Gemma, owner: ?*anyopaque, k: *const snap.Saved) !void {
        const s: *const Stream = @ptrCast(@alignCast(owner orelse return error.NoOwner));
        const l = x.lanes.getPtr(s) orelse return error.UnknownStream;
        try snap.restore(x.m, x.s, l.slot, k);
        l.len = k.at;
    }

    pub fn dropState(x: *Gemma, k: *snap.Saved) void {
        snap.drop(x.gpa, x.s, k);
    }

    fn freeSlot(x: *const Gemma) ?usize {
        return std.mem.indexOfScalar(bool, x.free, true);
    }

    /// A prompt pass begins: a slot, the stream's rule, a kept state restored; where its first chunk starts.
    fn opening(x: *Gemma, s: *Stream) !*Lane {
        const p = s.prompt();
        if (p.len == 0) return error.EmptyPrompt;
        if (p.len + @as(usize, s.max_new) + max_window > x.m.context) return error.PastContext;
        const got = try x.lanes.getOrPut(x.gpa, s);
        if (!got.found_existing) {
            const slot = x.freeSlot() orelse {
                x.lanes.removeByPtr(got.key_ptr);
                return error.NoFreeSlot;
            };
            x.free[slot] = false;
            got.value_ptr.* = .{ .slot = slot, .last = x.slot_bufs[2 * slot].ptr, .rule = x.slot_bufs[2 * slot + 1].ptr };
        }
        const l = got.value_ptr;
        l.sampled = false;
        l.mark = null;
        if (s.sampling) |sp| if (sp.temperature > 0) {
            const r = cuda.sample.Rule.from(lanes.gpu_rule.rule(sp));
            try x.upload(l.rule, std.mem.asBytes(&r));
            l.sampled = true;
        };
        s.cached = 0;
        if (s.images.len > 0) try x.encodeImages(s, l);
        l.fill = x.resumeAt(s, l, p.len);
        s.fill_at.store(@intCast(l.fill.?), .monotonic);
        return l;
    }

    /// Every image of the prompt through the embedder into the lane's buffer, in prompt order.
    fn encodeImages(x: *Gemma, s: *Stream, l: *Lane) !void {
        const v = x.vision orelse return error.GemmaServedWithoutVision;
        const p = s.prompt();
        var rows: usize = 0;
        var end: usize = 0;
        for (s.images) |im| {
            if (im.at < end or im.at + im.tokens > p.len or im.tokens * image_pool * image_pool != im.gh * im.gw) return error.GemmaImageRows;
            end = im.at + im.tokens;
            rows += im.tokens;
        }
        if (l.images) |*b| b.free();
        l.images = null;
        const width = x.m.geo.hidden * 2;
        var buf = try cuda.DeviceBuffer.alloc(x.d, rows * width);
        errdefer buf.free();
        var at: usize = 0;
        for (s.images) |im| {
            try v.encode(x.d, x.s, im.pixels, im.gh, im.gw, buf.ptr + at * width);
            at += im.tokens;
        }
        l.images = buf;
    }

    const image_pool = @import("cuda_image.zig").pool;

    /// Whether position `p` cuts through an image (strictly inside its rows).
    fn insideImage(s: *const Stream, p: usize) bool {
        for (s.images) |im| if (im.at < p and p < im.at + im.tokens) return true;
        return false;
    }

    /// The kept state's position, restored into the slot (0: none, or not this prompt's).
    fn resumeAt(x: *Gemma, s: *Stream, l: *const Lane, len: usize) usize {
        const saved = s.reuse.saved orelse return 0;
        const k: *const snap.Saved = @ptrCast(@alignCast(saved));
        // a drafter that keeps context resumes only a state that carries it
        if (k.at != s.reuse.at or k.at == 0 or k.at >= len or insideImage(s, k.at) or (x.keeper != null and s.drafts and k.draft == null)) {
            s.reuse_failed = true;
            return 0;
        }
        snap.restore(x.m, x.s, l.slot, k) catch {
            s.reuse_failed = true;
            return 0;
        };
        s.cached = k.at;
        return k.at;
    }

    /// One chunk from `l.fill` to the next grid line or mark; true once the prompt's last row's logits are written.
    fn chunk(x: *Gemma, s: *Stream, l: *Lane, streamed: bool) !bool {
        const p = s.prompt();
        const at = l.fill.?;
        if (s.isCancelled()) return error.Cancelled;
        var end = @min(p.len, (at / prompt_rows + 1) * prompt_rows);
        for (s.reuse.marks) |mk| if (mk > at and mk < end and !insideImage(s, mk)) {
            end = mk;
        };
        // never through an image: its rows see each other, so a chunk holds all of them (one is at most 280)
        for (s.images) |im| if (im.at < end and end < im.at + im.tokens) {
            end = if (im.at > at) im.at else im.at + im.tokens;
        };
        var injects: [8]fwd.Inject = undefined;
        var n: usize = 0;
        var off: usize = 0;
        for (s.images) |im| {
            if (im.at >= at and im.at + im.tokens <= end) {
                if (n == injects.len) return error.GemmaImagesInChunk;
                injects[n] = .{ .row = im.at - at, .rows = im.tokens, .src = l.images.?.ptr + off * x.m.geo.hidden * 2 };
                n += 1;
            }
            off += im.tokens;
        }
        const done = end == p.len;
        try x.run(&.{.{ .slot = l.slot, .p0 = at, .tokens = p[at..end], .images = injects[0..n] }}, true, end - at - 1, @intFromBool(done));
        l.tap = .{ .row = 0, .start = at, .rows = end - at, .gen = x.gen };
        if (done) try x.d.check(x.d.api.cuMemcpyDtoDAsync_v2(l.last, x.m.s.logits, cfg.vocab * 2, x.s.handle), "cuMemcpyDtoDAsync");
        s.fill_at.store(@intCast(end), .monotonic);
        l.fill = if (done) null else end;
        if (done) {
            l.len = p.len;
            if (l.images) |*b| {
                try x.s.synchronize(); // the last chunk read them
                b.free();
                l.images = null;
            }
        }
        if (!streamed) x.marked(s, end); // else after the drafter's absorb of the chunk (prefillFeatures, prefillStep)
        return done;
    }

    /// The prompt cache's mark at `end`: the state there is copied (with a drafter's context, once it holds it).
    fn marked(x: *Gemma, s: *Stream, end: usize) void {
        _ = x;
        if (s.reuse.hook) |hk| if (std.mem.indexOfScalar(u32, s.reuse.marks, @intCast(end)) != null) hk.at(hk.ptr, s, @intCast(end));
    }

    fn prefill(ptr: *anyopaque, s: *Stream) anyerror!void {
        const x = self(ptr);
        const l = try x.opening(s);
        errdefer x.dropLane(s);
        while (!try x.chunk(s, l, false)) {}
    }

    fn prefillStep(ptr: *anyopaque, s: *Stream) anyerror!bool {
        const x = self(ptr);
        const going = if (x.lanes.getPtr(s)) |l| l.fill != null else false;
        const l = if (going) x.lanes.getPtr(s).? else try x.opening(s);
        errdefer x.dropLane(s);
        if (l.mark) |at| x.marked(s, at); // the drafter's wrapper absorbed the last step's rows: its context is kept too
        l.mark = null;
        const later = x.keeper != null and s.drafts;
        const done = try x.chunk(s, l, later);
        if (later) {
            if (done) x.marked(s, s.prompt().len) else l.mark = l.fill;
        }
        return done;
    }

    fn handle(x: *Gemma) usize {
        const h = x.next;
        x.next += 1;
        return @intCast(h % ring);
    }

    fn first(ptr: *anyopaque, s: *Stream, position: u64) anyerror!u64 {
        const x = self(ptr);
        const l = x.lanes.getPtr(s) orelse return error.UnknownStream;
        if (position != l.len or l.fill != null) return error.PositionMismatch;
        const h = x.next;
        try x.drawRows(l, l.last, 0, 1, position, x.handle(), 0);
        return h;
    }

    fn value(x: *Gemma, feed: be.Feed) !u32 {
        return switch (feed) {
            .value => |v| v,
            .handle => |h| read(x, h),
        };
    }

    fn queue(ptr: *anyopaque, s: *Stream, feed: be.Feed, position: u64) anyerror!u64 {
        const x = self(ptr);
        const l = x.lanes.getPtr(s) orelse return error.UnknownStream;
        const t = try x.value(feed);
        if (position != l.len + 1) return error.PositionMismatch;
        try x.run(&.{.{ .slot = l.slot, .p0 = l.len, .tokens = &.{t} }}, false, 0, 1);
        l.tap = .{ .row = 0, .start = l.len, .rows = 1, .gen = x.gen };
        l.len += 1;
        const h = x.next;
        try x.drawRows(l, x.m.s.logits, 0, 1, position, x.handle(), 0);
        return h;
    }

    fn read(ptr: *anyopaque, h: u64) anyerror!u32 {
        const x = self(ptr);
        if (h >= x.next or x.next - h > ring) return error.NoSuchToken;
        var out: [1]u32 = undefined;
        try x.tokens(@intCast(h % ring), &out);
        return out[0];
    }

    fn verify(ptr: *anyopaque, windows: []const be.Window, out: []be.Verified) anyerror!void {
        const x = self(ptr);
        const chains = try x.gpa.alloc(fwd.Chain, windows.len);
        defer x.gpa.free(chains);
        var total: usize = 0;
        for (windows, chains) |w, *c| {
            if (w.held != 0) return error.NoHeldDrafts; // no draft head: every draft comes from the host
            if (w.parents != null) return error.ChainsOnly;
            if (w.masks.len != 0) return error.StructuresUnsupported;
            const l = x.lanes.getPtr(w.stream) orelse return error.UnknownStream;
            if (w.rows() > max_window) return error.WindowTooWide;
            l.base = l.len;
            l.rows.clearRetainingCapacity();
            try l.rows.append(x.gpa, w.pending);
            try l.rows.appendSlice(x.gpa, w.tokens);
            for (w.positions, 0..) |pos, r| if (pos != l.base + r + 1) return error.PositionMismatch;
            c.* = .{ .slot = l.slot, .p0 = l.base, .tokens = l.rows.items };
            total += l.rows.items.len;
        }
        if (total > round_rows) return error.RoundTooWide;
        try x.run(chains, false, 0, total);
        var at: usize = 0;
        for (windows, 0..) |w, i| {
            const l = x.lanes.getPtr(w.stream).?;
            l.tap = .{ .row = at, .start = l.base, .rows = l.rows.items.len, .gen = x.gen };
            try x.drawRows(l, x.m.s.logits, at, l.rows.items.len, w.positions[0], ring + at, i);
            at += l.rows.items.len;
        }
        var got: [round_rows]u32 = undefined;
        try x.tokens(ring, got[0..total]);
        at = 0;
        for (windows, out) |w, o| {
            const l = x.lanes.getPtr(w.stream).?;
            const n = l.rows.items.len;
            @memcpy(o.sampled[0..n], got[at..][0..n]);
            @memcpy(o.drafts[0 .. n - 1], l.rows.items[1..]);
            l.len = l.base + n; // a chain keeps every row unless rolled back
            at += n;
        }
    }

    fn keep(ptr: *anyopaque, windows: []const be.Window, paths: []const []const u32) anyerror!void {
        const x = self(ptr);
        for (windows, paths) |w, path| {
            const l = x.lanes.getPtr(w.stream) orelse return error.UnknownStream;
            if (path.len == 0 or path.len > l.rows.items.len) return error.EmptyPath;
            for (path, 0..) |r, i| if (r != i) return error.ChainsOnly; // a chain keeps a prefix of its rows
            l.len = l.base + path.len; // caches are positional: later rows overwrite what the rejected ones wrote
        }
    }

    /// No draft head is loaded (#1048 steps 10/11): absorbing is a no-op, a request for drafts is refused by name.
    fn draft(_: *anyopaque, requests: []const be.DraftRequest) anyerror!void {
        for (requests) |r| if (r.depth != 0 or r.lanes != null) return error.NoDraftHead;
    }

    fn dropLane(x: *Gemma, s: *const Stream) void {
        if (x.lanes.fetchRemove(s)) |kv| {
            var l = kv.value;
            x.free[l.slot] = true;
            l.rows.deinit(x.gpa);
            if (l.images) |*b| {
                x.s.synchronize() catch {};
                b.free();
            }
        }
    }

    fn release(ptr: *anyopaque, s: *Stream) void {
        self(ptr).dropLane(s);
    }
};
