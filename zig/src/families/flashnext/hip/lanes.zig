//! The lane core's HIP backend for Flash Next: each stream its own caches, each window verified and kept exactly.
const std = @import("std");
const hip = @import("hip");
const lanes = @import("lanes");
const serve = @import("hip_serve");
const Engine = @import("engine.zig").Engine;
const cache = @import("cache.zig");
const forward = @import("forward.zig");
const head_mod = @import("head.zig");
const load = @import("load.zig");
const worker = @import("worker.zig");
const snapshot = @import("snapshot.zig");
const pc = @import("engine_api").prompt_cache;

const be = lanes.backend;
const draw = serve.draw;

/// Drawn tokens a handle names, newest last.
const ring = 1024;
/// Drafts the head holds for a stream at most.
pub const max_depth = cache.max_window - 1;

pub const Fill = struct { at: usize, stops: [16]u32 = undefined, count: usize = 0, next: usize = 0, from: usize = 0, kept0: u64 = 0 };

pub const Lane = struct {
    caches: cache.Caches,
    /// The head's caches (none on a rank that does not draft).
    head: ?cache.Caches,
    /// The residual streams of the last forward's rows: the head's input.
    streams: hip.DeviceBuffer,
    held: [max_depth]u32 = undefined,
    held_n: usize = 0,
    /// The prompt pass: the row it reached and the cuts where the prompt cache keeps a state (null once it is in).
    fill: ?Fill = null,
    /// The last verify's tokens until its keep.
    pending: ?struct { tokens: [cache.max_window]u32, rows: usize } = null,
    first_row: ?[]u16 = null,
    id: u32 = 0,
};

/// The engine this thread's HIP calls go to.
threadlocal var bound: ?*Engine = null;

pub const Fx = struct {
    gpa: std.mem.Allocator,
    e: *Engine,
    streams: usize,
    lanes: std.AutoHashMapUnmanaged(*const lanes.Stream, *Lane) = .empty,
    drawn: [ring]u32 = undefined,
    next: u64 = 0,
    /// Tensor parallelism: the other ranks get every step first (worker.zig).
    link: ?hip.collective.Link = null,
    msg: std.ArrayList(u32) = .empty,
    next_id: u32 = 0,
    /// A one a stream for every row the head takes at once (a prompt chunk), and the draft vocabulary's token ids.
    ones: hip.DeviceBuffer,
    draft_ids: []u32,
    drafts: u32,
    /// The kept prompt states (null: no reuse), and the slot the next one takes on every rank.
    store: ?pc.Store = null,
    next_slot: u32 = 0,

    pub fn init(gpa: std.mem.Allocator, e: *Engine, streams: usize) !*Fx {
        const f = try gpa.create(Fx);
        errdefer gpa.destroy(f);
        const n = @max(e.o.chunk, cache.max_window) * e.cfg.hc_count;
        const one: u16 = if (e.act == .f16) 0x3c00 else 0x3f80;
        const host = try gpa.alloc(u16, n);
        defer gpa.free(host);
        @memset(host, one);
        f.* = .{ .gpa = gpa, .e = e, .streams = streams, .ones = try hip.DeviceBuffer.fromHost(&e.driver, std.mem.sliceAsBytes(host)), .draft_ids = &.{}, .drafts = @min(max_depth, e.o.policy.mtp.drafts) };
        errdefer f.ones.free();
        if (e.weights.model.mtp != null) f.draft_ids = try load.draftIds(gpa);
        return f;
    }

    pub fn deinit(f: *Fx) void {
        if (f.store) |*st| st.deinit();
        if (f.link != null) f.send(&.{@backingInt(worker.Op.stop)}) catch {};
        f.e.stream.synchronize() catch {};
        var it = f.lanes.valueIterator();
        while (it.next()) |l| f.free(l.*);
        f.lanes.deinit(f.gpa);
        f.msg.deinit(f.gpa);
        f.ones.free();
        f.gpa.free(f.draft_ids);
        f.gpa.destroy(f);
    }

    pub fn withLink(f: *Fx, link: hip.collective.Link) void {
        f.link = link;
    }

    /// Keep prompt states in `budget` host bytes on the prompt kernels' 64-row grid, each a token past its rows.
    pub fn keepPrompts(f: *Fx, budget: u64) void {
        if (budget == 0) return;
        f.store = pc.Store.init(f.gpa, .{ .ptr = f, .vtable = &.{ .bytes = snapBytes, .save = snapSave, .restore = snapRestore, .drop = snapDrop } }, .{ .lookahead = 1, .planned = true, .grid = 64 }, budget);
    }

    fn snapBytes(ptr: *anyopaque, at: u32) u64 {
        const f: *Fx = @ptrCast(@alignCast(ptr));
        var buf: [64]bool = undefined;
        const kinds = f.e.kinds(&buf) catch return std.math.maxInt(u64);
        const main = cache.Layout.of(f.e.weights.model.dims, kinds, 64, f.e.act).stateBytes(at);
        const head = if (f.e.weights.model.mtp) |h| cache.Layout.of(h.dims, &.{false}, 64, f.e.act).stateBytes(at) else 0;
        return main + head;
    }

    fn snapSave(ptr: *anyopaque, owner: ?*anyopaque, at: u32) anyerror!pc.Saved {
        const f = of(ptr);
        const lane: *Lane = @ptrCast(@alignCast(owner orelse return error.NoOwner));
        const saved = try snapshot.save(f.gpa, &f.e.driver, f.e.stream, &lane.caches, if (lane.head) |*h| h else null, at, f.e.cfg.index_ratio);
        saved.slot = f.next_slot;
        f.next_slot += 1;
        if (f.link != null) try f.send(&.{ @backingInt(worker.Op.snap), lane.id, saved.slot, at });
        return saved;
    }

    fn snapRestore(ptr: *anyopaque, owner: ?*anyopaque, saved: pc.Saved) anyerror!void {
        const f = of(ptr);
        const lane: *Lane = @ptrCast(@alignCast(owner orelse return error.NoOwner));
        const s: *snapshot.Saved = @ptrCast(@alignCast(saved));
        if (f.link != null) try f.send(&.{ @backingInt(worker.Op.restore), lane.id, s.slot });
        try snapshot.restore(f.gpa, &f.e.driver, f.e.stream, s, &lane.caches, if (lane.head) |*h| h else null, f.e.cfg.index_ratio);
    }

    fn snapDrop(ptr: *anyopaque, saved: pc.Saved) void {
        const f: *Fx = @ptrCast(@alignCast(ptr));
        const s: *snapshot.Saved = @ptrCast(@alignCast(saved));
        if (f.link != null) f.send(&.{ @backingInt(worker.Op.drop), s.slot }) catch {};
        snapshot.drop(f.gpa, s);
    }

    fn send(f: *Fx, words: []const u32) !void {
        if (f.link) |l| try l.send(words);
    }

    pub fn backend(f: *Fx) be.Backend {
        return .{ .ptr = f, .vtable = &.{
            .prefill = prefillFn,
            .prefill_step = prefillStepFn,
            .first = firstFn,
            .first_masked = firstMaskedFn,
            .held = heldTokensFn,
            .queue = queueFn,
            .read = readFn,
            .verify = verifyFn,
            .keep = keepFn,
            .draft = draftFn,
            .tree = heldFn,
            .release = releaseFn,
        } };
    }

    pub fn facts(f: *const Fx) lanes.Model {
        const drafting = f.e.weights.model.mtp != null and f.drafts > 0;
        return .{
            .exact_width = cache.max_window,
            .gpu_tokens = false,
            .mtp = drafting,
            .speculate = drafting,
            .speculate_early = false,
            .drafts = f.drafts,
            .hidden_rows = true,
            .batch_rows = @intCast(cache.max_window * f.streams),
            .max_streams = @intCast(f.streams),
            .draft_streams = false,
        };
    }

    fn free(f: *Fx, l: *Lane) void {
        f.e.stream.synchronize() catch {};
        f.e.unlease(l.caches);
        if (l.head) |*h| h.deinit(f.gpa);
        l.streams.free();
        if (l.first_row) |row| f.gpa.free(row);
        f.gpa.destroy(l);
    }

    fn of(ptr: *anyopaque) *Fx {
        const f: *Fx = @ptrCast(@alignCast(ptr));
        if (bound != f.e) {
            f.e.ctx.makeCurrent() catch |err| std.log.err("hipSetDevice on the lane thread: {s}", .{@errorName(err)});
            bound = f.e;
        }
        return f;
    }

    fn take(f: *Fx, token: u32) u64 {
        const at = f.next;
        f.drawn[at % ring] = token;
        f.next += 1;
        return at;
    }

    fn sampling(s: *const lanes.Stream) ?lanes.Sampling {
        return s.sampling;
    }

    /// The pending window's first `rows` rows kept; the other ranks keep the same.
    fn mark(f: *Fx, lane: *Lane, rows: usize) !void {
        const p = lane.pending orelse return;
        if (f.link != null) try f.send(&.{ @backingInt(worker.Op.keep), lane.id, @intCast(rows), @intCast(p.rows) });
        try f.e.keepWindow(&lane.caches, p.tokens[0..p.rows], rows);
        lane.pending = null;
    }

    /// Whether a stream other than `mine` is past its prompt: a prompt's chunks then hold its rounds back.
    fn decoding(f: *const Fx, mine: *const Lane) bool {
        var it = f.lanes.valueIterator();
        while (it.next()) |l| if (l.* != mine and l.*.fill == null) return true;
        return false;
    }

    fn prefillFn(ptr: *anyopaque, s: *lanes.Stream) anyerror!void {
        while (!try prefillStepFn(ptr, s)) {}
    }

    /// Whether a stream other than `s` holds cache slot `slot`.
    fn slotBusy(f: *const Fx, slot: u32, s: *const lanes.Stream) bool {
        var it = f.lanes.iterator();
        while (it.next()) |kv| if (kv.key_ptr.* != s and kv.value_ptr.*.caches.id == slot) return true;
        return false;
    }

    fn begin(f: *Fx, s: *lanes.Stream) !*Lane {
        const prompt = s.prompt();
        const total = prompt.len + s.max_new + cache.max_window + 1;
        if (prompt.len == 0 or total > f.e.o.capacity) return error.PromptTooLong;
        const gop = try f.lanes.getOrPut(f.gpa, s);
        if (gop.found_existing) f.free(gop.value_ptr.*);
        errdefer f.lanes.removeByPtr(gop.key_ptr);
        const e = f.e;
        const m = &e.weights.model;
        var slot: u32 = 0;
        while (f.slotBusy(slot, gop.key_ptr.*)) slot += 1;
        const caches = try e.lease(slot);
        errdefer e.unlease(caches);
        var head: ?cache.Caches = if (m.mtp) |*h| try head_mod.caches(f.gpa, &e.driver, h, total, e.act, e.eos) else null;
        errdefer if (head) |*h| h.deinit(f.gpa);
        var streams = try hip.DeviceBuffer.alloc(&e.driver, cache.max_window * m.dims.wide() * m.act.size());
        errdefer streams.free();
        const lane = try f.gpa.create(Lane);
        lane.* = .{ .caches = caches, .head = head, .streams = streams, .fill = .{ .at = 0 }, .id = f.next_id };
        f.next_id += 1;
        gop.value_ptr.* = lane;
        if (f.link != null) try f.send(&.{ @backingInt(worker.Op.begin), lane.id, slot });
        const fill = &lane.fill.?;
        if (f.store) |*st| {
            var arena: std.heap.ArenaAllocator = .init(f.gpa);
            defer arena.deinit();
            fill.kept0 = st.counts.kept;
            const plan = st.begin(arena.allocator(), prompt, s.history_len, s.shared_prefixes, &.{}, lane, &.{}) catch pc.Plan{};
            fill.at = plan.from;
            fill.from = plan.from;
            fill.count = @min(plan.marks.len, fill.stops.len);
            for (plan.marks[0..fill.count], 0..) |mk, i| fill.stops[i] = mk;
        }
        s.cached = @intCast(fill.at);
        return lane;
    }

    /// One prompt chunk (64-row aligned cuts); the head absorbs its rows; the last chunk draws the first token.
    fn prefillStepFn(ptr: *anyopaque, s: *lanes.Stream) anyerror!bool {
        const f = of(ptr);
        if (s.isCancelled()) return error.Cancelled;
        const lane = if (f.lanes.get(s)) |l| (if (l.fill != null) l else try f.begin(s)) else try f.begin(s);
        const e = f.e;
        const m = &e.weights.model;
        const prompt = s.prompt();
        const fill = &lane.fill.?;
        const at = fill.at;
        const stop: usize = if (fill.next < fill.count) fill.stops[fill.next] else prompt.len;
        // the idle chunk alone, the shorter one while other streams decode between chunks (both on the 64-row grid)
        const chunk: usize = if (f.decoding(lane)) e.o.policy.fill_rows else e.o.chunk;
        const to = @min(stop, at + chunk);
        const last = to == prompt.len;
        if (f.link != null) {
            f.msg.clearRetainingCapacity();
            try f.msg.appendSlice(f.gpa, &.{ @backingInt(worker.Op.fill), lane.id, @intFromBool(last), @intCast(to - at) });
            try f.msg.appendSlice(f.gpa, prompt[at..to]);
            try f.send(f.msg.items);
        }
        const out = try e.prompt(&lane.caches, prompt[at..to]);
        const rows = to - at;
        const row_bytes = m.dims.wide() * m.act.size();
        if (lane.head) |*hc| {
            // the head's keys for every prompt row with a next token (the last waits for the first draw)
            const n = if (last) rows - 1 else rows;
            if (n > 0) _ = try head_mod.run(e.ops(&e.arena), m, &e.host, hc, out.streams, prompt[at + 1 ..][0..n], at, f.ones);
        }
        fill.at = to;
        if (fill.next < fill.count and to == fill.stops[fill.next]) {
            if (f.store) |*st| _ = st.keep(prompt, @intCast(to), lane, &.{}, &.{});
            fill.next += 1;
        }
        if (!last) return false;
        if (f.store) |*st| st.report(prompt.len, @intCast(fill.from), st.counts.kept - fill.kept0);
        try hip.raw.copy(lane.streams, 0, out.streams.ptr + (rows - 1) * row_bytes, row_bytes, e.stream.handle);
        if (s.grammar != null and lane.first_row == null) lane.first_row = try f.gpa.alloc(u16, e.cfg.vocab);
        const o = e.ops(&e.arena);
        const logits = try e.logits(o, forward.at(out.mixed, (rows - 1) * m.dims.hidden), 1);
        var token: [1]u32 = undefined;
        try e.drawer.draw(o, e.stream, logits, &.{.{ .sampling = sampling(s), .position = prompt.len, .keep = if (s.grammar != null) lane.first_row else null }}, &token, false);
        _ = f.take(token[0]);
        lane.fill = null;
        return true;
    }

    fn firstFn(ptr: *anyopaque, s: *lanes.Stream, position: u64) anyerror!u64 {
        const f = of(ptr);
        if (position != s.prompt_len) return error.PositionMismatch;
        return f.next - 1;
    }

    fn firstMaskedFn(ptr: *anyopaque, s: *lanes.Stream, position: u64, bits: []const u32) anyerror!u64 {
        const f = of(ptr);
        if (position != s.prompt_len) return error.PositionMismatch;
        const lane = f.lanes.get(s) orelse return error.UnknownStream;
        const kept = lane.first_row orelse return error.StructuresUnsupported;
        const row = try f.gpa.dupe(u16, kept);
        defer f.gpa.free(row);
        return f.take(try f.e.drawer.drawHost(row, .{ .sampling = sampling(s), .position = position, .mask = bits }));
    }

    fn heldTokensFn(ptr: *anyopaque, s: *lanes.Stream, out: []u32) anyerror!void {
        const lane = of(ptr).lanes.get(s) orelse return error.UnknownStream;
        if (out.len > lane.held_n) return error.NoHeldDrafts;
        @memcpy(out, lane.held[0..out.len]);
    }

    fn queueFn(_: *anyopaque, _: *lanes.Stream, _: be.Feed, _: u64) anyerror!u64 {
        return error.NotPipelined;
    }

    fn readFn(ptr: *anyopaque, handle: u64) anyerror!u32 {
        const f = of(ptr);
        if (handle >= f.next or f.next - handle > ring) return error.NoSuchToken;
        return f.drawn[handle % ring];
    }

    /// Every window in one forward, each from its stream's kept length; every row drawn at its position.
    fn verifyFn(ptr: *anyopaque, windows: []const be.Window, out: []be.Verified) anyerror!void {
        const f = of(ptr);
        const e = f.e;
        const m = &e.weights.model;
        if (windows.len > forward.max_segs) return error.WindowTooWide;
        var tokens: [forward.max_segs][cache.max_window]u32 = undefined;
        var segs: [forward.max_segs]forward.Seg = undefined;
        var total: usize = 0;
        if (f.link != null) {
            f.msg.clearRetainingCapacity();
            try f.msg.appendSlice(f.gpa, &.{ @backingInt(worker.Op.verify), @intCast(windows.len), 0 });
        }
        for (windows, 0..) |w, i| {
            if (w.parents != null) return error.TreesNotBuilt;
            const lane = f.lanes.get(w.stream) orelse return error.UnknownStream;
            if (lane.pending) |p| try f.mark(lane, p.rows);
            const n = w.rows();
            if (n > cache.max_window) return error.WindowTooWide;
            if (w.held > lane.held_n or (w.held > 0 and w.tokens.len > 0)) return error.NoHeldDrafts;
            tokens[i][0] = w.pending;
            if (w.held > 0) @memcpy(tokens[i][1..n], lane.held[0..w.held]) else @memcpy(tokens[i][1..n], w.tokens);
            const len = lane.caches.len;
            for (w.positions, 0..) |p, r| if (p != len + 1 + r) return error.PositionMismatch;
            segs[i] = .{ .c = &lane.caches, .tokens = tokens[i][0..n], .pos = len };
            total += n;
            if (f.link != null) {
                try f.msg.appendSlice(f.gpa, &.{ lane.id, @intCast(n) });
                try f.msg.appendSlice(f.gpa, tokens[i][0..n]);
            }
        }
        const pick = try e.choose(segs[0..windows.len]);
        if (f.link != null) {
            f.msg.items[2] = @backingInt(pick);
            try f.send(f.msg.items);
        }
        const done = try e.round(segs[0..windows.len], pick);
        const fw = done.out;
        const row_bytes = m.dims.wide() * m.act.size();
        var off: usize = 0;
        for (windows) |w| {
            const lane = f.lanes.get(w.stream).?;
            try hip.raw.copy(lane.streams, 0, fw.streams.ptr + off * row_bytes, w.rows() * row_bytes, e.stream.handle);
            off += w.rows();
        }
        const o = e.ops(&e.arena);
        const logits = done.logits;
        var reqs: [forward.max_segs * cache.max_window]draw.Request = undefined;
        var drawn: [forward.max_segs * cache.max_window]u32 = undefined;
        off = 0;
        for (windows) |w| {
            const n = w.rows();
            const words = if (w.masks.len > 0) w.masks.len / n else 0;
            for (w.positions, 0..) |p, r| reqs[off + r] = .{ .sampling = sampling(w.stream), .position = p, .mask = w.masks[r * words ..][0..words] };
            off += n;
        }
        if (total > e.drawer.rows) return error.WindowTooWide;
        try e.drawer.draw(o, e.stream, logits, reqs[0..total], drawn[0..total], false);
        off = 0;
        for (windows, out, 0..) |w, *v, i| {
            const n = w.rows();
            const lane = f.lanes.get(w.stream).?;
            @memcpy(v.sampled[0..n], drawn[off..][0..n]);
            @memcpy(v.drafts, if (w.held > 0) lane.held[0..w.held] else w.tokens);
            lane.held_n = 0;
            lane.pending = .{ .tokens = tokens[i], .rows = n };
            off += n;
        }
    }

    fn keepFn(ptr: *anyopaque, windows: []const be.Window, paths: []const []const u32) anyerror!void {
        const f = of(ptr);
        for (windows, paths) |w, path| {
            const lane = f.lanes.get(w.stream) orelse return error.UnknownStream;
            for (path, 0..) |r, j| if (r != j) return error.TreesNotBuilt;
            if (lane.pending != null) try f.mark(lane, path.len);
        }
    }

    /// The head absorbs each stream's kept rows (residual streams and the token after each), then chains its drafts.
    fn draftFn(ptr: *anyopaque, requests: []const be.DraftRequest) anyerror!void {
        const f = of(ptr);
        const e = f.e;
        const m = &e.weights.model;
        const row_bytes = m.dims.wide() * m.act.size();
        for (requests) |r| {
            const lane = f.lanes.get(r.stream) orelse return error.UnknownStream;
            if (lane.pending) |p| try f.mark(lane, if (r.rows) |kept| kept.len else p.rows);
            lane.held_n = 0;
            if (r.position != lane.caches.len + 1) return error.PositionMismatch;
            const hc = if (lane.head) |*h| h else continue;
            const token: u32 = if (r.rows != null) r.follow[r.follow.len - 1] else switch (r.first orelse return error.NoFirstToken) {
                .handle => |at| f.drawn[at % ring],
                .value => |v| v,
            };
            var follow_buf: [cache.max_window]u32 = undefined;
            const follow: []const u32 = if (r.rows != null) r.follow else blk: {
                follow_buf[0] = token;
                break :blk follow_buf[0..1];
            };
            const start = lane.caches.len - follow.len;
            try e.stream.synchronize();
            e.host.used = 0;
            e.arena.reset();
            const o = e.ops(&e.arena);
            var out = try head_mod.run(o, m, &e.host, hc, .{ .ptr = lane.streams.base(), .kind = m.act }, follow, start, f.ones);
            var mixed = forward.at(out.mixed, (follow.len - 1) * m.dims.hidden);
            var streams = out.streams.ptr + (follow.len - 1) * row_bytes;
            const depth = @min(r.depth, max_depth);
            // a chain past the first round ends after a draft the head gives under the confidence (--mtp-confidence)
            const stop_under: f32 = if (r.rows != null) @floatCast(e.o.policy.mtp.confidence) else 0.0;
            var d: usize = 0;
            var chained: usize = 0;
            while (d < depth) : (d += 1) {
                var tok: [1]u32 = undefined;
                var prob: [1]f32 = undefined;
                try head_mod.draws(o, m, &e.host, e.stream, mixed, 1, f.draft_ids, &tok, &prob);
                lane.held[d] = tok[0];
                chained = d + 1;
                if (d + 1 == depth or prob[0] < stop_under) break;
                out = try head_mod.run(o, m, &e.host, hc, .{ .ptr = streams, .kind = m.act }, &tok, lane.caches.len + d, f.ones);
                mixed = out.mixed;
                streams = out.streams.ptr;
            }
            lane.held_n = chained;
        }
    }

    fn heldFn(ptr: *anyopaque, s: *lanes.Stream, _: std.mem.Allocator) anyerror!?lanes.stream.Held {
        const lane = of(ptr).lanes.get(s) orelse return error.UnknownStream;
        return .{ .count = @intCast(lane.held_n) };
    }

    fn releaseFn(ptr: *anyopaque, s: *lanes.Stream) void {
        const f = of(ptr);
        const kv = f.lanes.fetchRemove(s) orelse return;
        if (f.link != null) f.send(&.{ @backingInt(worker.Op.release), kv.value.id }) catch {};
        f.free(kv.value);
    }
};
