//! Ripple behind lanes.Backend: a cache slot a stream, prompts a chunk a step, one forward a round of windows.
const std = @import("std");
const cuda = @import("cuda");
const lanes = @import("lanes");
const fwd = @import("tp2_forward.zig");
const kept = @import("tp2_kept.zig");

const be = lanes.backend;
const Stream = lanes.Stream;

/// Prompt rows a step takes: the forward's chunk, cut at the prompt cache's marks.
pub const prompt_rows: u32 = fwd.max_rows;
/// Rows a round takes over every stream (tree attention's decode scratch), and the widest window a stream verifies.
pub const round_rows: u32 = fwd.tree_rows;
/// Drawn tokens a handle names (a ring; read before the round loop could wrap it).
const ring = 1024;

const Lane = struct {
    slot: usize,
    len: u64 = 0, // the cache length: positions [0, len) hold this stream's keys
    base: u64 = 0, // the cache length the last verify started from
    rows: std.ArrayList(u32) = .empty, // the last verify's row tokens
    last: u64, // the prompt's last row's fp32 logits [V] (the first draw reads them)
    rule: u64, // sample.cu's Rule, when the stream samples
    sampled: bool = false,
    fill: ?usize = null, // a prompt pass going a chunk at a time: where its next chunk starts
};

/// A kept state: its id on every rank, its position and the slot whose full rows it leaves in place.
pub const Saved = struct { id: i32, at: u32, slot: usize };

/// The torch-op images' functions the draws need (argmax by torch's first-maximum rule, bf16 casts) and sample.cu.
const Draw = struct {
    mods: [3]cuda.Module,
    argmax_fn: cuda.Function,
    to_bf16_fn: cuda.Function,
    draws: cuda.sample.Draws,

    fn load(d: *const cuda.Driver) !Draw {
        const kk = cuda.kernels;
        if (!kk.available) return error.BuiltWithoutKernels;
        var k: Draw = undefined;
        var made: usize = 0;
        errdefer for (k.mods[0..made]) |*m| m.unload();
        for ([_][]const u8{ kk.sample, kk.torch_argmax, kk.torch_pointwise }, 0..) |img, i| {
            k.mods[i] = try cuda.Module.load(d, img);
            made += 1;
        }
        k.draws = try cuda.sample.Draws.resolve(k.mods[0]);
        k.argmax_fn = try k.mods[1].function("tf_argmax_rows_i32_kernel");
        k.to_bf16_fn = try k.mods[2].function("tf_f32_to_bf16_kernel");
        return k;
    }
};

pub const Ripple = struct {
    gpa: std.mem.Allocator,
    m: *fwd.Model,
    k: Draw,
    lanes: std.AutoHashMapUnmanaged(*const Stream, Lane) = .empty,
    free: []bool, // slot i is free
    written: []std.ArrayList(u32), // the tokens a slot's full rows hold at positions 0.. (kept states resume on them)
    used: []u64, // when a slot was last taken (the least recent of equals goes first)
    clock: u64 = 0,
    logits: cuda.DeviceBuffer, // fp32 [round_rows, V]
    logits16: cuda.DeviceBuffer, // their bf16 copy for keyed draws
    tok: cuda.DeviceBuffer, // drawn tokens: the handle ring, then a round's rows
    meta: cuda.DeviceBuffer, // a draw's first position minus one, a window each
    slot_bufs: []cuda.DeviceBuffer, // a slot's last-row logits, then its rule: two a slot
    next: u64 = 0,
    next_id: i32 = 1,
    priority_copies: bool = false,
    copies: bool = true, // prompt-copy drafts beside a drafter's

    pub fn init(gpa: std.mem.Allocator, m: *fwd.Model) !Ripple {
        const v = m.c.vocab;
        const d = m.d;
        var k = try Draw.load(d);
        errdefer for (&k.mods) |*x| x.unload();
        const free = try gpa.alloc(bool, m.slots);
        @memset(free, true);
        const written = try gpa.alloc(std.ArrayList(u32), m.slots);
        @memset(written, .empty);
        const used = try gpa.alloc(u64, m.slots);
        @memset(used, 0);
        const bufs = try gpa.alloc(cuda.DeviceBuffer, 2 * m.slots);
        for (bufs, 0..) |*b, i| b.* = try cuda.DeviceBuffer.alloc(d, if (i % 2 == 0) v * 4 else @sizeOf(cuda.sample.Rule));
        return .{ .gpa = gpa, .m = m, .k = k, .free = free, .written = written, .used = used, .logits = try cuda.DeviceBuffer.alloc(d, round_rows * v * 4), .logits16 = try cuda.DeviceBuffer.alloc(d, round_rows * v * 2), .tok = try cuda.DeviceBuffer.alloc(d, (ring + round_rows) * 4), .meta = try cuda.DeviceBuffer.alloc(d, round_rows * 16), .slot_bufs = bufs };
    }

    pub fn deinit(x: *Ripple) void {
        var it = x.lanes.valueIterator();
        while (it.next()) |l| l.rows.deinit(x.gpa);
        x.lanes.deinit(x.gpa);
        for (x.slot_bufs) |*b| b.free();
        x.gpa.free(x.slot_bufs);
        for (x.written) |*w| w.deinit(x.gpa);
        x.gpa.free(x.written);
        x.gpa.free(x.used);
        x.gpa.free(x.free);
        x.logits.free();
        x.logits16.free();
        x.tok.free();
        x.meta.free();
        for (&x.k.mods) |*md| md.unload();
    }

    pub fn backend(x: *Ripple) be.Backend {
        return .{ .ptr = x, .vtable = &vtable, .chunk_invariant = false };
    }

    const vtable: be.Backend.VTable = .{ .prefill = prefill, .prefill_step = prefillStep, .cancel_prefill = cancelPrefill, .first = first, .queue = queue, .read = read, .verify = verify, .keep = keep, .draft = draft, .release = release, .reports_fill = true, .prepare_features = prepareFeatures, .features = features };

    /// The round loop's facts: no draft head; host copies verified in windows up to 16 rows, streams sharing a forward.
    pub fn facts(x: *const Ripple) lanes.Model {
        return .{ .root_copies = x.priority_copies, .context_copies = x.priority_copies, .copy_match = if (x.priority_copies) 4 else 8, .exact_width = round_rows, .copy_drafts = x.copies, .gpu_tokens = false, .mtp = false, .speculate = false, .hidden_rows = true, .batch_rows = round_rows, .max_streams = @intCast(@min(x.m.slots, round_rows)) };
    }

    fn self(ptr: *anyopaque) *Ripple {
        return @ptrCast(@alignCast(ptr));
    }

    fn launch(x: *Ripple, f: cuda.Function, grid: u32, block: u32, a: *cuda.Args) !void {
        try cuda.launch.launch(f, .{ .grid = .{ .x = grid }, .block = .{ .x = block } }, x.m.stream, a);
    }

    /// Rows [row0, row0 + rows) of `logits` drawn into tok[at..], row r at `position0 + r` (argmax when greedy).
    fn drawRows(x: *Ripple, l: *const Lane, logits: u64, row0: usize, rows: usize, position0: u64, at: usize, meta_at: usize) !void {
        const v = x.m.c.vocab;
        const src = logits + row0 * v * 4;
        var a: cuda.Args = .{};
        if (!l.sampled) {
            a.add(src);
            a.add(x.tok.ptr + at * 4);
            for ([_]usize{ rows, v, v }) |n| a.add(@as(i64, @intCast(n)));
            a.add(@as(i32, 1)); // fp32
            return x.launch(x.k.argmax_fn, @intCast(rows), 1024, &a);
        }
        a.add(src);
        a.add(x.logits16.ptr);
        a.add(@as(u64, rows * v));
        try x.launch(x.k.to_bf16_fn, @intCast(@max(1, @min((rows * v + 255) / 256, 65535))), 256, &a);
        const meta = [4]i32{ @intCast(position0 - 1), 0, 0, 0 };
        const mp = x.meta.ptr + meta_at * 16;
        try x.m.up(mp, std.mem.asBytes(&meta));
        try x.k.draws.draw(x.m.stream, x.logits16.ptr, v, l.rule, mp, 0, x.tok.ptr + at * 4, rows, null, null);
    }

    /// The slot's full rows now hold `tokens` at p0.. (and nothing it knows of past them).
    fn wrote(x: *Ripple, slot: usize, p0: usize, tokens: []const u32) !void {
        const w = &x.written[slot];
        w.shrinkRetainingCapacity(@min(w.items.len, p0));
        if (w.items.len != p0) return; // a gap: rows below p0 came from no tokens the slot recorded
        try w.appendSlice(x.gpa, tokens);
    }

    /// Whether `slot`'s full rows hold `tokens` from position 0.
    fn holds(x: *const Ripple, slot: usize, tokens: []const u32) bool {
        const w = x.written[slot].items;
        return w.len >= tokens.len and std.mem.eql(u32, w[0..tokens.len], tokens);
    }

    /// The prompt cache's state at `at` (the pass stands there): the slot's rings, copied on every rank.
    pub fn saveState(x: *Ripple, owner: ?*anyopaque, at: u32) !*Saved {
        const s: *const Stream = @ptrCast(@alignCast(owner orelse return error.NoOwner));
        const l = x.lanes.getPtr(s) orelse return error.UnknownStream;
        if ((l.fill orelse l.len) != at) return error.NotAtMark;
        const k = try x.gpa.create(Saved);
        errdefer x.gpa.destroy(k);
        k.* = .{ .id = x.next_id, .at = at, .slot = l.slot };
        x.next_id +%= 1;
        try x.m.keptOp(.save, l.slot, k.id, 0);
        return k;
    }

    pub fn dropState(x: *Ripple, k: *Saved) void {
        x.m.keptOp(.drop, 0, k.id, 0) catch |e| std.log.warn("ricursive: kept state {d} not dropped: {s}", .{ k.id, @errorName(e) });
        x.gpa.destroy(k);
    }

    /// The free slot for `p`: the state's own, else the one sharing most of the prompt (least recent of equals).
    fn pick(x: *const Ripple, p: []const u32, want: ?usize) ?usize {
        if (want) |w| if (x.free[w]) return w;
        var best: ?usize = null;
        var best_len: usize = 0;
        for (x.free, 0..) |f, i| if (f) {
            const w = x.written[i].items;
            const n = std.mem.indexOfDiff(u32, w[0..@min(w.len, p.len)], p[0..@min(w.len, p.len)]) orelse @min(w.len, p.len);
            if (best == null or n > best_len or (n == best_len and x.used[i] < x.used[best.?])) {
                best = i;
                best_len = n;
            }
        };
        return best;
    }

    /// A prompt pass begins: a slot, the stream's rule, a kept state restored; where its first chunk starts.
    fn opening(x: *Ripple, s: *Stream) !*Lane {
        const p = s.prompt();
        if (p.len == 0) return error.EmptyPrompt;
        if (p.len + @as(usize, s.max_new) > x.m.context) return error.PastContext;
        const got = try x.lanes.getOrPut(x.gpa, s);
        if (!got.found_existing) {
            const k: ?*const Saved = if (s.reuse.saved) |sv| @ptrCast(@alignCast(sv)) else null;
            const slot = x.pick(p, if (k) |kk| kk.slot else null) orelse {
                x.lanes.removeByPtr(got.key_ptr);
                return error.NoFreeSlot;
            };
            x.free[slot] = false;
            x.clock += 1;
            x.used[slot] = x.clock;
            got.value_ptr.* = .{ .slot = slot, .last = x.slot_bufs[2 * slot].ptr, .rule = x.slot_bufs[2 * slot + 1].ptr };
        }
        const l = got.value_ptr;
        l.sampled = false;
        if (s.sampling) |sp| if (sp.temperature > 0) {
            const r = cuda.sample.Rule.from(lanes.gpu_rule.rule(sp));
            try x.m.up(l.rule, std.mem.asBytes(&r));
            l.sampled = true;
        };
        s.cached = 0;
        l.fill = x.resumeAt(s, l, p) catch |e| blk: {
            std.log.warn("ricursive: kept state not resumed: {s}", .{@errorName(e)});
            s.reuse_failed = true;
            break :blk 0;
        };
        s.fill_at.store(@intCast(l.fill.?), .monotonic);
        return l;
    }

    /// The kept state's position, resumed in the lane's slot (0: none, or not this prompt's rows any more).
    fn resumeAt(x: *Ripple, s: *Stream, l: *const Lane, p: []const u32) !usize {
        const saved = s.reuse.saved orelse return 0;
        const k: *const Saved = @ptrCast(@alignCast(saved));
        if (k.at != s.reuse.at or k.at == 0 or k.at >= p.len or !x.holds(k.slot, p[0..k.at])) {
            s.reuse_failed = true;
            return 0;
        }
        if (k.slot != l.slot) { // its slot is busy: its full rows below `at` into this one
            try x.m.keptOp(.copy, l.slot, @intCast(k.slot), k.at);
            x.written[l.slot].clearRetainingCapacity();
            try x.written[l.slot].appendSlice(x.gpa, p[0..k.at]);
        }
        try x.m.keptOp(.restore, l.slot, k.id, 0);
        x.written[l.slot].shrinkRetainingCapacity(k.at);
        s.cached = k.at;
        return k.at;
    }

    /// One chunk from `l.fill` to the next chunk end or mark; true once the prompt's last row is in.
    fn chunk(x: *Ripple, s: *Stream, l: *Lane) !bool {
        const p = s.prompt();
        const at = l.fill.?;
        if (s.isCancelled()) {
            s.keepInterruptedPrefill(@intCast(at));
            return error.Cancelled;
        }
        var end = @min(p.len, (at / prompt_rows + 1) * prompt_rows);
        for (s.reuse.marks) |mk| if (mk > at and mk < end) {
            end = mk;
        };
        const done = end == p.len;
        try x.m.prompt(l.slot, p[at..end], at, done);
        try x.wrote(l.slot, at, p[at..end]);
        if (done) try x.m.logitsTo(l.last, 1);
        s.fill_at.store(@intCast(end), .monotonic);
        l.fill = if (done) null else end;
        if (done) l.len = p.len;
        if (s.reuse.hook) |hk| if (std.mem.indexOfScalar(u32, s.reuse.marks, @intCast(end)) != null) hk.at(hk.ptr, s, @intCast(end));
        return done;
    }

    fn prefill(ptr: *anyopaque, s: *Stream) anyerror!void {
        const x = self(ptr);
        const l = try x.opening(s);
        errdefer x.dropLane(s);
        while (!try x.chunk(s, l)) {}
    }

    fn prefillStep(ptr: *anyopaque, s: *Stream, _: u32) anyerror!bool {
        const x = self(ptr);
        const going = if (x.lanes.getPtr(s)) |l| l.fill != null else false;
        const l = if (going) x.lanes.getPtr(s).? else try x.opening(s);
        errdefer x.dropLane(s);
        return x.chunk(s, l);
    }

    fn cancelPrefill(ptr: *anyopaque, s: *Stream) !void {
        const l = self(ptr).lanes.getPtr(s) orelse return;
        if (l.fill) |at| s.keepInterruptedPrefill(@intCast(at));
    }

    fn handle(x: *Ripple) usize {
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

    fn value(x: *Ripple, feed: be.Feed) !u32 {
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
        const one = [1]u32{t};
        try x.m.decode(&.{.{ .slot = l.slot, .p0 = l.len, .tokens = &one }});
        try x.wrote(l.slot, l.len, &one);
        try x.m.logitsTo(x.logits.ptr, 1);
        l.len += 1;
        const h = x.next;
        try x.drawRows(l, x.logits.ptr, 0, 1, position, x.handle(), 0);
        return h;
    }

    fn read(ptr: *anyopaque, h: u64) anyerror!u32 {
        const x = self(ptr);
        if (h >= x.next or x.next - h > ring) return error.NoSuchToken;
        var out: [1]u32 = undefined;
        try x.m.stream.synchronize();
        try x.tok.download(@intCast((h % ring) * 4), std.mem.sliceAsBytes(&out));
        return out[0];
    }

    fn verify(ptr: *anyopaque, windows: []const be.Window, out: []be.Verified) anyerror!void {
        const x = self(ptr);
        var entries: [fwd.max_streams]fwd.Entry = undefined;
        if (windows.len > fwd.max_streams) return error.RoundTooWide;
        var total: usize = 0;
        for (windows, 0..) |w, i| {
            if (w.held != 0) return error.NoHeldDrafts; // no draft head: every draft comes from the host
            if (w.parents != null) return error.ChainsOnly;
            if (w.masks.len != 0) return error.StructuresUnsupported;
            const l = x.lanes.getPtr(w.stream) orelse return error.UnknownStream;
            l.base = l.len;
            l.rows.clearRetainingCapacity();
            try l.rows.append(x.gpa, w.pending);
            try l.rows.appendSlice(x.gpa, w.tokens);
            for (w.positions, 0..) |pos, r| if (pos != l.base + r + 1) return error.PositionMismatch;
            entries[i] = .{ .slot = l.slot, .p0 = l.base, .tokens = l.rows.items };
            total += l.rows.items.len;
        }
        if (total > round_rows) return error.RoundTooWide;
        try x.m.decode(entries[0..windows.len]);
        for (entries[0..windows.len]) |e| try x.wrote(e.slot, e.p0, e.tokens);
        try x.m.logitsTo(x.logits.ptr, total);
        var at: usize = 0;
        for (windows, 0..) |w, i| {
            const l = x.lanes.getPtr(w.stream).?;
            try x.drawRows(l, x.logits.ptr, at, l.rows.items.len, w.positions[0], ring + at, i);
            at += l.rows.items.len;
        }
        var got: [round_rows]u32 = undefined;
        try x.m.stream.synchronize();
        try x.tok.download(ring * 4, std.mem.sliceAsBytes(got[0..total]));
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
            x.written[l.slot].shrinkRetainingCapacity(@min(x.written[l.slot].items.len, l.len));
        }
    }

    /// No draft head is loaded: absorbing is a no-op, and a request for drafts is refused by name.
    fn draft(_: *anyopaque, requests: []const be.DraftRequest) anyerror!void {
        for (requests) |r| if (r.depth != 0 or r.lanes != null) return error.NoDraftHead;
    }

    /// A drafter's taps, before any forward: the model keeps their rows from now on.
    fn prepareFeatures(ptr: *anyopaque, taps: []const u32) anyerror!void {
        return self(ptr).m.setTaps(taps);
    }

    /// Tap rows [start, start + count) of the stream's slot (a call's rows, read before the next call; its last only).
    fn features(ptr: *anyopaque, s: *Stream, taps: []const u32, start: u64, count: u32) anyerror!be.Features {
        const x = self(ptr);
        const m = x.m;
        const l = x.lanes.getPtr(s) orelse return error.UnknownStream;
        if (m.tap_ring == 0 or !std.mem.eql(u32, taps, m.taps[0..m.n_taps])) return error.TapsNotPrepared;
        const held = fwd.tap_rows - fwd.max_rows; // what the ring still holds of the newest rows
        const n: u64 = @min(count, held);
        const row = m.n_taps * m.c.hidden * 2;
        return .{ .buffer = m.tap_ring + l.slot * fwd.tap_rows * row, .offset = start + count - n, .rows = @intCast(n), .row_bytes = @intCast(row), .space = .device, .dtype = .bf16 };
    }

    fn dropLane(x: *Ripple, s: *const Stream) void {
        if (x.lanes.fetchRemove(s)) |entry| {
            var l = entry.value;
            x.free[l.slot] = true;
            l.rows.deinit(x.gpa);
        }
    }

    fn release(ptr: *anyopaque, s: *Stream) void {
        self(ptr).dropLane(s);
    }
};

test "a slot's recorded rows: a write past a gap records nothing, a rewrite cuts what followed" {
    var w: std.ArrayList(u32) = .empty;
    defer w.deinit(std.testing.allocator);
    try w.appendSlice(std.testing.allocator, &.{ 1, 2, 3, 4 });
    w.shrinkRetainingCapacity(@min(w.items.len, 2));
    try w.appendSlice(std.testing.allocator, &.{9});
    try std.testing.expectEqualSlices(u32, &.{ 1, 2, 9 }, w.items);
}
