//! Mini behind lanes.Backend: a slot a stream, one-row windows; no drafts yet (C5), a conv state has no rollback.
const std = @import("std");
const cuda = @import("cuda");
const lanes = @import("lanes");
const mm = @import("mini_model.zig");
const snap = @import("mini_snap.zig");

const be = lanes.backend;
const Stream = lanes.Stream;

/// Prompt rows a step takes.
pub const prompt_rows = mm.prompt_rows;
/// Window rows a round takes over every stream.
pub const round_rows = mm.round_rows;
/// Drawn tokens a handle names (a ring; read before the round loop could wrap it).
const ring = 1024;

/// The serving launches beside the forward: argmax, bf16 logits and sample.cu's keyed draws.
pub const Draws = struct {
    mods: [3]cuda.Module,
    argmax_fn: cuda.Function,
    to_bf16_fn: cuda.Function,
    draws: cuda.sample.Draws,

    pub fn load(d: *const cuda.Driver) !Draws {
        if (!cuda.kernels.available) return error.BuiltWithoutKernels;
        const kk = cuda.kernels;
        var x: Draws = undefined;
        var loaded: usize = 0;
        errdefer for (x.mods[0..loaded]) |*m| m.unload();
        for ([_][]const u8{ kk.sample, kk.torch_argmax, kk.torch_pointwise }, 0..) |img, i| {
            x.mods[i] = try cuda.Module.load(d, img);
            loaded += 1;
        }
        x.draws = try cuda.sample.Draws.resolve(x.mods[0]);
        x.argmax_fn = try x.mods[1].function("tf_argmax_rows_i32_kernel");
        x.to_bf16_fn = try x.mods[2].function("tf_f32_to_bf16_kernel");
        return x;
    }

    pub fn deinit(x: *Draws) void {
        for (&x.mods) |*m| m.unload();
    }

    /// Each row's argmax over `v` fp32 logits, the first maximum on ties, into `out` (int32 a row).
    fn argmax(x: *const Draws, s: cuda.Stream, logits: u64, rows: usize, v: usize, out: u64) !void {
        var a: cuda.Args = .{};
        a.add(logits);
        a.add(out);
        for ([_]usize{ rows, v, v }) |n| a.add(@as(i64, @intCast(n)));
        a.add(@as(i32, 1)); // dtype 1: fp32
        try cuda.launch.launch(x.argmax_fn, .{ .grid = .{ .x = @intCast(rows) }, .block = .{ .x = 1024 } }, s, &a);
    }

    fn toBf16(x: *const Draws, s: cuda.Stream, src: u64, dst: u64, n: usize) !void {
        var a: cuda.Args = .{};
        a.add(src);
        a.add(dst);
        a.add(@as(u64, n));
        const grid: u32 = @intCast(@max(1, @min((n + 255) / 256, 65535)));
        try cuda.launch.launch(x.to_bf16_fn, .{ .grid = .{ .x = grid }, .block = .{ .x = 256 } }, s, &a);
    }
};

const Lane = struct {
    slot: usize,
    len: u64 = 0, // positions [0, len) hold this stream's keys and its conv states are at len
    base: u64 = 0, // the length the last verify started from
    rows: std.ArrayList(u32) = .empty, // the last verify's row tokens
    last: u64, // the prompt's last row's fp32 logits (the first draw reads them)
    rule: u64, // sample.cu's Rule, when the stream samples
    sampled: bool = false,
    fill: ?usize = null, // a prompt pass going a chunk at a time: where its next chunk starts
};

/// The positions a slot holds for a served window, in whole 1,024s.
pub fn capacity(window: usize) !usize {
    if (window == 0) return error.BadContext;
    return (try std.math.add(usize, window, 1023)) / 1024 * 1024;
}

/// A prompt and its reply fit `physical` positions.
pub fn fits(physical: usize, prompt: usize, reply: usize) bool {
    return prompt <= physical and reply <= physical - prompt;
}

pub const Mini = struct {
    gpa: std.mem.Allocator,
    d: *const cuda.Driver,
    m: *mm.Model,
    k: *const mm.Kernels,
    x: *const Draws,
    s: cuda.Stream,
    image_token: u32,
    lanes: std.AutoHashMapUnmanaged(*const Stream, Lane) = .empty,
    free: []bool, // slot i is free
    logits16: cuda.DeviceBuffer, // a round's logits as bf16 for keyed draws
    tok: cuda.DeviceBuffer, // drawn tokens: the handle ring, then a round's rows
    meta: cuda.DeviceBuffer, // a draw's first position minus one, a window each
    slot_bufs: []cuda.DeviceBuffer, // a slot's last-row logits, then its rule: two a slot
    next: u64 = 0,

    fn vocab(x: *const Mini) usize {
        return x.m.c.unpadded_vocab;
    }

    /// The device bytes `init` allocates: bf16 logits, the token ring, metas, each slot's last row and rule.
    pub fn openBytes(vocab_: usize, slots: usize) usize {
        return round_rows * vocab_ * 2 + (ring + round_rows) * 4 + round_rows * 16 + slots * (vocab_ * 4 + @sizeOf(cuda.sample.Rule));
    }

    pub fn init(gpa: std.mem.Allocator, d: *const cuda.Driver, m: *mm.Model, k: *const mm.Kernels, x: *const Draws, s: cuda.Stream, image_token: u32) !Mini {
        const v = m.c.unpadded_vocab;
        const free = try gpa.alloc(bool, m.slots);
        errdefer gpa.free(free);
        @memset(free, true);
        var logits16 = try cuda.DeviceBuffer.alloc(d, round_rows * v * 2);
        errdefer logits16.free();
        var tok = try cuda.DeviceBuffer.alloc(d, (ring + round_rows) * 4);
        errdefer tok.free();
        var meta = try cuda.DeviceBuffer.alloc(d, round_rows * 16);
        errdefer meta.free();
        const bufs = try gpa.alloc(cuda.DeviceBuffer, 2 * m.slots);
        errdefer gpa.free(bufs);
        var made: usize = 0;
        errdefer for (bufs[0..made]) |*b| b.free();
        for (bufs, 0..) |*b, i| {
            b.* = try cuda.DeviceBuffer.alloc(d, if (i % 2 == 0) v * 4 else @sizeOf(cuda.sample.Rule));
            made += 1;
        }
        return .{ .gpa = gpa, .d = d, .m = m, .k = k, .x = x, .s = s, .image_token = image_token, .free = free, .logits16 = logits16, .tok = tok, .meta = meta, .slot_bufs = bufs };
    }

    pub fn deinit(x: *Mini) void {
        var it = x.lanes.valueIterator();
        while (it.next()) |l| l.rows.deinit(x.gpa);
        x.lanes.deinit(x.gpa);
        for (x.slot_bufs) |*b| b.free();
        x.gpa.free(x.slot_bufs);
        x.gpa.free(x.free);
        x.logits16.free();
        x.tok.free();
        x.meta.free();
    }

    pub fn backend(x: *Mini) be.Backend {
        return .{ .ptr = x, .vtable = &vtable };
    }

    const vtable: be.Backend.VTable = .{ .prefill = prefill, .prefill_step = prefillStep, .first = first, .queue = queue, .read = read, .verify = verify, .keep = keep, .draft = draft, .release = release, .reports_fill = true };

    /// The round loop's facts: no draft head and no copies; one-row windows, streams sharing a forward.
    pub fn facts(x: *const Mini) lanes.Model {
        return .{ .copy_drafts = false, .exact_width = 1, .gpu_tokens = false, .mtp = false, .speculate = false, .hidden_rows = true, .plain_share = true, .batch_rows = round_rows, .max_streams = @intCast(@min(x.m.slots, round_rows)) };
    }

    fn self(ptr: *anyopaque) *Mini {
        return @ptrCast(@alignCast(ptr));
    }

    fn upload(x: *Mini, dst: u64, bytes: []const u8) !void {
        try cuda.copy_source.enqueue(x.d, dst, bytes, x.s);
    }

    /// Rows [row0, row0 + rows) of fp32 `logits` drawn into tok[at..], row r at `position0 + r` (argmax when greedy).
    fn drawRows(x: *Mini, l: *const Lane, logits: u64, row0: usize, rows: usize, position0: u64, at: usize, meta_at: usize) !void {
        const v = x.vocab();
        const src = logits + row0 * v * 4;
        if (!l.sampled) return x.x.argmax(x.s, src, rows, v, x.tok.ptr + at * 4);
        try x.x.toBf16(x.s, src, x.logits16.ptr, rows * v);
        const meta = [4]i32{ @intCast(position0 - 1), 0, 0, 0 };
        const mp = x.meta.ptr + meta_at * 16;
        try x.upload(mp, std.mem.asBytes(&meta));
        try x.x.draws.draw(x.s, x.logits16.ptr, v, l.rule, mp, 0, x.tok.ptr + at * 4, rows, null, null);
    }

    fn freeSlot(x: *const Mini) ?usize {
        return std.mem.indexOfScalar(bool, x.free, true);
    }

    /// Whether a request could start now: a free slot; one that never fits passes (its open refuses it).
    pub fn admits(x: *const Mini, prompt: usize, max_new: usize) bool {
        if (!fits(x.m.context, prompt, max_new)) return true;
        return x.freeSlot() != null;
    }

    /// A prompt pass begins: a slot and the stream's rule; images are refused by name until the vision slice (C6).
    fn opening(x: *Mini, s: *Stream) !*Lane {
        const p = s.prompt();
        if (p.len == 0) return error.EmptyPrompt;
        if (std.mem.indexOfScalar(u32, p, x.image_token) != null) return error.Inkling2MiniImages;
        if (!fits(x.m.context, p.len, @as(usize, s.max_new))) return error.PastContext;
        const got = try x.lanes.getOrPut(x.gpa, s);
        if (!got.found_existing) {
            const slot = x.freeSlot() orelse {
                x.lanes.removeByPtr(got.key_ptr);
                return error.NoFreeSlot;
            };
            got.value_ptr.* = .{ .slot = slot, .last = x.slot_bufs[2 * slot].ptr, .rule = x.slot_bufs[2 * slot + 1].ptr };
            x.free[slot] = false;
        }
        const l = got.value_ptr;
        l.sampled = false;
        if (cuda.sample.ruleFor(s.sampling)) |r| {
            try x.upload(l.rule, std.mem.asBytes(&r));
            l.sampled = true;
        }
        s.cached = 0;
        l.fill = x.resumeAt(s, l, p.len);
        s.fill_at.store(@intCast(l.fill.?), .monotonic);
        return l;
    }

    /// The kept state's position, restored into the slot (0: none, not this prompt's, or another plan's).
    fn resumeAt(x: *Mini, s: *Stream, l: *const Lane, len: usize) usize {
        const saved = s.reuse.saved orelse return 0;
        const k: *const snap.Saved = @ptrCast(@alignCast(saved));
        if (k.at != s.reuse.at or k.at == 0 or k.at >= len) {
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

    /// The prompt pass stands at `at` (a chunk start or a mark): copy the stream's state there.
    pub fn saveState(x: *Mini, owner: ?*anyopaque, at: u32) !*snap.Saved {
        const s: *const Stream = @ptrCast(@alignCast(owner orelse return error.NoOwner));
        const l = x.lanes.getPtr(s) orelse return error.UnknownStream;
        if ((l.fill orelse l.len) != at) return error.NotAtMark; // a state is copied only where the pass stands
        return snap.save(x.gpa, x.m, x.s, l.slot, at);
    }

    pub fn restoreState(x: *Mini, owner: ?*anyopaque, k: *const snap.Saved) !void {
        const s: *const Stream = @ptrCast(@alignCast(owner orelse return error.NoOwner));
        const l = x.lanes.getPtr(s) orelse return error.UnknownStream;
        try snap.restore(x.m, x.s, l.slot, k);
        l.len = k.at;
    }

    pub fn dropState(x: *Mini, k: *snap.Saved) void {
        snap.drop(x.gpa, x.s, k);
    }

    /// One chunk from `l.fill`; true once the prompt's last row is in.
    fn chunk(x: *Mini, s: *Stream, l: *Lane) !bool {
        const p = s.prompt();
        const at = l.fill.?;
        if (s.isCancelled()) return error.Cancelled;
        var end = @min(p.len, (at / prompt_rows + 1) * prompt_rows);
        for (s.reuse.marks) |mk| if (mk > at and mk < end) {
            end = mk;
        };
        const done = end == p.len;
        try x.m.forward(x.k, x.s, &.{.{ .slot = l.slot, .p0 = at, .tokens = p[at..end] }}, true, end - at - 1, @intFromBool(done));
        if (done) try x.d.check(x.d.api.cuMemcpyDtoDAsync_v2(l.last, x.m.s.logits, x.vocab() * 4, x.s.handle), "cuMemcpyDtoDAsync");
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

    fn handle(x: *Mini) usize {
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

    fn value(x: *Mini, feed: be.Feed) !u32 {
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
        try x.m.forward(x.k, x.s, &.{.{ .slot = l.slot, .p0 = l.len, .tokens = &.{t} }}, false, 0, 1);
        l.len += 1;
        const h = x.next;
        try x.drawRows(l, x.m.s.logits, 0, 1, position, x.handle(), 0);
        return h;
    }

    fn read(ptr: *anyopaque, h: u64) anyerror!u32 {
        const x = self(ptr);
        if (h >= x.next or x.next - h > ring) return error.NoSuchToken;
        var out: [1]u32 = undefined;
        try x.s.synchronize();
        try x.tok.download(@intCast((h % ring) * 4), std.mem.sliceAsBytes(&out));
        return out[0];
    }

    fn verify(ptr: *anyopaque, windows: []const be.Window, out: []be.Verified) anyerror!void {
        const x = self(ptr);
        const chains = try x.gpa.alloc(mm.Chain, windows.len);
        defer x.gpa.free(chains);
        var total: usize = 0;
        for (windows, chains) |w, *c| {
            if (w.held != 0) return error.NoHeldDrafts; // no draft head
            if (w.parents != null) return error.ChainsOnly;
            if (w.masks.len != 0) return error.StructuresUnsupported;
            const l = x.lanes.getPtr(w.stream) orelse return error.UnknownStream;
            if (w.rows() > mm.window_rows) return error.WindowTooWide;
            l.base = l.len;
            l.rows.clearRetainingCapacity();
            try l.rows.append(x.gpa, w.pending);
            try l.rows.appendSlice(x.gpa, w.tokens);
            for (w.positions, 0..) |pos, r| if (pos != l.base + r + 1) return error.PositionMismatch;
            c.* = .{ .slot = l.slot, .p0 = l.base, .tokens = l.rows.items };
            total += l.rows.items.len;
        }
        if (total > round_rows) return error.RoundTooWide;
        try x.m.forward(x.k, x.s, chains, false, 0, total);
        var at: usize = 0;
        for (windows, 0..) |w, i| {
            const l = x.lanes.getPtr(w.stream).?;
            try x.drawRows(l, x.m.s.logits, at, l.rows.items.len, w.positions[0], ring + at, i);
            at += l.rows.items.len;
        }
        var got: [round_rows]u32 = undefined;
        try x.s.synchronize();
        try x.tok.download(ring * 4, std.mem.sliceAsBytes(got[0..total]));
        at = 0;
        for (windows, out) |w, o| {
            const l = x.lanes.getPtr(w.stream).?;
            const n = l.rows.items.len;
            @memcpy(o.sampled[0..n], got[at..][0..n]);
            @memcpy(o.drafts[0 .. n - 1], l.rows.items[1..]);
            l.len = l.base + n;
            at += n;
        }
    }

    /// Every window keeps all its rows: the conv states hold no history to roll back to (no drafts are offered).
    fn keep(ptr: *anyopaque, windows: []const be.Window, paths: []const []const u32) anyerror!void {
        const x = self(ptr);
        for (windows, paths) |w, path| {
            const l = x.lanes.getPtr(w.stream) orelse return error.UnknownStream;
            if (path.len != l.rows.items.len) return error.Inkling2MiniNoRollback;
            for (path, 0..) |r, i| if (r != i) return error.ChainsOnly;
        }
    }

    /// No draft head: absorbing is a no-op, and a request for drafts is refused by name.
    fn draft(_: *anyopaque, requests: []const be.DraftRequest) anyerror!void {
        for (requests) |r| if (r.depth != 0 or r.lanes != null) return error.NoDraftHead;
    }

    fn dropLane(x: *Mini, s: *const Stream) void {
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

test "a slot holds the window in whole 1,024s; a reply past it does not fit" {
    try std.testing.expectEqual(@as(usize, 32768), try capacity(32768));
    try std.testing.expectEqual(@as(usize, 1024), try capacity(1000));
    try std.testing.expectError(error.BadContext, capacity(0));
    try std.testing.expect(fits(32768, 30000, 2768));
    try std.testing.expect(!fits(32768, 30000, 2769));
    try std.testing.expect(!fits(32768, 32769, 0));
}
