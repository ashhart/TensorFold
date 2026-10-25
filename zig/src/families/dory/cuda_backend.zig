//! Dory behind lanes.Backend: a cache slot a stream, a prompt chunk a step, one forward a round of windows.
const std = @import("std");
const cuda = @import("cuda");
const lanes = @import("lanes");
const fwd = @import("cuda_model.zig");
const snap = @import("cuda_snap.zig");
const pages = @import("cuda_pages.zig");
const kv = @import("core").kv_pool;

const be = lanes.backend;
const Stream = lanes.Stream;
const Kernels = fwd.Kernels;

/// Prompt rows a step takes: the forward's chunk grid, from position 0.
pub const prompt_rows = fwd.prompt_rows;
/// Window rows a round takes over every stream.
pub const round_rows = fwd.round_rows;
/// The widest window a stream verifies: one-row steps and every wider window give a row the same bits.
pub const max_window = fwd.window_rows;
/// Positions a slot keeps past a reply for a verify window's rows.
pub const reserve = max_window;
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
    lease: kv.Lease = .{}, // a paged slot's pages
};

/// The physical positions a slot holds for a served window: the window, the verify reserve, in whole 1,024s.
pub fn capacity(window: usize) !usize {
    if (window == 0) return error.BadContext;
    return (try std.math.add(usize, window, reserve + 1023)) / 1024 * 1024;
}

/// A prompt and its reply fit `physical` positions with the verify reserve left over.
pub fn fits(physical: usize, prompt: usize, reply: usize) bool {
    return prompt <= physical and reply <= physical - prompt and reserve <= physical - prompt - reply;
}

pub const Dory = struct {
    gpa: std.mem.Allocator,
    d: *const cuda.Driver,
    m: *fwd.Model,
    k: *const Kernels,
    s: cuda.Stream,
    lanes: std.AutoHashMapUnmanaged(*const Stream, Lane) = .empty,
    free: []bool, // slot i is free
    logits16: cuda.DeviceBuffer, // a round's logits as bf16 for keyed draws
    tok: cuda.DeviceBuffer, // drawn tokens: the handle ring, then a round's rows
    meta: cuda.DeviceBuffer, // a draw's first position minus one, a window each
    slot_bufs: []cuda.DeviceBuffer, // a slot's last-row logits, then its rule: two a slot
    next: u64 = 0,
    pool: ?kv.Pool = null, // the pages every slot leases (attach)

    /// The device bytes `init` allocates: bf16 logits, the token ring, metas, each slot's last row and rule.
    pub fn openBytes(vocab: usize, slots: usize) usize {
        return round_rows * vocab * 2 + (ring + round_rows) * 4 + round_rows * 16 + slots * (vocab * 4 + @sizeOf(cuda.sample.Rule));
    }

    pub fn init(gpa: std.mem.Allocator, d: *const cuda.Driver, m: *fwd.Model, k: *const Kernels, s: cuda.Stream) !Dory {
        const v = m.c.vocab;
        const free = try gpa.alloc(bool, m.slotCap());
        errdefer gpa.free(free);
        @memset(free, true);
        var logits16 = try cuda.DeviceBuffer.alloc(d, round_rows * v * 2);
        errdefer logits16.free();
        var tok = try cuda.DeviceBuffer.alloc(d, (ring + round_rows) * 4);
        errdefer tok.free();
        var meta = try cuda.DeviceBuffer.alloc(d, round_rows * 16);
        errdefer meta.free();
        const bufs = try gpa.alloc(cuda.DeviceBuffer, 2 * m.slotCap());
        errdefer gpa.free(bufs);
        var made: usize = 0;
        errdefer for (bufs[0..made]) |*b| b.free();
        for (bufs, 0..) |*b, i| {
            b.* = try cuda.DeviceBuffer.alloc(d, if (i % 2 == 0) v * 4 else @sizeOf(cuda.sample.Rule));
            made += 1;
        }
        return .{ .gpa = gpa, .d = d, .m = m, .k = k, .s = s, .free = free, .logits16 = logits16, .tok = tok, .meta = meta, .slot_bufs = bufs };
    }

    /// Makes a paged model's arena of `pool_pages` pages for `extra + 1` slots, and the pool that leases them.
    pub fn attach(x: *Dory, extra: usize, pool_pages: usize) !void {
        if (x.m.paging == null) try x.m.attach(x.s, extra, pool_pages); // a check may have attached it already
        x.pool = try kv.Pool.init(x.gpa, x.m.paging.?.pool, x.m.paging.?.pageBytes());
    }

    /// The pool's free pages in a seeded random order, so tables are not ascending (a check's hook).
    pub fn shufflePages(x: *Dory, seed: u64) void {
        var prng = std.Random.DefaultPrng.init(seed);
        if (x.pool) |*p| prng.random().shuffle(u32, p.free[0..p.nfree]);
    }

    pub fn deinit(x: *Dory) void {
        var it = x.lanes.valueIterator();
        while (it.next()) |l| {
            if (x.pool) |*p| p.finish(&l.lease);
            l.rows.deinit(x.gpa);
        }
        if (x.pool) |*p| p.deinit();
        x.lanes.deinit(x.gpa);
        for (x.slot_bufs) |*b| b.free();
        x.gpa.free(x.slot_bufs);
        x.gpa.free(x.free);
        x.logits16.free();
        x.tok.free();
        x.meta.free();
    }

    pub fn backend(x: *Dory) be.Backend {
        return .{ .ptr = x, .vtable = &vtable };
    }

    const vtable: be.Backend.VTable = .{ .prefill = prefill, .prefill_step = prefillStep, .first = first, .queue = queue, .read = read, .verify = verify, .keep = keep, .draft = draft, .release = release, .reports_fill = true };

    /// The round loop's facts: no draft head; host copies verified in windows up to 16 rows, streams sharing a forward.
    pub fn facts(x: *const Dory) lanes.Model {
        return .{ .exact_width = max_window, .gpu_tokens = false, .mtp = false, .speculate = false, .hidden_rows = true, .batch_rows = round_rows, .max_streams = @intCast(@min(x.m.slotCap(), round_rows)) };
    }

    fn self(ptr: *anyopaque) *Dory {
        return @ptrCast(@alignCast(ptr));
    }

    fn upload(x: *Dory, dst: u64, bytes: []const u8) !void {
        try cuda.copy_source.enqueue(x.d, dst, bytes, x.s);
    }

    /// Rows [row0, row0 + rows) of fp32 `logits` drawn into tok[at..], row r at `position0 + r` (argmax when greedy).
    fn drawRows(x: *Dory, l: *const Lane, logits: u64, row0: usize, rows: usize, position0: u64, at: usize, meta_at: usize) !void {
        const v = x.m.c.vocab;
        const src = logits + row0 * v * 4;
        if (!l.sampled) return x.k.argmax(x.s, src, rows, v, x.tok.ptr + at * 4);
        try x.k.toBf16(x.s, src, x.logits16.ptr, rows * v);
        const meta = [4]i32{ @intCast(position0 - 1), 0, 0, 0 };
        const mp = x.meta.ptr + meta_at * 16;
        try x.upload(mp, std.mem.asBytes(&meta));
        try x.k.draws.draw(x.s, x.logits16.ptr, v, l.rule, mp, 0, x.tok.ptr + at * 4, rows, null, null);
    }

    /// The stream's caches at `at` for the prompt cache, copied behind the chunk that ends there.
    pub fn saveState(x: *Dory, owner: ?*anyopaque, at: u32) !*snap.Saved {
        const s: *const Stream = @ptrCast(@alignCast(owner orelse return error.NoOwner));
        const l = x.lanes.getPtr(s) orelse return error.UnknownStream;
        if ((l.fill orelse l.len) != at) return error.NotAtMark; // a state is copied only where the pass stands
        return snap.save(x.gpa, x.m, x.s, l.slot, at);
    }

    /// The stream's slot made to hold `k`'s state (a restore outside its prompt pass).
    pub fn restoreState(x: *Dory, owner: ?*anyopaque, k: *const snap.Saved) !void {
        const s: *const Stream = @ptrCast(@alignCast(owner orelse return error.NoOwner));
        const l = x.lanes.getPtr(s) orelse return error.UnknownStream;
        try snap.restore(x.m, x.s, l.slot, k);
        l.len = k.at;
    }

    pub fn dropState(x: *Dory, k: *snap.Saved) void {
        snap.drop(x.gpa, x.s, k);
    }

    fn freeSlot(x: *const Dory) ?usize {
        return std.mem.indexOfScalar(bool, x.free[0..x.m.slots], true);
    }

    /// Pages a request of this prompt and reply holds.
    fn pagesFor(prompt: usize, max_new: usize) usize {
        return pages.need(prompt, max_new, reserve);
    }

    /// Whether a request could start now: a free slot and (paged) its pages; one that never fits passes.
    pub fn admits(x: *const Dory, prompt: usize, max_new: usize) bool {
        if (!fits(x.m.context, prompt, max_new)) return true;
        const p = x.pool orelse return x.freeSlot() != null;
        return x.freeSlot() != null and p.fits(pagesFor(prompt, max_new));
    }

    const Map = struct {
        x: *Dory,
        slot: usize,
        fn write(ctx: *anyopaque, ids: []const u32, from: usize) anyerror!void {
            const m: *const Map = @ptrCast(@alignCast(ctx));
            try m.x.m.mapPages(m.x.s, m.slot, ids[from..], from);
        }
    };

    /// A free slot for the request, its pages leased now when paged.
    fn takeSlot(x: *Dory, lease: *kv.Lease, prompt: usize, max_new: usize) !usize {
        const slot = x.freeSlot() orelse return error.NoFreeSlot;
        if (x.pool == null) return slot;
        const p = &x.pool.?;
        const n = pagesFor(prompt, max_new);
        try p.promise(lease, n);
        errdefer p.finish(lease);
        var map: Map = .{ .x = x, .slot = slot };
        try p.cover(lease, n, .{ .ctx = &map, .write = Map.write });
        return slot;
    }

    /// A prompt pass begins: a slot, the stream's rule, a kept state restored; where its first chunk starts.
    fn opening(x: *Dory, s: *Stream) !*Lane {
        const p = s.prompt();
        if (p.len == 0) return error.EmptyPrompt;
        if (!fits(x.m.context, p.len, @as(usize, s.max_new))) return error.PastContext;
        const got = try x.lanes.getOrPut(x.gpa, s);
        if (!got.found_existing) {
            var lease: kv.Lease = .{};
            const slot = x.takeSlot(&lease, p.len, s.max_new) catch |e| {
                x.lanes.removeByPtr(got.key_ptr);
                return e;
            };
            x.free[slot] = false;
            got.value_ptr.* = .{ .slot = slot, .last = x.slot_bufs[2 * slot].ptr, .rule = x.slot_bufs[2 * slot + 1].ptr, .lease = lease };
        }
        const l = got.value_ptr;
        l.sampled = false;
        if (s.sampling) |sp| if (sp.temperature > 0) {
            const r = cuda.sample.Rule.from(lanes.gpu_rule.rule(sp));
            try x.upload(l.rule, std.mem.asBytes(&r));
            l.sampled = true;
        };
        s.cached = 0;
        l.fill = x.resumeAt(s, l, p.len);
        s.fill_at.store(@intCast(l.fill.?), .monotonic);
        return l;
    }

    /// The kept state's position, restored into the slot (0: none, not this prompt's, or another loop count's).
    fn resumeAt(x: *Dory, s: *Stream, l: *const Lane, len: usize) usize {
        const saved = s.reuse.saved orelse return 0;
        const k: *const snap.Saved = @ptrCast(@alignCast(saved));
        if (k.at != s.reuse.at or k.at == 0 or k.at >= len or !snap.sameKey(k.key, snap.keyOf(x.m.c))) {
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
    fn chunk(x: *Dory, s: *Stream, l: *Lane) !bool {
        const p = s.prompt();
        const at = l.fill.?;
        if (s.isCancelled()) return error.Cancelled;
        var end = @min(p.len, (at / prompt_rows + 1) * prompt_rows);
        for (s.reuse.marks) |mk| if (mk > at and mk < end) {
            end = mk;
        };
        const done = end == p.len;
        try x.m.forward(x.k, x.s, &.{.{ .slot = l.slot, .p0 = at, .tokens = p[at..end] }}, true, end - at - 1, @intFromBool(done));
        if (done) try x.d.check(x.d.api.cuMemcpyDtoDAsync_v2(l.last, x.m.s.logits, x.m.c.vocab * 4, x.s.handle), "cuMemcpyDtoDAsync");
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

    fn prefillStep(ptr: *anyopaque, s: *Stream) anyerror!bool {
        const x = self(ptr);
        const going = if (x.lanes.getPtr(s)) |l| l.fill != null else false;
        const l = if (going) x.lanes.getPtr(s).? else try x.opening(s);
        errdefer x.dropLane(s);
        return x.chunk(s, l);
    }

    fn handle(x: *Dory) usize {
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

    fn value(x: *Dory, feed: be.Feed) !u32 {
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

    /// No draft head: absorbing is a no-op, and a request for drafts is refused by name.
    fn draft(_: *anyopaque, requests: []const be.DraftRequest) anyerror!void {
        for (requests) |r| if (r.depth != 0 or r.lanes != null) return error.NoDraftHead;
    }

    fn dropLane(x: *Dory, s: *const Stream) void {
        if (x.lanes.fetchRemove(s)) |entry| {
            var l = entry.value;
            if (x.pool) |*p| {
                p.finish(&l.lease);
                x.m.unmapPages(l.slot);
            }
            x.free[l.slot] = true;
            l.rows.deinit(x.gpa);
        }
    }

    fn release(ptr: *anyopaque, s: *Stream) void {
        self(ptr).dropLane(s);
    }
};

test "a slot holds the window and a verify window past it, in whole 1,024s; a reply past them does not fit" {
    try std.testing.expectEqual(@as(usize, 33792), try capacity(32768));
    try std.testing.expectEqual(@as(usize, 1024), try capacity(1000));
    try std.testing.expectError(error.BadContext, capacity(0));
    const physical = try capacity(32768);
    try std.testing.expect(fits(physical, 30000, 2768));
    try std.testing.expect(fits(physical, physical - reserve - 1, 1));
    try std.testing.expect(!fits(physical, physical - reserve, 1));
    try std.testing.expect(!fits(physical, physical + 1, 0));
}

test "Dory queued upload: Dory inputs survive caller mutation until stream completion" {
    const Send = struct {
        fn send(d: *const cuda.Driver, s: cuda.Stream, bytes: []const u8) !void {
            var x: Dory = undefined;
            x.d = d;
            x.s = s;
            try x.upload(0x1000, bytes);
        }
    };
    try @import("cuda_upload_test.zig").check(Send.send);
}
