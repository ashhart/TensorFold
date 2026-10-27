//! LFM3 on CUDA behind lanes.Backend: a slot a stream, a chunk a prompt step, one forward a round of windows.
const std = @import("std");
const cuda = @import("cuda");
const lanes = @import("lanes");
const cfg = @import("config.zig");
const fwd = @import("cuda_forward.zig");
const mdl = @import("cuda_model.zig");
const kern = @import("cuda_kernels.zig");
const kv = @import("core").kv_pool;
const snap = @import("cuda_snap.zig");

const be = lanes.backend;
const Stream = lanes.Stream;
const Pick = cuda.tree_round.Pick;

/// Prompt rows a step takes: the forward's chunk grid, from position 0.
pub const prompt_rows = mdl.prompt_rows;
/// Window rows a round takes over every stream.
pub const round_rows = mdl.round_rows;
/// The widest window a stream verifies: one-row steps and every wider window give a row the same bits.
pub const max_window = 16;
/// Drawn tokens a handle names (a ring; read before the round loop could wrap it).
const ring = 1024;

const Lane = struct {
    slot: usize,
    len: u64 = 0, // the cache length: positions [0, len) hold this stream's keys
    base: u64 = 0, // the cache length the last verify started from
    row0: usize = 0, // the stream's first row in the last verify's forward (its conv snapshots)
    rows: std.ArrayList(u32) = .empty, // the last verify's row tokens
    last: u64, // the prompt's last row's bf16 logits [V] (the first draw reads them)
    rule: u64, // sample.cu's Rule, when the stream samples
    sampled: bool = false,
    fill: ?usize = null, // a prompt pass going a chunk at a time: where its next chunk starts
    lease: kv.Lease = .{}, // a paged slot's pages
};

/// Pages a request holds: its prompt, its reply and one last verify window.
pub fn pagesFor(prompt: usize, max_new: usize) usize {
    return (prompt + max_new + max_window + kern.page - 1) / kern.page;
}

pub const Lfm3 = struct {
    gpa: std.mem.Allocator,
    d: *const cuda.Driver,
    m: *mdl.Model,
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
    pool: ?kv.Pool = null, // the pages every slot leases (attach)

    /// The device bytes `init` allocates: the token and pick rings, metas, each slot's last row and rule.
    pub fn openBytes(slots: usize) usize {
        return (ring + round_rows) * (4 + @sizeOf(Pick)) + round_rows * 16 + slots * (cfg.vocab * 2 + @sizeOf(cuda.sample.Rule));
    }

    pub fn init(gpa: std.mem.Allocator, d: *const cuda.Driver, m: *mdl.Model, k: *const kern.Kernels, s: cuda.Stream) !Lfm3 {
        const free = try gpa.alloc(bool, m.slotCap());
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
        const bufs = try gpa.alloc(cuda.DeviceBuffer, 2 * m.slotCap());
        errdefer gpa.free(bufs);
        var made: usize = 0;
        errdefer for (bufs[0..made]) |*b| b.free();
        for (bufs, 0..) |*b, i| {
            b.* = try cuda.DeviceBuffer.alloc(d, if (i % 2 == 0) cfg.vocab * 2 else @sizeOf(cuda.sample.Rule));
            made += 1;
        }
        return .{ .gpa = gpa, .d = d, .m = m, .k = k, .s = s, .free = free, .tok = tok, .picks = picks, .greedy = greedy, .meta = meta, .slot_bufs = bufs };
    }

    /// Makes a paged model's arena of `pages` pages for `extra + 1` slots, and the pool that leases them.
    pub fn attach(x: *Lfm3, extra: usize, pages: usize) !void {
        try x.m.attach(x.s, extra, pages);
        x.pool = try kv.Pool.init(x.gpa, pages, mdl.page_elems * 2);
    }

    pub fn deinit(x: *Lfm3) void {
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
        x.gpa.free(x.greedy);
        x.tok.free();
        x.picks.free();
        x.meta.free();
    }

    pub fn backend(x: *Lfm3) be.Backend {
        return .{ .ptr = x, .vtable = &vtable };
    }

    const vtable: be.Backend.VTable = .{ .prefill = prefill, .prefill_step = prefillStep, .first = first, .queue = queue, .read = read, .verify = verify, .keep = keep, .draft = draft, .release = release, .reports_fill = true };

    /// The round loop's facts: no draft head; host copies verified in windows up to 16 rows, streams sharing a forward.
    pub fn facts(x: *const Lfm3) lanes.Model {
        return .{ .exact_width = max_window, .gpu_tokens = false, .mtp = false, .speculate = false, .hidden_rows = true, .batch_rows = round_rows, .max_streams = @intCast(@min(x.m.slotCap(), round_rows)) };
    }

    fn self(ptr: *anyopaque) *Lfm3 {
        return @ptrCast(@alignCast(ptr));
    }

    fn upload(x: *Lfm3, dst: u64, bytes: []const u8) !void {
        try x.d.check(x.d.api.cuMemcpyHtoDAsync_v2(dst, bytes.ptr, bytes.len, x.s.handle), "cuMemcpyHtoDAsync");
    }

    /// Rows [row0, row0 + rows) of bf16 `logits` into entries [at, at + rows), row r at `position0 + r`.
    fn drawRows(x: *Lfm3, l: *const Lane, logits: u64, row0: usize, rows: usize, position0: u64, at: usize, meta_at: usize) !void {
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
    fn tokens(x: *Lfm3, at: usize, out: []u32) !void {
        try x.s.synchronize();
        var picks: [round_rows]Pick = undefined;
        try x.tok.download(at * 4, std.mem.sliceAsBytes(out));
        try x.picks.download(at * @sizeOf(Pick), std.mem.sliceAsBytes(picks[0..out.len]));
        for (out, picks[0..out.len], x.greedy[at..][0..out.len]) |*t, p, g| if (g) {
            if (p.nonfinite != 0) return error.NonFiniteLogits;
            t.* = p.token;
        };
    }

    /// The stream's state at `at` for the prompt cache, copied behind the chunk that ends there.
    pub fn saveState(x: *Lfm3, owner: ?*anyopaque, at: u32) !*snap.Saved {
        const s: *const Stream = @ptrCast(@alignCast(owner orelse return error.NoOwner));
        const l = x.lanes.getPtr(s) orelse return error.UnknownStream;
        if ((l.fill orelse l.len) != at) return error.NotAtMark; // a state is copied only where the pass stands
        return snap.save(x.gpa, x.m, x.s, l.slot, at);
    }

    /// The stream's slot made to hold `k`'s state (a restore outside its prompt pass).
    pub fn restoreState(x: *Lfm3, owner: ?*anyopaque, k: *const snap.Saved) !void {
        const s: *const Stream = @ptrCast(@alignCast(owner orelse return error.NoOwner));
        const l = x.lanes.getPtr(s) orelse return error.UnknownStream;
        try snap.restore(x.m, x.s, l.slot, k);
        l.len = k.at;
    }

    pub fn dropState(x: *Lfm3, k: *snap.Saved) void {
        snap.drop(x.gpa, x.s, k);
    }

    fn freeSlot(x: *const Lfm3) ?usize {
        return std.mem.indexOfScalar(bool, x.free[0..x.m.slots], true);
    }

    fn fitsContext(x: *const Lfm3, prompt: usize, max_new: usize) bool {
        return prompt + max_new + max_window <= x.m.context;
    }

    /// Whether a request could start now: a free slot and (paged) its pages; one that never fits passes.
    pub fn admits(x: *const Lfm3, prompt: usize, max_new: usize) bool {
        if (!x.fitsContext(prompt, max_new)) return true; // its prompt pass refuses it by name
        const p = x.pool orelse return x.freeSlot() != null;
        return x.freeSlot() != null and p.fits(pagesFor(prompt, max_new));
    }

    const Map = struct {
        x: *Lfm3,
        slot: usize,
        fn write(ctx: *anyopaque, ids: []const u32, from: usize) anyerror!void {
            const m: *const Map = @ptrCast(@alignCast(ctx));
            try m.x.m.mapPages(m.x.s, m.slot, ids[from..], from);
        }
    };

    /// A free slot for the request, its pages leased now when paged; LaneMemory: not yet.
    fn takeSlot(x: *Lfm3, lease: *kv.Lease, prompt: usize, max_new: usize) !usize {
        const slot = x.freeSlot() orelse return error.NoFreeSlot;
        const p = if (x.pool) |*p| p else return slot;
        const n = pagesFor(prompt, max_new);
        try p.promise(lease, n);
        errdefer p.finish(lease);
        var map: Map = .{ .x = x, .slot = slot };
        try p.cover(lease, n, .{ .ctx = &map, .write = Map.write });
        return slot;
    }

    /// A prompt pass begins: a slot, the stream's rule, a kept state restored; where its first chunk starts.
    fn opening(x: *Lfm3, s: *Stream) !*Lane {
        const p = s.prompt();
        if (p.len == 0) return error.EmptyPrompt;
        if (s.images.len > 0) return error.Lfm3TextOnly;
        if (!x.fitsContext(p.len, s.max_new)) return error.PastContext;
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
        try x.m.clearSlot(x.s, l.slot);
        l.fill = x.resumeAt(s, l, p.len);
        s.fill_at.store(@intCast(l.fill.?), .monotonic);
        return l;
    }

    /// The kept state's position, restored into the slot (0: none, or not this prompt's).
    fn resumeAt(x: *Lfm3, s: *Stream, l: *const Lane, len: usize) usize {
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

    /// One chunk from `l.fill` to the next grid line or mark; true once the prompt's last row's logits are written.
    fn chunk(x: *Lfm3, s: *Stream, l: *Lane) !bool {
        const p = s.prompt();
        const at = l.fill.?;
        if (s.isCancelled()) return error.Cancelled;
        var end = @min(p.len, (at / prompt_rows + 1) * prompt_rows);
        for (s.reuse.marks) |mk| if (mk > at and mk < end) {
            end = mk;
        };
        const done = end == p.len;
        try fwd.forward(x.m, x.k, x.s, &.{.{ .slot = l.slot, .p0 = at, .tokens = p[at..end] }}, true, end - at - 1, @intFromBool(done));
        if (done) try x.d.check(x.d.api.cuMemcpyDtoDAsync_v2(l.last, x.m.s.logits, cfg.vocab * 2, x.s.handle), "cuMemcpyDtoDAsync");
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

    fn handle(x: *Lfm3) usize {
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

    fn value(x: *Lfm3, feed: be.Feed) !u32 {
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
        try fwd.forward(x.m, x.k, x.s, &.{.{ .slot = l.slot, .p0 = l.len, .tokens = &.{t} }}, false, 0, 1);
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
            l.row0 = total;
            total += l.rows.items.len;
        }
        if (total > round_rows) return error.RoundTooWide;
        try fwd.forward(x.m, x.k, x.s, chains, false, 0, total);
        var at: usize = 0;
        for (windows, 0..) |w, i| {
            const l = x.lanes.getPtr(w.stream).?;
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
        if (windows.len > round_rows) return error.RoundTooWide;
        var slots: [round_rows]i32 = undefined;
        var rows: [round_rows]i32 = undefined;
        for (windows, paths, 0..) |w, path, i| {
            const l = x.lanes.getPtr(w.stream) orelse return error.UnknownStream;
            if (path.len == 0 or path.len > l.rows.items.len) return error.EmptyPath;
            for (path, 0..) |r, j| if (r != j) return error.ChainsOnly; // a chain keeps a prefix of its rows
            l.len = l.base + path.len; // caches are positional: later rows overwrite what the rejected ones wrote
            slots[i] = @intCast(l.slot);
            rows[i] = @intCast(l.row0 + path.len - 1); // the conv states after the last kept row
        }
        try fwd.commit(x.m, x.k, x.s, slots[0..windows.len], rows[0..windows.len]);
    }

    /// No draft head: absorbing is a no-op, a request for drafts is refused by name (copy drafts come from the host).
    fn draft(_: *anyopaque, requests: []const be.DraftRequest) anyerror!void {
        for (requests) |r| if (r.depth != 0 or r.lanes != null) return error.NoDraftHead;
    }

    fn dropLane(x: *Lfm3, s: *const Stream) void {
        if (x.lanes.fetchRemove(s)) |entry| {
            var l = entry.value;
            if (x.pool) |*p| {
                p.finish(&l.lease);
                x.m.unmapPages(l.slot);
            }
            x.free[l.slot] = true;
            l.rows.deinit(x.gpa);
            if (x.lanes.count() == 0) x.report();
        }
    }

    /// The idle server's line: nothing promised, every page free (kept states are copies, not pages).
    fn report(x: *const Lfm3) void {
        if (@import("builtin").is_test) return;
        const p = x.pool orelse return;
        const c = p.census();
        const ok = p.idle() and c.free == c.total;
        std.log.info("kv pool idle{s}: promised {d}, free {d} + store {d} == total {d}", .{ if (ok) "" else " (UNEVEN)", c.promised, c.free, c.kept, c.total });
    }

    fn release(ptr: *anyopaque, s: *Stream) void {
        self(ptr).dropLane(s);
    }
};

test "a request holds its prompt, its reply and one verify window in whole pages" {
    try std.testing.expectEqual(@as(usize, 1), pagesFor(1, 1));
    try std.testing.expectEqual(@as(usize, 1), pagesFor(400, 96));
    try std.testing.expectEqual(@as(usize, 2), pagesFor(400, 97));
    try std.testing.expectEqual(@as(usize, 65), pagesFor(32768, 0));
}
