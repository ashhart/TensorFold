//! Kolibri-1 behind lanes.Backend: a cache slot a stream, prompts a chunk a step, one forward a round of windows.
const std = @import("std");
const cuda = @import("cuda");
const lanes = @import("lanes");
const fwd = @import("forward.zig");
const kern = @import("kernels.zig");
const snap = @import("snap.zig");
const limits = @import("context.zig");
const kv = @import("core").kv_pool;
const kp = @import("core").kv_prompts;

const be = lanes.backend;
const Stream = lanes.Stream;

/// Prompt rows a step takes: the forward's chunk grid, from position 0.
pub const prompt_rows = fwd.prompt_rows;
/// Window rows a round takes over every stream.
pub const round_rows = fwd.round_rows;
/// The widest window a stream verifies: one-row steps and every wider window give a row the same bits.
pub const max_window = limits.reserve;
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
    lease: kv.Lease = .{}, // paged: the full layers' pages it holds and is promised
    marks: []const u32 = &.{}, // TF_KV_SHARE: where its pass keeps states, the store's plan (owned)
    from: u32 = 0, // TF_KV_SHARE: the position its adopted pages and restored rings resume at
    kept: u32 = 0, // TF_KV_SHARE: states its pass kept
};

/// Pages a request holds at most: its prompt, its reply and the verify reserve.
pub fn pagesFor(prompt: usize, max_new: usize) usize {
    return pagesOf(prompt + max_new + max_window);
}

/// Pages that hold positions [0, positions).
fn pagesOf(positions: usize) usize {
    return (positions + fwd.page_tokens - 1) / fwd.page_tokens;
}

pub const Kolibri = struct {
    gpa: std.mem.Allocator,
    d: *const cuda.Driver,
    m: *fwd.Model,
    k: *const kern.Kernels,
    s: cuda.Stream,
    lanes: std.AutoHashMapUnmanaged(*const Stream, Lane) = .empty,
    free: []bool, // slot i is free
    logits: cuda.DeviceBuffer, // fp32 [round_rows, V]
    logits16: cuda.DeviceBuffer, // their bf16 copy for keyed draws
    tok: cuda.DeviceBuffer, // drawn tokens: the handle ring, then a round's rows
    meta: cuda.DeviceBuffer, // a draw's first position minus one, a window each
    slot_bufs: []cuda.DeviceBuffer, // a slot's last-row logits, then its rule: two a slot
    next: u64 = 0,
    steps: []?fwd.Model.Step, // a slot's one-row step, captured at its first use (null: not yet)
    graphed: bool, // TENSORFOLD_KOLIBRI_GRAPHS=0 runs every step eager
    pool: ?kv.Pool = null, // paged: the pages every slot leases (attach)
    prompts: ?kp.Prompts = null, // TF_KV_SHARE: the store keeping prompts as the pool's pages (cuda_prompt_pages.zig)
    pending: ?*const snap.Saved = null, // TF_KV_SHARE: a resumed state's rings, put in once its lane opens

    /// The device bytes `init` allocates: a round's logits, the token ring, and each slot's last row and rule.
    pub fn openBytes(vocab: usize, slots: usize) usize {
        return round_rows * vocab * 6 + (ring + round_rows) * 4 + round_rows * 16 + slots * (vocab * 4 + @sizeOf(cuda.sample.Rule));
    }

    pub fn init(gpa: std.mem.Allocator, d: *const cuda.Driver, m: *fwd.Model, k: *const kern.Kernels, s: cuda.Stream) !Kolibri {
        const v = m.c.vocab;
        const free = try gpa.alloc(bool, m.slotCap());
        errdefer gpa.free(free);
        @memset(free, true);
        var logits = try cuda.DeviceBuffer.alloc(d, round_rows * v * 4);
        errdefer logits.free();
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
        const steps = try gpa.alloc(?fwd.Model.Step, m.slotCap());
        @memset(steps, null);
        return .{ .gpa = gpa, .d = d, .m = m, .k = k, .s = s, .free = free, .logits = logits, .logits16 = logits16, .tok = tok, .meta = meta, .slot_bufs = bufs, .steps = steps, .graphed = !envOff("TENSORFOLD_KOLIBRI_GRAPHS") };
    }

    /// Makes a paged model's arena of `pages` pages for `extra + 1` slots, and the pool that leases them.
    pub fn attach(x: *Kolibri, extra: usize, pages: usize) !void {
        try x.m.attach(x.s, extra, pages);
        for (x.steps) |*g| if (g.*) |*e| { // a step captured before the arena reads the linear caches attach freed
            e.exec.deinit();
            g.* = null;
        };
        x.pool = try kv.Pool.init(x.gpa, pages, x.m.pageBytes());
    }

    /// The pool's free pages in a seeded random order, leased in that order (a check's hook).
    pub fn shufflePages(x: *Kolibri, seed: u64) void {
        var prng = std.Random.DefaultPrng.init(seed);
        if (x.pool) |*p| prng.random().shuffle(u32, p.free[0..p.nfree]);
    }

    pub fn deinit(x: *Kolibri) void {
        var it = x.lanes.valueIterator();
        while (it.next()) |l| {
            if (x.pool) |*p| p.finish(&l.lease);
            l.rows.deinit(x.gpa);
            x.gpa.free(l.marks);
        }
        if (x.pool) |*p| p.deinit();
        x.lanes.deinit(x.gpa);
        for (x.slot_bufs) |*b| b.free();
        x.gpa.free(x.slot_bufs);
        for (x.steps) |*g| if (g.*) |*e| e.exec.deinit();
        x.gpa.free(x.steps);
        x.gpa.free(x.free);
        x.logits.free();
        x.logits16.free();
        x.tok.free();
        x.meta.free();
    }

    pub fn backend(x: *Kolibri) be.Backend {
        return .{ .ptr = x, .vtable = &vtable };
    }

    const vtable: be.Backend.VTable = .{ .prefill = prefill, .prefill_step = prefillStep, .cancel_prefill = cancelPrefill, .first = first, .queue = queue, .read = read, .verify = verify, .keep = keep, .draft = draft, .release = release, .reports_fill = true };

    /// The round loop's facts: no draft head; host copies verified in windows up to 16 rows, streams sharing a forward.
    pub fn facts(x: *const Kolibri) lanes.Model {
        return .{ .exact_width = max_window, .gpu_tokens = false, .mtp = false, .speculate = false, .hidden_rows = true, .batch_rows = round_rows, .max_streams = @intCast(@min(x.m.slotCap(), round_rows)) };
    }

    fn envOff(name: [*:0]const u8) bool {
        const v = std.c.getenv(name) orelse return false;
        return std.mem.eql(u8, std.mem.span(v), "0");
    }

    fn self(ptr: *anyopaque) *Kolibri {
        return @ptrCast(@alignCast(ptr));
    }

    fn ops(x: *const Kolibri) kern.Ops {
        return .{ .k = x.k, .s = x.s };
    }

    fn upload(x: *Kolibri, dst: u64, bytes: []const u8) !void {
        try x.d.check(x.d.api.cuMemcpyHtoDAsync_v2(dst, bytes.ptr, bytes.len, x.s.handle), "cuMemcpyHtoDAsync");
    }

    /// Rows [row0, row0 + rows) of `logits` drawn into tok[at..], row r at `position0 + r` (argmax when greedy).
    fn drawRows(x: *Kolibri, l: *const Lane, logits: u64, row0: usize, rows: usize, position0: u64, at: usize, meta_at: usize) !void {
        const v = x.m.c.vocab;
        const o = x.ops();
        const src = logits + row0 * v * 4;
        if (!l.sampled) return o.argmax(src, rows, v, x.tok.ptr + at * 4);
        try o.toBf16(src, x.logits16.ptr, rows * v);
        const meta = [4]i32{ @intCast(position0 - 1), 0, 0, 0 };
        const mp = x.meta.ptr + meta_at * 16;
        try x.upload(mp, std.mem.asBytes(&meta));
        try x.k.draws.draw(x.s, x.logits16.ptr, v, l.rule, mp, 0, x.tok.ptr + at * 4, rows, null, null);
    }

    /// The stream's cache state at `at` for the prompt cache, copied behind the chunk that ends there.
    pub fn saveState(x: *Kolibri, owner: ?*anyopaque, at: u32) !*snap.Saved {
        const s: *const Stream = @ptrCast(@alignCast(owner orelse return error.NoOwner));
        const l = x.lanes.getPtr(s) orelse return error.UnknownStream;
        if ((l.fill orelse l.len) != at) return error.NotAtMark; // a state is copied only where the pass stands
        if (x.prompts == null) return snap.save(x.gpa, x.m, x.s, l.slot, at, 0, &.{});
        // TF_KV_SHARE: the store holds [0, at) as pages, and the sliding rings in pages it holds too
        const blob = try x.holdBlob(snap.blobPages(x.m, at, at));
        return snap.save(x.gpa, x.m, x.s, l.slot, at, at, blob) catch |e| {
            for (blob) |id| x.pool.?.release(id);
            x.gpa.free(blob);
            return e;
        };
    }

    /// `n` free pool pages the store holds for a state's rings; NoCachePages: the pool has none, even reclaiming.
    fn holdBlob(x: *Kolibri, n: usize) ![]u32 {
        if (n == 0) return &.{};
        const ids = try x.gpa.alloc(u32, n);
        if (!x.pool.?.hold(ids)) {
            x.gpa.free(ids);
            return error.NoCachePages;
        }
        return ids;
    }

    /// The stream's slot made to hold `k`'s state (a restore outside its prompt pass); TF_KV_SHARE: before its lane.
    pub fn restoreState(x: *Kolibri, owner: ?*anyopaque, k: *const snap.Saved) !void {
        if (x.prompts != null) {
            x.pending = k; // put in once the lane holds the adopted pages (opening)
            return;
        }
        const s: *const Stream = @ptrCast(@alignCast(owner orelse return error.NoOwner));
        const l = x.lanes.getPtr(s) orelse return error.UnknownStream;
        try snap.restore(x.m, x.s, l.slot, k);
        l.len = k.at;
    }

    pub fn dropState(x: *Kolibri, k: *snap.Saved) void {
        if (k.blob.len > 0) {
            x.s.synchronize() catch {}; // a restore's queued copies out of them are done
            for (k.blob) |id| x.pool.?.release(id);
            x.gpa.free(k.blob);
            k.blob = &.{};
        }
        snap.drop(x.gpa, x.s, k);
    }

    /// The lowest free slot.
    fn freeSlot(x: *const Kolibri) ?usize {
        return std.mem.indexOfScalar(bool, x.free[0..x.m.slots], true);
    }

    /// Whether a request could start now: a free slot and (paged) its pages; one that never fits passes.
    pub fn admits(x: *const Kolibri, prompt: usize, max_new: usize) bool {
        if (!limits.fits(x.m.context, prompt, max_new)) return true; // its prompt pass refuses it by name
        const p = x.pool orelse return x.freeSlot() != null;
        return x.freeSlot() != null and p.fits(pagesFor(prompt, max_new));
    }

    const Map = struct {
        x: *Kolibri,
        slot: usize,
        fn write(ctx: *anyopaque, ids: []const u32, from: usize) anyerror!void {
            const m: *const Map = @ptrCast(@alignCast(ctx));
            try m.x.m.mapPages(m.x.s, m.slot, ids[from..], from);
        }
    };

    /// The lane's pages cover positions [0, end): pages taken from its promise as its context grows.
    fn reach(x: *Kolibri, l: *Lane, end: usize) !void {
        const p = if (x.pool) |*p| p else return;
        const n = pagesOf(end);
        if (n <= l.lease.ids.items.len) return;
        var map: Map = .{ .x = x, .slot = l.slot };
        try p.cover(&l.lease, n, .{ .ctx = &map, .write = Map.write });
    }

    /// A free slot for the request, its pages promised now when paged (taken as its rows need them, `held` adopted).
    fn takeSlot(x: *Kolibri, lease: *kv.Lease, prompt: usize, max_new: usize, held: usize) !usize {
        const slot = x.freeSlot() orelse return error.NoFreeSlot;
        if (x.pool) |*p| try p.promise(lease, pagesFor(prompt, max_new) - held);
        return slot;
    }

    /// TF_KV_SHARE: the store's plan for this prompt, its pages in `adopt` and its rings pending; none without a store.
    fn plan(x: *Kolibri, s: *Stream, adopt: *std.ArrayList(u32)) !kp.Plan {
        const pr = x.prompts orelse return .{};
        x.pending = null;
        const got = try pr.begin(s.prompt(), s.history_len, s.shared_prefixes, @ptrCast(s), adopt);
        defer pr.free(got.marks);
        const marks = x.gpa.dupe(u32, got.marks) catch |e| {
            for (adopt.items) |id| x.pool.?.release(id);
            return e;
        };
        return .{ .from = got.from, .marks = marks };
    }

    /// A new lane's table starts with the kept pages, held once more by its lease (rows below `from`, never written).
    fn adoptHeld(x: *Kolibri, l: *Lane, ids: []const u32) !void {
        try x.pool.?.adopt(&l.lease, ids); // releases them again if it fails
        try x.m.mapPages(x.s, l.slot, ids, 0);
    }

    /// A prompt pass begins: a slot, the stream's rule, a kept state restored; where its first chunk starts.
    fn opening(x: *Kolibri, s: *Stream) !*Lane {
        const p = s.prompt();
        if (p.len == 0) return error.EmptyPrompt;
        if (!limits.fits(x.m.context, p.len, @as(usize, s.max_new))) return error.PastContext;
        const got = try x.lanes.getOrPut(x.gpa, s);
        if (!got.found_existing) {
            var lease: kv.Lease = .{};
            var adopt: std.ArrayList(u32) = .empty;
            defer adopt.deinit(x.gpa);
            const kept = x.plan(s, &adopt) catch |e| {
                x.lanes.removeByPtr(got.key_ptr);
                return e;
            };
            const slot = x.takeSlot(&lease, p.len, s.max_new, adopt.items.len) catch |e| {
                for (adopt.items) |id| x.pool.?.release(id);
                x.gpa.free(kept.marks);
                x.lanes.removeByPtr(got.key_ptr);
                return e;
            };
            x.free[slot] = false;
            got.value_ptr.* = .{ .slot = slot, .last = x.slot_bufs[2 * slot].ptr, .rule = x.slot_bufs[2 * slot + 1].ptr, .lease = lease, .marks = kept.marks, .from = kept.from };
            if (adopt.items.len > 0) x.adoptHeld(got.value_ptr, adopt.items) catch |e| {
                x.dropLane(s);
                return e;
            };
        }
        const l = got.value_ptr;
        try x.reach(l, p.len); // the prompt's pages (a kept state's restore writes into them too)
        l.sampled = false;
        if (s.sampling) |sp| if (sp.temperature > 0) {
            const r = cuda.sample.Rule.from(lanes.gpu_rule.rule(sp));
            try x.upload(l.rule, std.mem.asBytes(&r));
            l.sampled = true;
        };
        s.cached = 0;
        l.fill = if (x.prompts != null) try x.resumeShared(s, l) else x.resumeAt(s, l, p.len);
        s.fill_at.store(@intCast(l.fill.?), .monotonic);
        return l;
    }

    /// The kept state's position, restored into the slot (0: none, or not this prompt's); any row resumes exactly.
    fn resumeAt(x: *Kolibri, s: *Stream, l: *const Lane, len: usize) usize {
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

    /// TF_KV_SHARE: where the adopted pages end, the pending rings put in (a failure fails the pass: pages are shared).
    fn resumeShared(x: *Kolibri, s: *Stream, l: *Lane) !usize {
        const k = x.pending orelse return if (l.from == 0) 0 else error.KeptStateMismatch;
        x.pending = null;
        if (k.at != l.from) return error.KeptStateMismatch;
        try snap.restore(x.m, x.s, l.slot, k);
        s.cached = l.from;
        return l.from;
    }

    /// TF_KV_SHARE: at mark `at` the store takes the lane's pages and rings; pages it held already replace the lane's.
    fn keepPages(x: *Kolibri, pr: kp.Prompts, s: *Stream, l: *Lane, at: usize) !void {
        const n = at / fwd.page_tokens;
        const path = try x.gpa.alloc(u32, n);
        defer x.gpa.free(path);
        const mine = try x.gpa.dupe(u32, l.lease.ids.items[0..n]);
        defer x.gpa.free(mine);
        const shared = pr.keep(s.prompt(), @intCast(at), @ptrCast(s), mine, path);
        if (shared == n) l.kept += 1;
        if (std.mem.eql(u32, mine[0..shared], path[0..shared])) return;
        try x.m.mapPages(x.s, l.slot, path[0..shared], 0); // the same tokens' rows: a resume from any row is exact
        for (path[0..shared], 0..) |id, i| x.pool.?.share(&l.lease, i, id);
    }

    /// One chunk from `l.fill` to the next grid line or mark; true once the prompt's last row is in. `_: u32`.
    fn chunk(x: *Kolibri, s: *Stream, l: *Lane, _: u32) !bool {
        const p = s.prompt();
        const at = l.fill.?;
        if (s.isCancelled()) {
            s.keepInterruptedPrefill(@intCast(at));
            return error.Cancelled;
        }
        var end = @min(p.len, (at / prompt_rows + 1) * prompt_rows);
        const marks = if (x.prompts != null) l.marks else s.reuse.marks;
        for (marks) |mk| if (mk > at and mk < end) {
            end = mk;
        };
        try x.m.forward(x.ops(), &.{.{ .slot = l.slot, .p0 = at, .tokens = p[at..end] }}, true, null, l.last);
        s.fill_at.store(@intCast(end), .monotonic);
        const done = end == p.len;
        l.fill = if (done) null else end;
        if (done) l.len = p.len;
        if (x.prompts) |pr| {
            if (!done and std.mem.indexOfScalar(u32, l.marks, @intCast(end)) != null) x.keepPages(pr, s, l, end) catch |e| {
                std.log.warn("kolibri1: prompt pages not kept: {s}", .{@errorName(e)});
            };
            if (done) pr.report(p.len, s.cached, l.kept);
            return done;
        }
        if (s.reuse.hook) |hk| if (std.mem.indexOfScalar(u32, s.reuse.marks, @intCast(end)) != null) hk.at(hk.ptr, s, @intCast(end));
        return done;
    }

    fn prefill(ptr: *anyopaque, s: *Stream) anyerror!void {
        const x = self(ptr);
        const l = try x.opening(s);
        errdefer x.dropLane(s);
        while (!try x.chunk(s, l, 0)) {}
    }

    fn prefillStep(ptr: *anyopaque, s: *Stream, _: u32) anyerror!bool {
        const x = self(ptr);
        const going = if (x.lanes.getPtr(s)) |l| l.fill != null else false;
        const l = if (going) x.lanes.getPtr(s).? else try x.opening(s);
        errdefer x.dropLane(s);
        return x.chunk(s, l, 0);
    }

    fn cancelPrefill(ptr: *anyopaque, s: *Stream) !void {
        const l = self(ptr).lanes.getPtr(s) orelse return;
        if (l.fill) |at| s.keepInterruptedPrefill(@intCast(at));
    }

    fn handle(x: *Kolibri) usize {
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

    fn value(x: *Kolibri, feed: be.Feed) !u32 {
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
        const logits = try x.step(l, t);
        l.len += 1;
        const h = x.next;
        try x.drawRows(l, logits, 0, 1, position, x.handle(), 0);
        return h;
    }

    /// One row at the lane's cache length: the slot's captured graph (captured at its first step), else eager.
    fn step(x: *Kolibri, l: *Lane, token: u32) !u64 {
        try x.reach(l, l.len + 1);
        if (!x.graphed) {
            try x.m.forward(x.ops(), &.{.{ .slot = l.slot, .p0 = l.len, .tokens = &.{token} }}, false, null, x.logits.ptr);
            return x.logits.ptr;
        }
        const g = &x.steps[l.slot];
        if (g.* == null) g.* = try x.m.captureStep(x.ops(), l.slot);
        try x.m.step(x.ops(), &g.*.?, token, l.len);
        return g.*.?.logits;
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
            try x.reach(l, l.base + l.rows.items.len);
            c.* = .{ .slot = l.slot, .p0 = l.base, .tokens = l.rows.items };
            total += l.rows.items.len;
        }
        if (total > round_rows) return error.RoundTooWide;
        var logits = x.logits.ptr;
        if (total == 1) {
            logits = try x.step(x.lanes.getPtr(windows[0].stream).?, chains[0].tokens[0]); // a lone row: the step
        } else {
            const rows = try x.gpa.alloc(usize, total);
            defer x.gpa.free(rows);
            for (rows, 0..) |*r, i| r.* = i;
            try x.m.forward(x.ops(), chains, false, rows, logits);
        }
        var at: usize = 0;
        for (windows, 0..) |w, i| {
            const l = x.lanes.getPtr(w.stream).?;
            try x.drawRows(l, logits, at, l.rows.items.len, w.positions[0], ring + at, i);
            at += l.rows.items.len;
        }
        const got = try x.gpa.alloc(u32, total);
        defer x.gpa.free(got);
        try x.s.synchronize();
        try x.tok.download(ring * 4, std.mem.sliceAsBytes(got));
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

    /// No draft head is loaded: absorbing is a no-op, and a request for drafts is refused by name.
    fn draft(_: *anyopaque, requests: []const be.DraftRequest) anyerror!void {
        for (requests) |r| if (r.depth != 0 or r.lanes != null) return error.NoDraftHead;
    }

    fn dropLane(x: *Kolibri, s: *const Stream) void {
        if (x.lanes.fetchRemove(s)) |entry| {
            var l = entry.value;
            if (x.pool) |*p| { // one CUDA stream: a later lease's writes queue behind this stream's last reads
                p.finish(&l.lease);
                x.m.unmapPages(l.slot);
            }
            x.free[l.slot] = true;
            l.rows.deinit(x.gpa);
            x.gpa.free(l.marks);
            if (x.lanes.count() == 0) x.report();
        }
    }

    /// The idle server's line: nothing promised, every page free (kept states are copies, not pages).
    fn report(x: *const Kolibri) void {
        if (@import("builtin").is_test) return;
        const p = x.pool orelse return;
        const c = p.census();
        const ok = p.idle() and (x.prompts != null or c.free == c.total); // TF_KV_SHARE: the store holds pages
        std.log.info("kv pool idle{s}: promised {d}, free {d} + store {d} == total {d}", .{ if (ok) "" else " (UNEVEN)", c.promised, c.free, c.kept, c.total });
    }

    fn release(ptr: *anyopaque, s: *Stream) void {
        self(ptr).dropLane(s);
    }
};

test "a request holds its prompt, its reply and the verify reserve in whole pages of 512 positions" {
    try std.testing.expectEqual(@as(usize, 1), pagesFor(1, 1));
    try std.testing.expectEqual(@as(usize, 1), pagesFor(400, 96));
    try std.testing.expectEqual(@as(usize, 2), pagesFor(400, 97));
    try std.testing.expectEqual(@as(usize, 65), pagesFor(32768, 0));
}
