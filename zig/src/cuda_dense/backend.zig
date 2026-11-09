//! A dense decoder behind lanes.Backend: a slot and pool pages a stream, a chunk a prompt step, one forward a round.
const std = @import("std");
const cuda = @import("cuda");
const lanes = @import("lanes");
const kv = @import("core").kv_pool;
const pass = @import("pass.zig");
const mdl = @import("model.zig");
const kern = @import("kernels.zig");
const pages = @import("pages.zig");
const draws_ = @import("draws.zig");

const Draws = draws_.Draws;
const ring = draws_.ring;
const be = lanes.backend;
const Stream = lanes.Stream;

/// Prompt rows a step takes: the forward's chunk grid, from position 0.
pub const prompt_rows = mdl.prompt_rows;
/// Window rows a round takes over every stream.
pub const round_rows = mdl.round_rows;
/// The widest window a stream verifies: one-row steps and every wider window give a row the same bits.
pub const max_window = pages.window;

const Lane = struct {
    slot: usize,
    len: u64 = 0, // the cache length: positions [0, len) hold this stream's keys
    base: u64 = 0, // the cache length the last verify started from
    rows: std.ArrayList(u32) = .empty, // the last verify's row tokens
    last: u64, // the prompt's last row's bf16 logits [V] (the first draw reads them)
    rule: u64, // sample.cu's Rule, when the stream samples
    sampled: bool = false,
    fill: ?usize = null, // a prompt pass going a chunk at a time: where its next chunk starts
    lease: kv.Lease = .{}, // the pages it holds and is promised
};

pub fn Dense(comptime R: type) type {
    return struct {
        const D = @This();
        gpa: std.mem.Allocator,
        d: *const cuda.Driver,
        e: pass.Engine(R),
        m: *mdl.Model,
        lanes: std.AutoHashMapUnmanaged(*const Stream, Lane) = .empty,
        free: []bool, // slot i is free
        draws: Draws,
        slot_bufs: []cuda.DeviceBuffer, // a slot's last-row logits, then its rule: two a slot
        pool: ?kv.Pool = null, // the pages every slot leases (attach)

        /// The device bytes `init` allocates: the draws and a round's slots' last rows and rules.
        pub fn openBytes(vocab: usize) usize {
            return Draws.openBytes() + round_rows * (vocab * 2 + @sizeOf(cuda.sample.Rule));
        }

        pub fn init(gpa: std.mem.Allocator, d: *const cuda.Driver, m: *mdl.Model, k: *const kern.Kernels, own: *const R.Kernels, w: *const R.Weights, s: cuda.Stream) !D {
            const free = try gpa.alloc(bool, round_rows);
            errdefer gpa.free(free);
            @memset(free, true);
            var draws = try Draws.init(gpa, d, k, s, m.g.vocab);
            errdefer draws.deinit(gpa);
            const bufs = try gpa.alloc(cuda.DeviceBuffer, 2 * round_rows);
            errdefer gpa.free(bufs);
            var made: usize = 0;
            errdefer for (bufs[0..made]) |*b| b.free();
            for (bufs, 0..) |*b, i| {
                b.* = try cuda.DeviceBuffer.alloc(d, if (i % 2 == 0) m.g.vocab * 2 else @sizeOf(cuda.sample.Rule));
                made += 1;
            }
            return .{ .gpa = gpa, .d = d, .e = .{ .m = m, .k = k, .own = own, .w = w, .s = s }, .m = m, .free = free, .draws = draws, .slot_bufs = bufs };
        }

        pub fn deinit(x: *D) void {
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
            x.draws.deinit(x.gpa);
        }

        /// The model's arena of `n` pages for `extra + 1` slots, and the pool that leases them.
        pub fn attach(x: *D, extra: usize, n: usize) !void {
            if (x.pool != null) return error.BadAttach;
            try x.m.attach(x.e.s, extra, n);
            x.pool = try kv.Pool.init(x.gpa, n, x.m.pageBytes());
        }

        /// The pool's free pages in a seeded random order, leased in that order (a check's hook).
        pub fn shufflePages(x: *D, seed: u64) void {
            var prng = std.Random.DefaultPrng.init(seed);
            if (x.pool) |*p| prng.random().shuffle(u32, p.free[0..p.nfree]);
        }

        pub fn backend(x: *D) be.Backend {
            return .{ .ptr = x, .vtable = &vtable };
        }

        const vtable: be.Backend.VTable = .{ .prefill = prefill, .prefill_step = prefillStep, .cancel_prefill = cancelPrefill, .first = first, .queue = queue, .read = read, .verify = verify, .keep = keep, .draft = draft, .release = release, .reports_fill = true };

        /// The round loop's facts: no draft head; host copies in windows up to 16 rows, streams sharing a forward.
        pub fn facts(_: *const D) lanes.Model {
            return .{ .exact_width = max_window, .gpu_tokens = false, .mtp = false, .speculate = false, .hidden_rows = true, .batch_rows = round_rows, .max_streams = round_rows };
        }

        fn self(ptr: *anyopaque) *D {
            return @ptrCast(@alignCast(ptr));
        }

        fn forward(x: *D, chains: []const pass.Chain, prompt: bool, head0: usize, heads: usize) !void {
            try pass.forward(R, x.e, chains, prompt, head0, heads);
        }

        fn freeSlot(x: *const D) ?usize {
            return std.mem.indexOfScalar(bool, x.free[0..x.m.slots], true);
        }

        /// Whether a request could start now: a free slot and its pages; one that never fits passes (refused later).
        pub fn admits(x: *const D, prompt: usize, max_new: usize) bool {
            if (prompt + max_new > x.m.context) return true;
            const p = x.pool orelse return false;
            return x.freeSlot() != null and p.fits(pages.pagesFor(prompt, max_new));
        }

        const Map = struct {
            x: *D,
            slot: usize,
            fn write(ctx: *anyopaque, ids: []const u32, from: usize) anyerror!void {
                const m: *const Map = @ptrCast(@alignCast(ctx));
                try m.x.m.mapPages(m.x.e.s, m.slot, ids[from..], from);
            }
        };

        /// The lane's pages cover positions [0, end): pages taken from its promise as its context grows.
        fn reach(x: *D, l: *Lane, end: usize) !void {
            const p = if (x.pool) |*p| p else return error.NoPages;
            const n = pages.pagesOf(end);
            if (n <= l.lease.ids.items.len) return;
            var map: Map = .{ .x = x, .slot = l.slot };
            try p.cover(&l.lease, n, .{ .ctx = &map, .write = Map.write });
        }

        /// A free slot for the request, its pages promised now (taken as its rows need them).
        fn takeSlot(x: *D, lease: *kv.Lease, prompt: usize, max_new: usize) !usize {
            const p = if (x.pool) |*p| p else return error.NoPages;
            const slot = x.freeSlot() orelse return error.NoFreeSlot;
            try p.promise(lease, pages.pagesFor(prompt, max_new));
            return slot;
        }

        /// A prompt pass begins: a slot, its pages, the stream's rule; no prompt cache, so its first chunk starts at 0.
        fn opening(x: *D, s: *Stream) !*Lane {
            const p = s.prompt();
            if (p.len == 0) return error.EmptyPrompt;
            if (s.images.len > 0) return error.DenseTextOnly;
            if (p.len + @as(usize, s.max_new) > x.m.context) return error.PastContext;
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
            try x.reach(l, p.len);
            l.sampled = false;
            if (s.sampling) |sp| if (sp.temperature > 0) {
                const r = cuda.sample.Rule.from(lanes.gpu_rule.rule(sp));
                try mdl.put(x.d, x.e.s, l.rule, std.mem.asBytes(&r));
                l.sampled = true;
            };
            s.cached = 0;
            if (s.reuse.saved != null) s.reuse_failed = true; // no prompt cache: every pass starts at 0
            l.fill = 0;
            s.fill_at.store(0, .monotonic);
            return l;
        }

        /// One chunk from `l.fill` to the next grid line or mark; true once the prompt's last row's logits are written.
        fn chunk(x: *D, s: *Stream, l: *Lane) !bool {
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
            try x.forward(&.{.{ .slot = l.slot, .p0 = at, .tokens = p[at..end] }}, true, end - at - 1, @intFromBool(done));
            if (done) try x.d.check(x.d.api.cuMemcpyDtoDAsync_v2(l.last, x.m.s.logits, x.m.g.vocab * 2, x.e.s.handle), "cuMemcpyDtoDAsync");
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

        fn first(ptr: *anyopaque, s: *Stream, position: u64) anyerror!u64 {
            const x = self(ptr);
            const l = x.lanes.getPtr(s) orelse return error.UnknownStream;
            if (position != l.len or l.fill != null) return error.PositionMismatch;
            const h = x.draws.handle();
            try x.draws.rows(l.sampled, l.rule, l.last, 0, 1, position, h.at, 0);
            return h.h;
        }

        fn value(x: *D, feed: be.Feed) !u32 {
            return switch (feed) {
                .value => |v| v,
                .handle => |h| x.draws.read(h),
            };
        }

        fn queue(ptr: *anyopaque, s: *Stream, feed: be.Feed, position: u64) anyerror!u64 {
            const x = self(ptr);
            const l = x.lanes.getPtr(s) orelse return error.UnknownStream;
            const t = try x.value(feed);
            if (position != l.len + 1) return error.PositionMismatch;
            try x.reach(l, l.len + 1);
            try x.forward(&.{.{ .slot = l.slot, .p0 = l.len, .tokens = &.{t} }}, false, 0, 1);
            l.len += 1;
            const h = x.draws.handle();
            try x.draws.rows(l.sampled, l.rule, x.m.s.logits, 0, 1, position, h.at, 0);
            return h.h;
        }

        fn read(ptr: *anyopaque, h: u64) anyerror!u32 {
            return self(ptr).draws.read(h);
        }

        fn verify(ptr: *anyopaque, windows: []const be.Window, out: []be.Verified) anyerror!void {
            const x = self(ptr);
            if (windows.len > round_rows) return error.RoundTooWide;
            const chains = try x.gpa.alloc(pass.Chain, windows.len);
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
            try x.forward(chains, false, 0, total);
            var at: usize = 0;
            var sampled = false;
            for (windows) |w| {
                const l = x.lanes.getPtr(w.stream).?;
                try x.draws.pick(l.sampled, x.m.s.logits, at, l.rows.items.len, ring + at);
                sampled = sampled or l.sampled;
                at += l.rows.items.len;
            }
            if (sampled) try x.draws.finite(ring, total); // one wait a round, before any draw reads a row
            at = 0;
            for (windows, 0..) |w, i| {
                const l = x.lanes.getPtr(w.stream).?;
                if (l.sampled) try x.draws.draw(l.rule, x.m.s.logits, at, l.rows.items.len, w.positions[0], ring + at, i);
                at += l.rows.items.len;
            }
            var got: [round_rows]u32 = undefined;
            try x.draws.tokens(ring, got[0..total]);
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
                for (path, 0..) |r, j| if (r != j) return error.ChainsOnly; // a chain keeps a prefix of its rows
                l.len = l.base + path.len; // caches are positional: later rows overwrite what the rejected ones wrote
            }
        }

        /// No draft head: absorbing is a no-op, a request for drafts is refused by name (copies come from the host).
        fn draft(_: *anyopaque, requests: []const be.DraftRequest) anyerror!void {
            for (requests) |r| if (r.depth != 0 or r.lanes != null) return error.NoDraftHead;
        }

        fn dropLane(x: *D, s: *const Stream) void {
            if (x.lanes.fetchRemove(s)) |entry| {
                var l = entry.value;
                if (x.pool) |*p| p.finish(&l.lease); // one CUDA stream: a later lease's writes queue behind these reads
                x.m.unmapPages(l.slot);
                x.free[l.slot] = true;
                l.rows.deinit(x.gpa);
                if (x.lanes.count() == 0) x.report();
            }
        }

        /// The idle server's line: nothing promised, every page free.
        fn report(x: *const D) void {
            if (@import("builtin").is_test) return;
            const p = x.pool orelse return;
            const c = p.census();
            const ok = p.idle() and c.free == c.total;
            std.log.info("kv pool idle{s}: promised {d}, free {d} == total {d}", .{ if (ok) "" else " (UNEVEN)", c.promised, c.free, c.total });
        }

        fn release(ptr: *anyopaque, s: *Stream) void {
            self(ptr).dropLane(s);
        }
    };
}
