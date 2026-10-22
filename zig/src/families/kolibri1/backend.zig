//! Kolibri-1 behind lanes.Backend: a cache slot a stream, prompts a chunk a step, one forward a round of windows.
const std = @import("std");
const cuda = @import("cuda");
const lanes = @import("lanes");
const fwd = @import("forward.zig");
const kern = @import("kernels.zig");
const snap = @import("snap.zig");
const limits = @import("context.zig");

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
};

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

    /// The device bytes `init` allocates: a round's logits, the token ring, and each slot's last row and rule.
    pub fn openBytes(vocab: usize, slots: usize) usize {
        return round_rows * vocab * 6 + (ring + round_rows) * 4 + round_rows * 16 + slots * (vocab * 4 + @sizeOf(cuda.sample.Rule));
    }

    pub fn init(gpa: std.mem.Allocator, d: *const cuda.Driver, m: *fwd.Model, k: *const kern.Kernels, s: cuda.Stream) !Kolibri {
        const v = m.c.vocab;
        const free = try gpa.alloc(bool, m.slots);
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
        const bufs = try gpa.alloc(cuda.DeviceBuffer, 2 * m.slots);
        errdefer gpa.free(bufs);
        var made: usize = 0;
        errdefer for (bufs[0..made]) |*b| b.free();
        for (bufs, 0..) |*b, i| {
            b.* = try cuda.DeviceBuffer.alloc(d, if (i % 2 == 0) v * 4 else @sizeOf(cuda.sample.Rule));
            made += 1;
        }
        const steps = try gpa.alloc(?fwd.Model.Step, m.slots);
        @memset(steps, null);
        return .{ .gpa = gpa, .d = d, .m = m, .k = k, .s = s, .free = free, .logits = logits, .logits16 = logits16, .tok = tok, .meta = meta, .slot_bufs = bufs, .steps = steps, .graphed = !envOff("TENSORFOLD_KOLIBRI_GRAPHS") };
    }

    pub fn deinit(x: *Kolibri) void {
        var it = x.lanes.valueIterator();
        while (it.next()) |l| l.rows.deinit(x.gpa);
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

    const vtable: be.Backend.VTable = .{ .prefill = prefill, .prefill_step = prefillStep, .first = first, .queue = queue, .read = read, .verify = verify, .keep = keep, .draft = draft, .release = release, .reports_fill = true };

    /// The round loop's facts: no draft head; host copies verified in windows up to 16 rows, streams sharing a forward.
    pub fn facts(x: *const Kolibri) lanes.Model {
        return .{ .exact_width = max_window, .gpu_tokens = false, .mtp = false, .speculate = false, .hidden_rows = true, .batch_rows = round_rows, .max_streams = @intCast(@min(x.m.slots, round_rows)) };
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
        return snap.save(x.gpa, x.m, x.s, l.slot, at);
    }

    /// The stream's slot made to hold `k`'s state (a restore outside its prompt pass).
    pub fn restoreState(x: *Kolibri, owner: ?*anyopaque, k: *const snap.Saved) !void {
        const s: *const Stream = @ptrCast(@alignCast(owner orelse return error.NoOwner));
        const l = x.lanes.getPtr(s) orelse return error.UnknownStream;
        try snap.restore(x.m, x.s, l.slot, k);
        l.len = k.at;
    }

    pub fn dropState(x: *Kolibri, k: *snap.Saved) void {
        snap.drop(x.gpa, x.s, k);
    }

    /// The lowest free slot.
    fn freeSlot(x: *const Kolibri) ?usize {
        return std.mem.indexOfScalar(bool, x.free, true);
    }

    /// A prompt pass begins: a slot, the stream's rule, a kept state restored; where its first chunk starts.
    fn opening(x: *Kolibri, s: *Stream) !*Lane {
        const p = s.prompt();
        if (p.len == 0) return error.EmptyPrompt;
        if (!limits.fits(x.m.context, p.len, @as(usize, s.max_new))) return error.PastContext;
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

    /// One chunk from `l.fill` to the next grid line or mark; true once the prompt's last row's logits are written.
    fn chunk(x: *Kolibri, s: *Stream, l: *Lane) !bool {
        const p = s.prompt();
        const at = l.fill.?;
        if (s.isCancelled()) return error.Cancelled;
        var end = @min(p.len, (at / prompt_rows + 1) * prompt_rows);
        for (s.reuse.marks) |mk| if (mk > at and mk < end) {
            end = mk;
        };
        try x.m.forward(x.ops(), &.{.{ .slot = l.slot, .p0 = at, .tokens = p[at..end] }}, true, null, l.last);
        s.fill_at.store(@intCast(end), .monotonic);
        const done = end == p.len;
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
    fn step(x: *Kolibri, l: *const Lane, token: u32) !u64 {
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
        if (x.lanes.fetchRemove(s)) |kv| {
            var l = kv.value;
            x.free[l.slot] = true;
            l.rows.deinit(x.gpa);
        }
    }

    fn release(ptr: *anyopaque, s: *Stream) void {
        self(ptr).dropLane(s);
    }
};
