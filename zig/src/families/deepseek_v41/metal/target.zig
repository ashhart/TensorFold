//! The lanes' verify target on Metal: a slot's prompt (prompt.zig), its windows of rows (forward.zig), keeps and reuse.
const std = @import("std");
const mtl = @import("metal");
const host = @import("dsv41");
const kv = @import("kv.zig");
const rope = @import("rope.zig");
const ctx_ = @import("ctx.zig");
const scratch = @import("scratch.zig");
const forward = @import("forward.zig");
const prompt = @import("prompt.zig");
const pipelines = @import("pipelines.zig");
const snapshots = @import("snapshots.zig");

const iface = host.iface;
const lanes = host.lanes;
const Config = host.config.Config;
const Allocator = std.mem.Allocator;
const R = pipelines.max_rows;

/// What the target reads and borrows: the device, kernels, RoPE tables, layer scratch, weights and Engram rows.
pub const Env = struct {
    device: mtl.Device,
    queue: mtl.Queue,
    c: *const Config,
    p: *const pipelines.Pipelines,
    tables: *const rope.Tables,
    s: *const scratch.Scratch,
    model: forward.Model,
    lookup: ?forward.Lookup,
};

pub const Options = struct {
    slots: u32,
    /// positions a slot holds (the context limit)
    limit: u32,
    prompt: prompt.Opts = .{},
    /// bytes of kept prompt states (0: no prompt cache)
    cache_bytes: u64 = 0,
};

/// What the target did, for the tests and the logs.
pub const Stats = struct { prompts: u64 = 0, hits: u64 = 0, cached_rows: u64 = 0, windows: u64 = 0, rows: u64 = 0, rewinds: u64 = 0, launches: u64 = 0 };

/// A slot: its device state, prompt tail, ids (rows run, the last window's included until its keep) and taps.
const Slot = struct {
    kv: kv.Slot,
    tail: prompt.Tail,
    hist: std.ArrayList(u32) = .empty,
    /// the last window: its first position and rows; `open` until its keep
    base: u64 = 0,
    rows: u32 = 0,
    open: bool = false,
    /// the taps of the last `ring rows` positions, position p at row p % ring rows
    ring: []u16,
    /// the taps `taps` names after a prompt: its last rows in order
    ptaps: mtl.Buffer,
    ptaps_rows: u32 = 0,
    after_prompt: bool = false,
    /// the most positions any request wrote here since the slot was last cleared
    high: u64 = 0,
    live: bool = false,
    /// every launch since the request began completed (a release keeps only a whole state)
    whole: bool = false,
    /// the rows the last prompt took from kept state
    resumed: u32 = 0,
};

pub const Target = struct {
    a: Allocator,
    env: Env,
    opt: Options,
    b: forward.Bufs,
    pr: prompt.Prompt,
    slots: []Slot,
    cache: snapshots.Store,
    stats: Stats = .{},
    /// a command buffer failed: the device state is no longer trusted (the health hook)
    failed: bool = false,

    pub fn init(a: Allocator, env: Env, opt: Options) !*Target {
        const c = env.c;
        if (opt.slots == 0 or opt.limit == 0 or opt.limit > env.tables.positions) return error.BadOptions;
        _ = try prompt.check(c);
        const t = try a.create(Target);
        errdefer a.destroy(t);
        t.* = .{ .a = a, .env = env, .opt = opt, .b = undefined, .pr = undefined, .slots = &.{}, .cache = .{ .a = a, .cap = opt.cache_bytes } };
        t.b = try forward.Bufs.init(env.device, c);
        errdefer t.b.deinit();
        t.pr = try prompt.Prompt.init(env.device, c, opt.prompt, opt.limit);
        errdefer t.pr.deinit();
        const slots = try a.alloc(Slot, opt.slots);
        errdefer a.free(slots);
        var made: usize = 0;
        errdefer for (slots[0..made]) |*s| freeSlot(a, s);
        const cap = try prompt.tailRows(c, opt.prompt);
        const ptb = @max(4, @as(usize, ringRows(c)) * tapWidth(c) * 2);
        for (slots) |*s| {
            var st = try kv.Slot.init(a, env.device, c, opt.limit);
            errdefer st.deinit();
            var tail = try prompt.Tail.init(env.device, c, cap);
            errdefer tail.deinit();
            const ptaps = try env.device.buffer(ptb, mtl.ResourceOptions.shared);
            errdefer ptaps.deinit();
            const ring = try a.alloc(u16, @as(usize, ringRows(c)) * tapWidth(c));
            @memset(ring, 0);
            s.* = .{ .kv = st, .tail = tail, .ring = ring, .ptaps = ptaps };
            made += 1;
        }
        t.slots = slots;
        return t;
    }

    fn freeSlot(a: Allocator, s: *Slot) void {
        s.kv.deinit();
        s.tail.deinit();
        s.ptaps.deinit();
        a.free(s.ring);
        s.hist.deinit(a);
    }

    pub fn deinit(t: *Target) void {
        for (t.slots) |*s| freeSlot(t.a, s);
        t.a.free(t.slots);
        t.cache.deinit();
        t.pr.deinit();
        t.b.deinit();
        t.a.destroy(t);
    }

    /// The taps ring's rows: the drafter's context window.
    fn ringRows(c: *const Config) u32 {
        return c.window;
    }

    /// Values in a taps row: every target layer's hidden.
    fn tapWidth(c: *const Config) usize {
        return c.dspark_targets.len * c.hidden;
    }

    pub fn target(t: *Target) iface.Target {
        return .{ .ptr = t, .vtable = &.{ .prefill = prefill, .window = window, .keep = keep, .taps = taps, .release = release, .resumed = resumedFn } };
    }

    fn self(ptr: *anyopaque) *Target {
        return @ptrCast(@alignCast(ptr));
    }

    /// The device buffer and offset of an address a `taps` named (the pass's ingest reads it there).
    pub fn bufferOf(t: *const Target, address: u64) ?struct { buf: mtl.Buffer, off: usize } {
        for (t.slots) |s| {
            const bufs = [_]?mtl.Buffer{ s.ptaps, s.kv.taps };
            for (bufs) |ob| if (ob) |b| {
                const lo = @intFromPtr(b.contents());
                if (address >= lo and address < lo + b.length()) return .{ .buf = b, .off = @intCast(address - lo) };
            };
        }
        return null;
    }

    // prompts ------------------------------------------------------------------------------------------------------

    fn resumedFn(ptr: *anyopaque, slot: u32) u64 {
        const t = self(ptr);
        return if (slot < t.slots.len) t.slots[slot].resumed else 0;
    }

    fn prefill(ptr: *anyopaque, slot: u32, ids: []const u32, smp: ?lanes.Sampling, draw: u64) anyerror!u32 {
        const t = self(ptr);
        if (slot >= t.slots.len) return error.NoSlot;
        const s = &t.slots[slot];
        const c = t.env.c;
        if (s.live) return error.SlotLive;
        if (ids.len == 0) return error.EmptyPrompt;
        if (ids.len >= t.opt.limit) return error.PastLimit;
        for (ids) |id| if (id >= c.vocab) return error.BadToken;
        s.whole = false;
        s.open = false;
        s.rows = 0;
        t.stats.prompts += 1;
        // a kept state that starts this prompt, else a cleared slot
        const hit = if (t.cache.on()) t.cache.best(ids) else null;
        const at: u32 = if (hit) |e| @intCast(e.ids.len) else 0;
        if (hit) |e| {
            t.cache.restore(c, e, &s.kv, s.high, s.ring);
            t.stats.hits += 1;
            t.stats.cached_rows += at;
        } else snapshots.clear(c, &s.kv, s.high);
        s.resumed = at;
        s.tail.end = at;
        s.tail.done = at;
        s.hist.clearRetainingCapacity();
        try s.hist.appendSlice(t.a, ids);
        s.high = @max(at, ids.len);
        const n: u32 = @intCast(ids.len);
        const V: usize = c.vocab;
        var logits: []const f32 = undefined;
        if (at == n) logits = hit.?.logits.? else {
            t.pr.run(t.promptEnv(), &s.kv, &s.tail, ids) catch |e| return t.fail(e);
            const all = t.pr.logits(c);
            logits = all[(t.pr.head - 1) * V ..][0..V];
            const tr = t.pr.tapRows(c);
            for (0..t.pr.taps) |i| t.ringPut(s, n - t.pr.taps + i, tr[i * tapWidth(c) ..][0..tapWidth(c)]);
        }
        t.promptTaps(s, n);
        const choice = try t.choose(logits, draw, smp);
        s.base = n;
        s.live = true;
        s.whole = true;
        if (t.cache.on() and at < n) t.cache.save(c, &s.kv, ids, s.ring, logits) catch {};
        return choice;
    }

    fn promptEnv(t: *const Target) prompt.Env {
        const e = t.env;
        return .{ .c = e.c, .p = e.p, .tables = e.tables, .model = e.model, .lookup = e.lookup, .queue = e.queue };
    }

    fn ringPut(t: *const Target, s: *Slot, pos: u64, row: []const u16) void {
        const w = tapWidth(t.env.c);
        if (w == 0) return;
        const at: usize = @intCast(pos % ringRows(t.env.c));
        @memcpy(s.ring[at * w ..][0..w], row);
    }

    /// The taps of the prompt's last rows (the ring's, up to `ring rows`) in order, for `taps` after a prefill.
    fn promptTaps(t: *const Target, s: *Slot, n: u32) void {
        const c = t.env.c;
        const w = tapWidth(c);
        s.after_prompt = true;
        s.ptaps_rows = if (w == 0) 0 else @min(n, ringRows(c));
        const out = s.ptaps.slice(u16, @as(usize, s.ptaps_rows) * w);
        for (0..s.ptaps_rows) |i| {
            const pos = n - s.ptaps_rows + i;
            @memcpy(out[i * w ..][0..w], s.ring[(pos % ringRows(c)) * w ..][0..w]);
        }
    }

    /// The keyed choice of a logits row over the whole vocabulary (greedy: the first maximum, ties to the lower id).
    fn choose(t: *const Target, row: []const f32, draw: u64, smp: ?lanes.Sampling) !u32 {
        const s = smp orelse return argmax(row);
        if (s.temperature <= 0) return argmax(row);
        const vals = try t.a.alloc(f64, row.len);
        defer t.a.free(vals);
        const ids = try t.a.alloc(u64, row.len);
        defer t.a.free(ids);
        for (vals, ids, row, 0..) |*v, *id, x, i| {
            v.* = x;
            id.* = i;
        }
        return @intCast(try lanes.sampling.choose(t.a, vals, ids, draw, s));
    }

    fn argmax(row: []const f32) u32 {
        var best: usize = 0;
        for (row, 0..) |x, i| if (x > row[best]) {
            best = i;
        };
        return @intCast(best);
    }

    fn fail(t: *Target, e: anyerror) anyerror {
        if (e == error.CommandBufferFailed) t.failed = true;
        return e;
    }

    // windows ------------------------------------------------------------------------------------------------------

    /// A round's windows: slots' rows share a launch (up to R rows, a slot once), each launch one forward.
    fn window(ptr: *anyopaque, segments: []const iface.Segment, choices: [][]u32) anyerror!void {
        const t = self(ptr);
        var from: usize = 0;
        while (from < segments.len) {
            var rows: usize = 0;
            var to = from;
            while (to < segments.len) : (to += 1) {
                const g = segments[to];
                if (g.slot >= t.slots.len) return error.NoSlot;
                if (to > from and (rows + g.tokens.len > R or dup(segments[from..to], g.slot))) break;
                rows += g.tokens.len;
            }
            try t.windowGroup(segments[from..to], choices[from..to]);
            from = to;
        }
    }

    fn dup(done: []const iface.Segment, slot: u32) bool {
        for (done) |o| if (o.slot == slot) return true;
        return false;
    }

    /// One launch over the rows of these segments (distinct slots, at most R rows in all).
    fn windowGroup(t: *Target, group: []const iface.Segment, choices: [][]u32) !void {
        const c = t.env.c;
        const V: usize = c.vocab;
        var total: u32 = 0;
        for (group, choices) |g, out| {
            const s = &t.slots[g.slot];
            const n = g.tokens.len;
            if (!s.live) return error.SlotNotLive;
            if (g.start != s.hist.items.len) return error.PositionMismatch;
            if (n == 0 or total + n > R or g.draws.len != n or out.len < n) return error.BadRows;
            if (g.start + n > t.opt.limit) return error.PastLimit;
            // the decode forward runs consecutive positions: a tree's rows must be one chain
            if (g.parents) |ps| for (ps, 0..) |q, r| if (q != @as(i32, @intCast(r)) - 1) return error.TreeWindow;
            for (g.tokens) |id| if (id >= c.vocab) return error.BadToken;
            total += @intCast(n);
        }
        for (group) |g| {
            const s = &t.slots[g.slot];
            s.whole = false;
            try s.hist.appendSlice(t.a, g.tokens);
            s.high = @max(s.high, s.hist.items.len);
        }
        t.launch(group) catch |e| return t.fail(e);
        const lg = t.b.logitsOf(c, total);
        const w = tapWidth(c);
        var r0: usize = 0;
        for (group, choices) |g, out| {
            const s = &t.slots[g.slot];
            const n = g.tokens.len;
            for (0..n) |r| out[r] = try t.choose(lg[(r0 + r) * V ..][0..V], g.draws[r], g.sampling);
            if (s.kv.taps) |tb| for (0..n) |r| t.ringPut(s, g.start + r, tb.slice(u16, n * w)[r * w ..][0..w]);
            s.base = g.start;
            s.rows = @intCast(n);
            s.open = true;
            s.after_prompt = false;
            s.tail.end = @intCast(s.hist.items.len);
            s.tail.done = s.tail.end;
            s.whole = true;
            t.stats.windows += 1;
            t.stats.rows += n;
            r0 += n;
        }
        t.stats.launches += 1;
    }

    /// One launch of the decode forward over the segments' rows, each slot's ids at its positions from `start`.
    fn launch(t: *Target, group: []const iface.Segment) !void {
        const e = t.env;
        const c = e.c;
        const look = c.engram_max_ngram - 1;
        var pieces: [R]forward.Piece = undefined;
        var spans: [R]scratch.Span = undefined;
        var segs: [R]ctx_.Seg = undefined;
        var r0: u32 = 0;
        for (group, 0..) |g, i| {
            const s = &t.slots[g.slot];
            const n: u32 = @intCast(g.tokens.len);
            const start: u32 = @intCast(g.start);
            pieces[i] = .{ .ids = g.tokens, .lookback = s.hist.items[start -| look..start] };
            spans[i] = .{ .p0 = start, .n = n };
            segs[i] = .{ .slot = &s.kv, .p0 = start, .r0 = r0, .n = n };
            r0 += n;
        }
        try forward.prepareSegs(&t.b, c, pieces[0..group.len], e.lookup);
        try e.s.writeSpans(c, spans[0..group.len]);
        const x: ctx_.Ctx = .{ .c = c, .p = e.p, .s = e.s, .slot = segs[0].slot, .tables = e.tables, .p0 = segs[0].p0, .rows = r0, .segs = segs[0..group.len] };
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        try forward.run(&x, e.queue, e.model, &t.b);
    }

    /// Keep the window's first rows (a chain from row 0): later rows' ids go, the compressor's carry rewinds.
    fn keep(ptr: *anyopaque, slot: u32, path: []const u32) anyerror!void {
        const t = self(ptr);
        if (slot >= t.slots.len) return error.NoSlot;
        const s = &t.slots[slot];
        if (!s.open) return error.NoWindow;
        if (path.len == 0 or path.len > s.rows) return error.NotAPath;
        for (path, 0..) |r, i| if (r != i) return error.NotAPath;
        s.open = false;
        const m: u32 = @intCast(path.len);
        if (m == s.rows) return;
        const len = s.base + m;
        s.hist.shrinkRetainingCapacity(@intCast(len));
        for (s.kv.regions) |*r| kv.rewindCarry(r, t.env.c, s.base, m);
        s.tail.end = @intCast(len);
        s.tail.done = s.tail.end;
        t.stats.rewinds += 1;
    }

    fn taps(ptr: *anyopaque, slot: u32) iface.Taps {
        const t = self(ptr);
        const s = &t.slots[slot];
        const w: u32 = @intCast(tapWidth(t.env.c));
        if (s.after_prompt) return .{ .address = @intFromPtr(s.ptaps.contents()), .rows = s.ptaps_rows, .stride = w };
        const tb = s.kv.taps orelse return .{ .address = 0, .rows = 0, .stride = w };
        return .{ .address = @intFromPtr(tb.contents()), .rows = s.rows, .stride = w };
    }

    /// The request ended: its state is kept (the prompt cache) when every launch of it completed.
    fn release(ptr: *anyopaque, slot: u32) void {
        const t = self(ptr);
        if (slot >= t.slots.len) return;
        const s = &t.slots[slot];
        if (!s.live) return;
        s.live = false;
        s.open = false;
        if (s.whole and t.cache.on()) t.cache.save(t.env.c, &s.kv, s.hist.items, s.ring, null) catch {};
    }

    /// The serial reference without lanes: the prompt, then one row a step, each drawn at its own position.
    pub fn serial(t: *Target, slot: u32, ids: []const u32, n: usize, smp: ?lanes.Sampling) ![]u32 {
        var out: std.ArrayList(u32) = .empty;
        errdefer out.deinit(t.a);
        var tok = try prefill(t, slot, ids, smp, ids.len);
        while (out.items.len < n) {
            try out.append(t.a, tok);
            const at: u64 = ids.len + out.items.len - 1;
            var ch: [1]u32 = undefined;
            var cs = [_][]u32{&ch};
            try window(t, &.{.{ .slot = slot, .start = at, .tokens = &.{tok}, .parents = null, .draws = &.{at + 1}, .sampling = smp }}, &cs);
            tok = ch[0];
        }
        release(t, slot);
        return out.toOwnedSlice(t.a);
    }
};
