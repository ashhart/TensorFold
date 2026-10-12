//! GLM-5.3 on a pair: a stream a slot; prompt chunks, verify windows and MTP drafts, run on both Macs in step.
const std = @import("std");
const mtl = @import("metal");
const pr = @import("../../core/pair.zig");
const cfg = @import("config.zig");
const kn = @import("kernels.zig");
const gpu = @import("gpu.zig");
const md = @import("model.zig");
const snap = @import("snap.zig");
const Ref = kn.Ref;

const R = gpu.max_rows;
/// Drafts a round at most (the MTP head's chain).
pub const max_depth = 4;
const passes = 2 + max_depth;

/// A verify window: its slot and input tokens (the pending token, then the drafts).
pub const Win = struct { slot: u32, tokens: []const u32 };
/// A draft request: the kept rows (`prompt`: the prompt's last row) and the tokens after them, then a chain.
pub const Draft = struct { slot: u32, prompt: bool, follow: []const u32, depth: u32 };

pub const Slot = struct {
    used: bool = false,
    /// Cache rows written (the stream's tokens so far).
    len: u32 = 0,
    prompt: std.ArrayList(u32) = .empty,
    /// This slot's first row in the last window, its rows no keep has settled yet (all kept unless one does), and the
    /// rows the last settle kept.
    win_row: u32 = 0,
    open: u32 = 0,
    kept: u32 = 0,
    held: [max_depth]u32 = @splat(0),
    held_n: u32 = 0,
};

/// One GPU pass's host-written tables: token ids, rows, picks and gather indices.
const Pass = struct { ids: mtl.Buffer, rows: mtl.Buffer, picks: mtl.Buffer, idx: mtl.Buffer };

pub const Engine = struct {
    gpa: std.mem.Allocator,
    io: std.Io,
    device: mtl.Device,
    queue: mtl.Queue,
    pair: *pr.Pair,
    m: md.Model,
    s: gpu.Scratch,
    slots: []Slot,
    pass: [passes]Pass,
    /// Normed hidden rows for the MTP head: the last window's [0, R), a chunk's [R, 2R), each slot's last row from 2R.
    xs: mtl.Buffer,
    words: std.ArrayList(u32) = .empty,
    /// Kept prompt states by id (the same ids on both Macs).
    snaps: std.AutoHashMapUnmanaged(u32, *snap.Snap) = .empty,
    context: u32,
    load_seconds: f64 = 0,

    pub fn load(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, settings: []const u8, layers: ?u32, mtp: bool, context: u32, streams: u32, limit: u64) !*Engine {
        const e = try gpa.create(Engine);
        errdefer gpa.destroy(e);
        const t0 = std.Io.Timestamp.now(io, .awake);
        e.gpa = gpa;
        e.io = io;
        e.device = try mtl.Device.init();
        e.queue = try e.device.queue();
        e.pair = try pr.Pair.init(gpa, io, e.device, settings);
        const text = try std.Io.Dir.cwd().readFileAlloc(io, try std.fmt.allocPrint(gpa, "{s}/config.json", .{dir}), gpa, .limited(1 << 20));
        var c = try cfg.parse(gpa, text);
        try cfg.split(&c, e.pair.rank, 2);
        e.context = context;
        e.m = try md.Model.load(gpa, io, e.device, dir, c, layers orelse c.layers, mtp, context, streams, e.pair, limit);
        e.s = try gpu.Scratch.init(e.device, &e.m.c, context);
        e.slots = try gpa.alloc(Slot, streams);
        for (e.slots) |*s| s.* = .{};
        for (&e.pass) |*p| p.* = .{ .ids = try buf(e.device, R * 4), .rows = try buf(e.device, R * @sizeOf(kn.Row)), .picks = try buf(e.device, R * 4), .idx = try buf(e.device, R * 4) };
        e.xs = try buf(e.device, (2 * R + @as(usize, streams)) * e.m.c.hidden * 4);
        e.words = .empty;
        e.snaps = .empty;
        try e.pair.ready();
        e.load_seconds = @as(f64, @floatFromInt(t0.durationTo(std.Io.Timestamp.now(io, .awake)).nanoseconds)) / 1e9;
        return e;
    }

    fn buf(device: mtl.Device, n: usize) !mtl.Buffer {
        return device.buffer(n, mtl.ResourceOptions.shared);
    }

    pub fn deinit(e: *Engine) void {
        e.pair.deinit();
        snap.deinit(e);
        for (e.slots) |*s| s.prompt.deinit(e.gpa);
        e.m.deinit();
        e.s.deinit();
        for (e.pass) |p| inline for (.{ p.ids, p.rows, p.picks, p.idx }) |b| b.deinit();
        e.xs.deinit();
        e.words.deinit(e.gpa);
        e.queue.deinit();
        e.device.deinit();
        e.gpa.destroy(e);
    }

    pub fn rank(e: *const Engine) u32 {
        return e.pair.rank;
    }
    pub fn hasMtp(e: *const Engine) bool {
        return e.m.mtp != null;
    }

    fn u32s(b: mtl.Buffer, n: usize) []u32 {
        return b.slice(u32, n);
    }

    /// A scratch whose rows table is pass `i`'s.
    fn scratch(e: *const Engine, i: usize) gpu.Scratch {
        var s = e.s;
        s.rows = e.pass[i].rows;
        return s;
    }

    fn setRow(e: *Engine, i: usize, r: usize, slot: u32, pos: u32) void {
        e.pass[i].rows.slice(kn.Row, R)[r] = .{ .slot = slot, .pos = pos, .list = @intCast(r * e.m.c.i_topk) };
    }

    fn finish(e: *Engine, cb: mtl.CommandBuffer) !void {
        cb.commit();
        cb.wait();
        if (cb.failure()) |msg| {
            std.log.err("glm_dsa: GPU command failed: {s}", .{msg});
            return error.GpuFailure;
        }
        if (e.pair.gaveUp() != 0 or e.pair.failed.load(.acquire)) return error.PairLinkFailed;
    }

    /// A fresh stream in slot `i` with its prompt.
    pub fn begin(e: *Engine, i: u32, prompt: []const u32) !void {
        if (i >= e.slots.len or prompt.len == 0 or prompt.len + max_depth + R + 1 > e.context) return error.ContextFull;
        const s = &e.slots[i];
        s.used = true;
        s.len = 0;
        s.open = 0;
        s.held_n = 0;
        s.prompt.clearRetainingCapacity();
        try s.prompt.appendSlice(e.gpa, prompt);
    }

    /// Prompt rows [at, at + n) of slot i into its caches (and the head's), the first token at the end (`every`: all).
    pub fn chunk(e: *Engine, i: u32, at: u32, n: u32, every: bool) !?u32 {
        if (i >= e.slots.len) return error.ChunkOutOfStep;
        const s = &e.slots[i];
        if (!s.used or s.open > 0 or at != s.len or n == 0 or n > R or at + n > s.prompt.items.len) return error.ChunkOutOfStep;
        const hid = e.m.c.hidden;
        const last = at + n == s.prompt.items.len;
        const toks = s.prompt.items;
        @memcpy(u32s(e.pass[0].ids, n), toks[at..][0..n]);
        for (0..n) |r| e.setRow(0, r, i, at + @as(u32, @intCast(r)));
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        const cb = e.queue.commandBuffer();
        const enc = cb.compute(.serial);
        const s0 = e.scratch(0);
        try e.m.backbone(&s0, enc, n, at + n - 1, .{ .buf = e.pass[0].ids });
        if (last or every) try e.m.pick(&s0, enc, n, .{ .buf = e.pass[0].picks });
        if (e.hasMtp()) {
            // the head's rows: the one before this chunk (the slot's xs row), then all but the last
            const carry: u32 = @intFromBool(at > 0);
            const mr = carry + n - 1;
            const idx = u32s(e.pass[1].idx, R);
            for (0..mr) |r| {
                const pos: u32 = at - carry + @as(u32, @intCast(r));
                idx[r] = if (carry == 1 and r == 0) 2 * R + i else @intCast(R + r - carry);
                u32s(e.pass[1].ids, R)[r] = toks[pos + 1];
                e.setRow(1, r, i, pos);
            }
            // the chunk's hidden rows into xs [R, R + n); the slot's last row is read before it is replaced
            for (u32s(e.pass[2].idx, n), 0..) |*x, r| x.* = @intCast(r);
            kn.gather(&e.m.K.k, enc, .{ .buf = s0.x }, .{ .buf = e.pass[2].idx }, hid, hid, hid, .{ .buf = e.xs, .off = R * hid * 4 }, n);
            enc.barrier();
            if (mr > 0) {
                kn.gather(&e.m.K.k, enc, .{ .buf = e.xs }, .{ .buf = e.pass[1].idx }, hid, hid, hid, .{ .buf = e.m.head.prev }, mr);
                enc.barrier();
            }
            u32s(e.pass[3].idx, 1)[0] = R + n - 1;
            kn.gather(&e.m.K.k, enc, .{ .buf = e.xs }, .{ .buf = e.pass[3].idx }, hid, hid, hid, .{ .buf = e.xs, .off = (2 * R + @as(usize, i)) * hid * 4 }, 1);
            enc.barrier();
            if (mr > 0) {
                const s1 = e.scratch(1);
                try e.m.draft(&s1, enc, mr, at + n - 2, .{ .buf = e.pass[1].ids }, .{ .buf = e.m.head.prev });
            }
        }
        enc.end();
        try e.finish(cb);
        s.len = at + n;
        return if (last) u32s(e.pass[0].picks, n)[n - 1] else null;
    }

    /// One forward over every window's rows: each row's next token into `out` (rows in window order).
    pub fn window(e: *Engine, wins: []const Win, out: []u32) !void {
        var rows: u32 = 0;
        var top: u32 = 0;
        for (wins) |w| {
            if (w.slot >= e.slots.len) return error.WindowOutOfStep;
            const s = &e.slots[w.slot];
            if (s.open > 0) settle(s, s.open);
            if (!s.used or w.tokens.len == 0 or s.len + w.tokens.len + max_depth + 1 > e.context) return error.WindowOutOfStep;
            if (rows + w.tokens.len > R) return error.WindowOutOfStep;
            s.win_row = rows;
            s.open = @intCast(w.tokens.len);
            s.kept = 0;
            s.held_n = 0;
            for (w.tokens, 0..) |t, j| {
                u32s(e.pass[0].ids, R)[rows] = t;
                e.setRow(0, rows, w.slot, s.len + @as(u32, @intCast(j)));
                top = @max(top, s.len + @as(u32, @intCast(j)));
                rows += 1;
            }
        }
        if (out.len < rows) return error.WindowOutOfStep;
        const hid = e.m.c.hidden;
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        const cb = e.queue.commandBuffer();
        const enc = cb.compute(.serial);
        const s0 = e.scratch(0);
        try e.m.backbone(&s0, enc, rows, top, .{ .buf = e.pass[0].ids });
        try e.m.pick(&s0, enc, rows, .{ .buf = e.pass[0].picks });
        for (u32s(e.pass[2].idx, rows), 0..) |*x, r| x.* = @intCast(r);
        kn.gather(&e.m.K.k, enc, .{ .buf = s0.x }, .{ .buf = e.pass[2].idx }, hid, hid, hid, .{ .buf = e.xs }, rows);
        enc.end();
        try e.finish(cb);
        @memcpy(out[0..rows], u32s(e.pass[0].picks, rows));
    }

    fn settle(s: *Slot, kept: u32) void {
        s.len += kept;
        s.kept = kept;
        s.open = 0;
    }

    /// The cache length the lane core counts: a window no keep settled yet counts whole.
    pub fn cacheLen(e: *const Engine, i: u32) u32 {
        return e.slots[i].len + e.slots[i].open;
    }

    /// The first `kept` rows of slot i's last window stay in its caches (a draft may have settled them already).
    pub fn keep(e: *Engine, i: u32, kept: u32) !void {
        if (i >= e.slots.len) return error.KeepOutOfStep;
        const s = &e.slots[i];
        if (!s.used or kept == 0) return error.KeepOutOfStep;
        if (s.open > 0) {
            if (kept > s.open) return error.KeepOutOfStep;
            settle(s, kept);
        } else if (kept != s.kept) return error.KeepOutOfStep;
    }

    /// Each request's rows into the MTP head's caches, then its chain of drafts (held, read with `held`).
    pub fn draft(e: *Engine, reqs: []const Draft) !void {
        if (!e.hasMtp()) return error.NoMtpHead;
        if (reqs.len == 0 or reqs.len > R) return error.DraftOutOfStep;
        var rows: u32 = 0;
        var depth: u32 = 0;
        var ends: [R]u32 = undefined;
        var top: u32 = 0;
        for (reqs, 0..) |q, k| {
            if (q.slot >= e.slots.len) return error.DraftOutOfStep;
            const s = &e.slots[q.slot];
            const n: u32 = @intCast(q.follow.len);
            if (!s.used or q.depth > max_depth or n == 0 or rows + n > R) return error.DraftOutOfStep;
            if (q.prompt and (n != 1 or s.open > 0 or s.len != s.prompt.items.len)) return error.DraftOutOfStep;
            if (!q.prompt and s.open > 0) {
                if (n > s.open) return error.DraftOutOfStep;
                settle(s, n);
            } else if (!q.prompt and n != s.kept) return error.DraftOutOfStep;
            for (q.follow, 0..) |t, j| {
                const pos = s.len - n + @as(u32, @intCast(j));
                u32s(e.pass[1].idx, R)[rows] = if (q.prompt) 2 * R + q.slot else s.win_row + @as(u32, @intCast(j));
                u32s(e.pass[1].ids, R)[rows] = t;
                e.setRow(1, rows, q.slot, pos);
                top = @max(top, pos);
                rows += 1;
            }
            ends[k] = rows - 1;
            depth = @max(depth, q.depth);
            if (s.len + max_depth + 1 > e.context) return error.ContextFull;
        }
        const hid = e.m.c.hidden;
        const nq: u32 = @intCast(reqs.len);
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        const cb = e.queue.commandBuffer();
        const enc = cb.compute(.serial);
        kn.gather(&e.m.K.k, enc, .{ .buf = e.xs }, .{ .buf = e.pass[1].idx }, hid, hid, hid, .{ .buf = e.m.head.prev }, rows);
        enc.barrier();
        const s1 = e.scratch(1);
        try e.m.draft(&s1, enc, rows, top, .{ .buf = e.pass[1].ids }, .{ .buf = e.m.head.prev });
        if (depth > 0) try e.m.pick(&s1, enc, rows, .{ .buf = e.pass[1].picks });
        var d: u32 = 2;
        while (d <= depth) : (d += 1) {
            const p = &e.pass[d];
            const prev = &e.pass[d - 1];
            for (reqs, 0..) |q, k| {
                u32s(p.idx, R)[k] = if (d == 2) ends[k] else @intCast(k);
                e.setRow(d, k, q.slot, e.slots[q.slot].len - 1 + d - 1);
            }
            kn.gather(&e.m.K.k, enc, .{ .buf = prev.picks }, .{ .buf = p.idx }, 1, 1, 1, .{ .buf = p.ids }, nq);
            kn.gather(&e.m.K.k, enc, .{ .buf = s1.h }, .{ .buf = p.idx }, hid, hid, hid, .{ .buf = e.m.head.prev }, nq);
            enc.barrier();
            const sd = e.scratch(d);
            try e.m.draft(&sd, enc, nq, top + d - 1, .{ .buf = p.ids }, .{ .buf = e.m.head.prev });
            try e.m.pick(&sd, enc, nq, .{ .buf = p.picks });
        }
        enc.end();
        try e.finish(cb);
        for (reqs, 0..) |q, k| {
            const s = &e.slots[q.slot];
            s.held_n = q.depth;
            for (0..q.depth) |j| s.held[j] = if (j == 0) u32s(e.pass[1].picks, R)[ends[k]] else u32s(e.pass[j + 1].picks, R)[k];
        }
    }

    /// The last chunk's or window's picks.
    pub fn picks(e: *const Engine, n: usize) []const u32 {
        return u32s(e.pass[0].picks, n);
    }

    pub fn held(e: *const Engine, i: u32) []const u32 {
        const s = &e.slots[i];
        return s.held[0..s.held_n];
    }

    pub fn release(e: *Engine, i: u32) void {
        if (i < e.slots.len) e.slots[i].used = false;
    }
};
