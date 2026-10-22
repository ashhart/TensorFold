//! One admitted stream's caches and metadata beside a shared Flash Next runner and scratch plane.
const std = @import("std");
const mtl = @import("metal");
const fz = @import("replay.zig");
const Engine = @import("engine.zig").Engine;
const Buf = fz.Buf;

pub const Session = struct {
    follow: ?mtl.ResidencySet.Follow = null,
    m: fz.Model,
    owned: std.ArrayList(mtl.Buffer) = .empty,
    select: [fz.CATCH]?fz.Select = @splat(null),
    draft_select: [fz.MAXR]?fz.Select = @splat(null),
    draft_ids: Buf = undefined,
    shape: mtl.Buffer = undefined,
    blocks: Buf = undefined, // an image stream's indexer block positions [cap / 4 + 1, 3] (int32), the standard frame's (m.bpos)
    last: Buf = undefined,
    held: [fz.MAXR]u32 = @splat(0),
    held_n: usize = 0,
    first: u32 = 0,
    used: bool = false,
    tokens: [fz.MAXR]u32 = @splat(0),
    row0: usize = 0,
    round: @import("batch_meta.zig").Round = .{},

    pub fn init(gpa: std.mem.Allocator, e: *Engine, capacity: usize, original: bool) !Session {
        var s: Session = .{ .m = e.m.* };
        errdefer s.deinit(gpa);
        s.m.cap = if (original) e.r.window.capacity else capacity;
        s.m.pos = 0;
        s.m.state = 0;
        s.m.state_row = 0;
        s.m.recs = null;
        s.m.marks = null;
        if (!original) {
            for (&s.m.layers) |*l| if (l.linear) {
                l.cs[0] = try s.buffer(gpa, e, fz.CS_ROW);
                l.so[0] = try s.buffer(gpa, e, fz.SO_ROW);
            } else {
                try s.attention(gpa, e, l);
            };
            try s.attention(gpa, e, &s.m.mtp);
            s.m.ple.cin = try s.buffer(gpa, e, (fz.PLE_TAIL + fz.MAXR) * fz.WIDE * 2);
        }
        s.m.t.rows = try s.words(gpa, e, &.{1});
        s.m.t.mdims = try s.words(gpa, e, &.{ 1, 16, 0, 0, 0, 0, 0, 0 });
        s.m.t.pos8 = try s.buffer(gpa, e, fz.MAXR * 4);
        s.m.t.nk8 = try s.buffer(gpa, e, fz.MAXR * 4);
        s.m.t.kvmeta = try s.words(gpa, e, &.{ 0, @intCast(s.m.cap), 1 });
        s.m.t.ple_ids = try s.buffer(gpa, e, fz.MAXR * 16 * 4);
        s.shape = (try s.words(gpa, e, &.{ 1, 2, @intCast(s.m.cap), 256 })).b;
        s.blocks = try s.buffer(gpa, e, blockCount(s.m.cap) * 3 * 4);
        s.last = try s.buffer(gpa, e, fz.WIDE * 2);
        for (&s.select) |*sel| {
            sel.* = try s.selector(gpa, e);
        }
        s.draft_ids = try s.buffer(gpa, e, fz.MAXR * 4);
        for (&s.draft_select) |*sel| sel.* = try s.selectorMeta(gpa, e, s.select[fz.CATCH - 1]);
        // Sessions draft serially, but every encoded step needs immutable host metadata.
        for (s.m.mtp.slots[0..fz.MAXR]) |*sl| sl.* = .{
            .rows = try s.words(gpa, e, &.{1}),
            .md = try s.words(gpa, e, &.{ 1, 16, 0, 0, 0, 0, 0, 0 }),
            .md4 = try s.words(gpa, e, &.{ 4, 16, 0, 0, 0, 0, 0, 0 }),
            .ids8 = try s.buffer(gpa, e, fz.MAXR * 4),
            .pos8 = try s.buffer(gpa, e, fz.MAXR * 4),
            .nk8 = try s.buffer(gpa, e, fz.MAXR * 4),
            .kvmeta = try s.buffer(gpa, e, 3 * 4),
            .n_add = try s.buffer(gpa, e, 4),
        };
        return s;
    }

    fn blockCount(cap: usize) usize {
        return cap / 4 + 1;
    }

    /// An image stream in the standard frame: every indexer block's three positions (its first row's: the frame's below
    /// the last image's end, else the slot plus `v.delta`) into `blocks`; the rounds then pool and rotate by them.
    pub fn imageBlocks(s: *Session, v: *const fz.vis_prompt.VisPrompt) void {
        v.blockPositions(s.blocks.b.slice(i32, blockCount(s.m.cap) * 3), 0, blockCount(s.m.cap));
        s.m.bpos = s.blocks;
        s.m.delta = v.delta;
    }

    pub fn buffer(s: *Session, gpa: std.mem.Allocator, e: *Engine, size: usize) !Buf {
        const b = try e.r.allocate(@max(size, 64));
        errdefer b.deinit();
        try s.owned.append(gpa, b);
        if (s.follow) |f| f.join(f.ctx, b);
        @memset(b.contents()[0..@max(size, 64)], 0);
        return .{ .b = b };
    }

    fn words(s: *Session, gpa: std.mem.Allocator, e: *Engine, values: []const i32) !Buf {
        const b = try s.buffer(gpa, e, values.len * 4);
        @memcpy(b.b.slice(i32, values.len), values);
        return b;
    }

    fn attention(s: *Session, gpa: std.mem.Allocator, e: *Engine, l: anytype) !void {
        l.keys = try s.buffer(gpa, e, 2 * s.m.cap * 256 * 2);
        l.vals = try s.buffer(gpa, e, 2 * s.m.cap * 256 * 2);
        l.raw = try s.buffer(gpa, e, s.m.cap * 128 * 2);
        l.pooled = try s.buffer(gpa, e, (s.m.cap + 3) / 4 * 128 * 2);
        l.pooled_n = 0;
    }

    fn selector(s: *Session, gpa: std.mem.Allocator, e: *Engine) !?fz.Select {
        var sel = e.r.sel orelse return null;
        sel.starts = try s.buffer(gpa, e, fz.CATCH * 256);
        sel.sc = try s.buffer(gpa, e, fz.MAXR * ((s.m.cap + 3) / 4) * 4);
        sel.keys = try s.buffer(gpa, e, fz.MAXR * fz.KW * 4);
        return s.selectorMeta(gpa, e, sel);
    }

    // Large score/key scratch is shared by sequential GPU steps; CPU-written inputs are not.
    fn selectorMeta(s: *Session, gpa: std.mem.Allocator, e: *Engine, source: ?fz.Select) !?fz.Select {
        var sel = source orelse return null;
        sel.start = try s.buffer(gpa, e, 16);
        sel.complete = try s.buffer(gpa, e, fz.MAXR * 4);
        sel.ends = try s.buffer(gpa, e, fz.MAXR * 4);
        sel.counts = try s.buffer(gpa, e, fz.MAXR * 4);
        sel.sparse = try s.buffer(gpa, e, fz.MAXR * 4);
        sel.pooled_shape = (try s.buffer(gpa, e, 16)).b;
        sel.q_shape = (try s.buffer(gpa, e, 16)).b;
        sel.sc_shape = (try s.buffer(gpa, e, 16)).b;
        sel.ids_shape = (try s.buffer(gpa, e, 16)).b;
        return sel;
    }

    /// Absorb a kept window and generate its drafts with one GPU submission and one readback.
    pub fn draftChain(s: *Session, follow: []const u32, hidden: Buf, depth: usize) !void {
        if (follow.len == 0 or follow.len > fz.MAXR or depth == 0 or depth >= fz.MAXR) return error.DraftOutOfStep;
        const m = &s.m;
        const h = &m.mtp;
        const r = m.r;
        const old_select = r.sel;
        defer r.sel = old_select;
        h.pos -= h.drafted;
        h.drafted = 0;
        h.pooled_n = @min(h.pooled_n, h.pos / 4);
        @memcpy(h.slots[0].ids8.b.slice(u32, follow.len), follow);
        const cb = r.queue.commandBuffer();
        r.enc = cb.compute(if (r.serial) .serial else .concurrent);
        var ended = false;
        errdefer if (!ended) r.enc.end();
        for (0..depth) |i| {
            r.sel = s.draft_select[i];
            const rows = if (i == 0) follow.len else 1;
            const input = if (i == 0) h.slots[0].ids8 else s.draft_ids.at((i - 1) * 4);
            const streams = if (i == 0) hidden else h.h[1];
            try m.mtpEncode(i, rows, input, streams, s.draft_ids.at(i * 4));
            h.pos += rows;
        }
        ended = true; // finish ends the encoder even if submission fails.
        try m.finish(cb);
        h.drafted = depth - 1;
        @memcpy(s.held[0..depth], s.draft_ids.b.slice(u32, depth));
    }

    /// Copies this session's cache, recurrent state, n-gram tail and head to a free larger session on the GPU.
    pub fn moveTo(s: *Session, dst: *Session, e: *Engine) !void {
        const m = &s.m;
        const d = &dst.m;
        if (m.state != 0 or m.state_row != 0 or s.round.pending) return error.StateNotSettled;
        const rows = @min(m.cap, m.pos + 2 * fz.MAXR + 2); // the cache's rows and the head's chained ones past them
        if (rows > d.cap) return error.ContextFull;
        const cb = e.r.queue.commandBuffer();
        const blit = cb.blit();
        for (&m.layers, &d.layers) |*a, *b| if (a.linear) {
            blit.copy(a.cs[0].b, a.cs[0].off, b.cs[0].b, b.cs[0].off, fz.CS_ROW);
            blit.copy(a.so[0].b, a.so[0].off, b.so[0].b, b.so[0].off, fz.SO_ROW);
        } else {
            moveRows(blit, a, b, m.cap, d.cap, rows);
        };
        moveRows(blit, &m.mtp, &d.mtp, m.cap, d.cap, rows);
        blit.copy(m.ple.cin.b, m.ple.cin.off, d.ple.cin.b, d.ple.cin.off, (fz.PLE_TAIL + fz.MAXR) * fz.WIDE * 2);
        blit.copy(s.last.b, s.last.off, dst.last.b, dst.last.off, fz.WIDE * 2);
        blit.end();
        cb.commit();
        cb.wait();
        if (cb.failure()) |message| {
            std.log.err("session move failed: {s}", .{message});
            return error.GpuFailed;
        }
        d.pos = m.pos;
        d.state = 0;
        d.state_row = 0;
        d.ple.hist = m.ple.hist;
        d.mtp.pos = m.mtp.pos;
        d.mtp.drafted = m.mtp.drafted;
        d.delta = m.delta;
        d.bpos = null;
        if (m.bpos != null) { // the block positions it has, then the blocks past them at their slot plus delta
            const out = dst.blocks.b.slice(i32, blockCount(d.cap) * 3);
            @memcpy(out[0 .. blockCount(m.cap) * 3], s.blocks.b.slice(i32, blockCount(m.cap) * 3));
            for (blockCount(m.cap)..blockCount(d.cap)) |b| @memcpy(out[b * 3 ..][0..3], &@as([3]i32, @splat(@as(i32, @intCast(4 * b)) + m.delta)));
            d.bpos = dst.blocks;
        }
        dst.held = s.held;
        dst.held_n = s.held_n;
        dst.first = s.first;
        dst.round = s.round;
        dst.tokens = s.tokens;
        dst.row0 = s.row0;
        dst.used = true;
    }

    /// A sparse-attention layer's first `rows` keys, values and indexer keys and its pooled blocks, head by head.
    fn moveRows(blit: mtl.BlitEncoder, a: anytype, b: anytype, from: usize, to: usize, rows: usize) void {
        const row = 256 * 2;
        for (0..2) |hd| {
            blit.copy(a.keys.b, a.keys.off + hd * from * row, b.keys.b, b.keys.off + hd * to * row, rows * row);
            blit.copy(a.vals.b, a.vals.off + hd * from * row, b.vals.b, b.vals.off + hd * to * row, rows * row);
        }
        blit.copy(a.raw.b, a.raw.off, b.raw.b, b.raw.off, rows * 128 * 2);
        if (a.pooled_n > 0) blit.copy(a.pooled.b, a.pooled.off, b.pooled.b, b.pooled.off, a.pooled_n * 128 * 2);
        b.pooled_n = a.pooled_n;
    }

    /// The head's absorb of a kept window and `depth` chained drafts, encoded unsubmitted (heldDrafts reads them).
    pub fn encodeDrafts(s: *Session, follow: []const u32, hidden: Buf, depth: usize) !void {
        if (follow.len == 0 or follow.len > fz.MAXR or depth >= fz.MAXR) return error.DraftOutOfStep;
        const m = &s.m;
        const h = &m.mtp;
        const r = m.r;
        const old_select = r.sel;
        defer r.sel = old_select;
        h.pos -= h.drafted;
        h.drafted = 0;
        h.pooled_n = @min(h.pooled_n, h.pos / 4);
        @memcpy(h.slots[0].ids8.b.slice(u32, follow.len), follow);
        for (0..@max(depth, 1)) |i| {
            r.sel = s.draft_select[i];
            const rows = if (i == 0) follow.len else 1;
            const input = if (i == 0) h.slots[0].ids8 else s.draft_ids.at((i - 1) * 4);
            const streams = if (i == 0) hidden else h.h[1];
            try m.mtpEncode(i, rows, input, streams, s.draft_ids.at(i * 4));
            h.pos += rows;
        }
    }

    /// Reads completed encoded drafts and trims chained rows at the next absorb.
    pub fn heldDrafts(s: *Session, depth: usize) void {
        s.m.mtp.drafted = depth -| 1;
        @memcpy(s.held[0..depth], s.draft_ids.b.slice(u32, depth));
        s.held_n = depth;
    }

    pub fn setResident(s: *Session, hook: ?mtl.ResidencySet.Follow) void {
        for (s.owned.items) |b| {
            if (s.follow) |f| f.leave(f.ctx, b);
            if (hook) |f| f.join(f.ctx, b);
        }
        s.follow = hook;
    }

    pub fn deinit(s: *Session, gpa: std.mem.Allocator) void {
        s.setResident(null);
        for (s.owned.items) |b| b.deinit();
        s.owned.deinit(gpa);
    }

    pub fn normalize(s: *Session) void {
        const m = &s.m;
        if (m.state == 0 and m.state_row == 0) return;
        for (&m.layers) |*l| if (l.linear) {
            copy(l.cs[m.state].at(m.state_row * fz.CS_ROW), l.cs[0], fz.CS_ROW);
            copy(l.so[m.state].at(m.state_row * fz.SO_ROW), l.so[0], fz.SO_ROW);
        };
        m.state = 0;
        m.state_row = 0;
    }

    pub fn keep(s: *Session, n: usize) !void {
        try s.round.keep(n);
        s.m.keepRows(s.tokens[0..s.round.rows], n);
    }

    pub fn normalizeGpu(s: *Session, blit: mtl.BlitEncoder) void {
        const m = &s.m;
        if (m.state == 0 and m.state_row == 0) return;
        for (&m.layers) |*l| if (l.linear) {
            const cs = l.cs[m.state].at(m.state_row * fz.CS_ROW);
            const so = l.so[m.state].at(m.state_row * fz.SO_ROW);
            blit.copy(cs.b, cs.off, l.cs[0].b, l.cs[0].off, fz.CS_ROW);
            blit.copy(so.b, so.off, l.so[0].b, l.so[0].off, fz.SO_ROW);
        };
    }

    pub fn copy(src: Buf, dst: Buf, n: usize) void {
        std.mem.copyForwards(u8, dst.b.contents()[dst.off..][0..n], src.b.contents()[src.off..][0..n]);
    }

    pub fn bytes(capacity: usize) u64 {
        return 36 * (fz.CS_ROW + fz.SO_ROW) + 13 * (capacity * 2304 + (capacity + 3) / 4 * 256) +
            overhead(capacity);
    }

    /// Metadata and selectors are owned even when the original engine supplies this slot's caches.
    pub fn overhead(capacity: usize) u64 {
        return fz.CATCH * (fz.MAXR * ((capacity + 3) / 4) * 4 + fz.MAXR * fz.KW * 4 + 4096) +
            fz.MAXR * 17 * 64 + 64 + @max(blockCount(capacity) * 3 * 4, 64) + (2 << 20);
    }
};
