//! Stream slots on one Mac: a stream's caches each; prompt chunks, shared windows, keeps and drafts run on them.
const std = @import("std");
const mtl = @import("metal");
const st = @import("state.zig");
const fwd = @import("forward.zig");
const mtp = @import("mtp.zig");
const prompt_mod = @import("prompt.zig");
const Engine = @import("engine.zig").Engine;
const Ref = @import("weights.zig").Ref;

/// One stream's rows in a shared window: its pending token, then the drafts the slot holds, then host drafts.
pub const Win = struct { slot: u32, pending: u32, held: u32, tokens: []const u32 };

/// One stream's head: it absorbs the prompt's last row or the last window's first rows (a next token each), drafts.
pub const Draft = struct { slot: u32, prompt: bool, follow: []const u32, depth: u32 };

pub const Slot = struct {
    s: st.State,
    held: Ref, // u32 [max_rows]: the head's drafts for the next window
    last: Ref, // bf16 [hidden]: the prompt's last row, the head's first input
    used: bool = false,
    mtp: bool = false, // the head follows this stream (it drafts)
    held_n: u32 = 0,
    prompt_len: u32 = 0,
    row0: u32 = 0, // the last window's first row
    rows: u32 = 0, // the last window's rows until a keep settles them (0: settled)
    width: u32 = 0, // the last window's rows
    seen: u64 = 0, // the window that last ran this slot's rows
};

pub const Slots = struct {
    e: *Engine,
    slots: []Slot,
    rows: Ref, // bf16 [max_rows, hidden]: the rows every stream's head absorbs, gathered
    ids: Ref, // u32 [max_rows]: their next tokens
    lasts: Ref, // bf16 [max_rows, hidden]: each drafting stream's last absorbed row (m_x), gathered
    picks: Ref, // u32 [max_rows]: a draft level's picks, a stream each
    open: ?Open = null, // keeps and drafts encode here; the next window commits them with its forward
    windows: u64 = 0,
    log: bool = false, // GLM_WINDOWS=1: each window's streams, rows, GPU time and the GPU's idle gap before it
    last_end: f64 = 0,
    digest: u64 = 0, // the last window's picks hashed: a pair's ranks compare theirs

    const Open = struct { cb: mtl.CommandBuffer, enc: mtl.ComputeEncoder, pool: mtl.objc.Pool };

    /// `n` slots: the engine's caches first (no generate while they serve), the rest sharing its projections.
    pub fn init(gpa: std.mem.Allocator, e: *Engine, n: u32) !Slots {
        const slots = try gpa.alloc(Slot, @max(n, 1));
        errdefer gpa.free(slots);
        for (slots, 0..) |*slot, i| slot.* = .{
            .s = if (i == 0) e.s else try st.initState(&e.arena, &e.c, e.s.cap, &e.s),
            .held = try e.arena.buffer(st.max_rows * 4),
            .last = try e.arena.buffer(@as(usize, e.c.hidden) * 2),
        };
        const plane = @as(usize, st.max_rows) * e.c.hidden * 2;
        return .{ .e = e, .slots = slots, .log = std.c.getenv("GLM_WINDOWS") != null, .rows = try e.arena.buffer(plane), .ids = try e.arena.buffer(st.max_rows * 4), .lasts = try e.arena.buffer(plane), .picks = try e.arena.buffer(st.max_rows * 4) };
    }

    pub fn deinit(sl: *Slots, gpa: std.mem.Allocator) void {
        gpa.free(sl.slots);
    }

    /// A free slot, or null when every slot holds a stream.
    pub fn free(sl: *const Slots) ?u32 {
        for (sl.slots, 0..) |s, i| if (!s.used) return @intCast(i);
        return null;
    }

    fn ctx(sl: *Slots, slot: *Slot) fwd.Ctx {
        var x = sl.e.ctx();
        x.s = &slot.s;
        return x;
    }

    /// The open command buffer's encoder; its autorelease pool stays pushed until `flush` (one thread drives them).
    fn encoder(sl: *Slots) mtl.ComputeEncoder {
        if (sl.open == null) {
            const pool = mtl.objc.Pool.push();
            const b = sl.e.begin();
            sl.open = .{ .cb = b.cb, .enc = b.enc, .pool = pool };
        }
        return sl.open.?.enc;
    }

    /// Commit the open command buffer and wait for it.
    pub fn flush(sl: *Slots) !void {
        const o = sl.open orelse return;
        sl.open = null;
        defer o.pool.pop();
        try sl.e.finish(o.cb, o.enc);
    }

    /// Slot `i`'s cache length once its last window settles (a window no keep has settled keeps every row).
    pub fn length(sl: *const Slots, i: u32) u32 {
        const slot = &sl.slots[i];
        return slot.s.pos + slot.rows;
    }

    fn slotAt(sl: *Slots, i: u32) !*Slot {
        if (i >= sl.slots.len or !sl.slots[i].used) return error.SlotOutOfStep;
        return &sl.slots[i];
    }

    /// A new stream in slot `i`: its prompt into the engine's prompt ids, its caches cleared.
    pub fn begin(sl: *Slots, i: u32, prompt: []const u32, drafts: bool) !void {
        if (i >= sl.slots.len or sl.slots[i].used) return error.SlotOutOfStep;
        if (prompt.len == 0 or prompt.len + st.max_rows + 1 > sl.e.s.cap) return error.ContextFull;
        sl.settleAll(); // the prompt pass overwrites the projections a window's keep replays
        try sl.flush();
        sl.e.sync();
        const slot = &sl.slots[i];
        slot.s.reset();
        slot.used = true;
        slot.mtp = drafts and sl.e.hasMtp();
        slot.held_n = 0;
        slot.rows = 0;
        slot.prompt_len = @intCast(prompt.len);
        @memcpy(Engine.u32s(sl.e.prompt_ids, prompt.len), prompt);
    }

    /// The rows the prompt chunk at `at` takes: a tensor-unit chunk, or a window of up to 16 rows (generate's split).
    pub fn chunkRows(sl: *const Slots, i: u32, at: u32) u32 {
        const left = sl.slots[i].prompt_len - at;
        const chunked = sl.e.pr != null and left > st.max_rows;
        return @min(if (chunked) sl.e.chunk_rows else st.max_rows, left);
    }

    /// Prompt rows [at, at + n) of slot `i`, the head absorbing each with its next token; the last pick to `picks`.
    pub fn chunk(sl: *Slots, i: u32, at: u32, n: u32) !void {
        const e = sl.e;
        const slot = try sl.slotAt(i);
        const P = slot.prompt_len;
        if (at >= P or n != sl.chunkRows(i, at)) return error.ChunkOutOfStep;
        const D = e.c.hidden;
        const last = at + n == P;
        const absorb = if (last) n - 1 else n;
        try sl.flush();
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        var x = sl.ctx(slot);
        const b = e.begin();
        const ids = e.prompt_ids.at(@as(usize, at) * 4);
        const next = e.prompt_ids.at(@as(usize, at + 1) * 4);
        if (e.pr != null and P - at > st.max_rows) {
            const pr = &e.pr.?;
            var px = x;
            px.sc = &pr.streams;
            prompt_mod.backbone(pr, &px, b.enc, ids, n, at);
            fwd.flipKda(&x);
            if (last) Engine.copyRow(&x, b.enc, pr.streams.hidden.at(@as(usize, n - 1) * D * 2), slot.last, D);
            if (slot.mtp and absorb > 0) prompt_mod.mtp(pr, &px, b.enc, pr.streams.hidden, next, absorb, at);
        } else {
            fwd.backbone(&x, b.enc, ids, n, at);
            fwd.flipKda(&x);
            if (last) Engine.copyRow(&x, b.enc, e.sc.hidden.at(@as(usize, n - 1) * D * 2), slot.last, D);
            if (slot.mtp and absorb > 0) mtp.absorb(&x, b.enc, e.sc.hidden, next, absorb, at);
        }
        if (slot.mtp and absorb > 0) slot.s.mtp_pos = at + absorb;
        if (last) fwd.head(&x, b.enc, slot.last, e.sc.logits, e.sc.picks, 1);
        try e.finish(b.cb, b.enc);
        if (last) slot.s.pos = P;
    }

    /// One forward over every window, each stream's rows against its own caches; each row's pick into `picks`.
    pub fn window(sl: *Slots, wins: []const Win) !void {
        const e = sl.e;
        var segs: [st.max_rows]fwd.Seg = undefined;
        if (wins.len == 0 or wins.len > segs.len) return error.WindowOutOfStep;
        sl.settleAll();
        const ids = Engine.u32s(e.sc.ids, st.max_rows);
        var x = e.ctx();
        const enc = sl.encoder();
        var total: u32 = 0;
        for (wins, segs[0..wins.len], 0..) |w, *g, wi| {
            const slot = try sl.slotAt(w.slot);
            const rows: u32 = 1 + w.held + @as(u32, @intCast(w.tokens.len));
            for (wins[0..wi]) |o| if (o.slot == w.slot) return error.WindowOutOfStep;
            if (w.held > slot.held_n or total + rows > st.max_rows or slot.s.pos + rows + 1 > slot.s.cap) return error.WindowOutOfStep;
            ids[total] = w.pending;
            @memcpy(ids[total + 1 + w.held ..][0..w.tokens.len], w.tokens);
            if (w.held > 0) copyWords(&x, enc, slot.held, e.sc.ids.at(@as(usize, total + 1) * 4), w.held);
            g.* = .{ .s = &slot.s, .row0 = total, .rows = rows, .pos = slot.s.pos };
            slot.row0 = total;
            slot.rows = rows;
            slot.width = rows;
            total += rows;
        }
        x.s = segs[0].s;
        x.segs = segs[0..wins.len];
        fwd.backbone(&x, enc, e.sc.ids, total, segs[0].pos);
        fwd.head(&x, enc, e.sc.hidden, e.sc.logits, e.sc.picks, total);
        try sl.flush();
        if (sl.log) std.log.info("glm window: streams {d} rows {d} gpu {d:.2} ms gap {d:.2} ms", .{ wins.len, total, (e.gpu[1] - e.gpu[0]) * 1e3, (e.gpu[0] - sl.last_end) * 1e3 });
        sl.last_end = e.gpu[1];
        sl.windows += 1;
        for (wins) |w| sl.slots[w.slot].seen = sl.windows;
        sl.digest = std.hash.Wyhash.hash(sl.windows, std.mem.sliceAsBytes(Engine.u32s(e.sc.picks, total)));
    }

    /// The last window's first `kept` rows of slot `i` stay in its caches.
    pub fn keep(sl: *Slots, i: u32, kept: u32) !void {
        const slot = try sl.slotAt(i);
        if (slot.rows == 0 or kept == 0 or kept > slot.rows) return error.KeepOutOfStep;
        sl.settle(slot, kept);
    }

    /// A slot's last window settled at its first `kept` rows: rejected rows replayed out of its KDA states, flipped.
    fn settle(sl: *Slots, slot: *Slot, kept: u32) void {
        var x = sl.ctx(slot);
        if (kept < slot.rows) fwd.keepKdaAt(&x, sl.encoder(), slot.row0, slot.rows, kept);
        fwd.flipKda(&x);
        slot.s.pos += kept;
        slot.rows = 0;
    }

    /// Every window no keep settled kept all its rows.
    fn settleAll(sl: *Slots) void {
        for (sl.slots) |*slot| if (slot.used and slot.rows > 0) sl.settle(slot, slot.rows);
    }

    /// Drafts every slot can take now: distinct streams with the head, at most 16 rows to absorb, each its own.
    pub fn checkDrafts(sl: *Slots, reqs: []const Draft) !void {
        var total: usize = 0;
        if (reqs.len == 0 or reqs.len > st.max_rows) return error.DraftOutOfStep;
        for (reqs, 0..) |r, k| {
            const slot = try sl.slotAt(r.slot);
            const n = r.follow.len;
            if (!slot.mtp or n == 0 or r.depth >= st.max_rows or (r.prompt and n != 1)) return error.DraftOutOfStep;
            if (!r.prompt and (slot.seen != sl.windows or n > slot.width)) return error.DraftOutOfStep; // this window's
            for (reqs[0..k]) |o| if (o.slot == r.slot) return error.DraftOutOfStep;
            total += n;
        }
        if (total > st.max_rows) return error.DraftOutOfStep;
    }

    /// Every stream's head at once: one absorb over all their rows, then a chained step a level for those drafting on.
    pub fn draftAll(sl: *Slots, reqs: []const Draft) !void {
        const e = sl.e;
        const D: usize = e.c.hidden;
        try sl.checkDrafts(reqs);
        var order: [st.max_rows]u32 = undefined; // deepest first: each level's streams are a prefix
        for (reqs, 0..) |r, k| {
            var at = k;
            while (at > 0 and reqs[order[at - 1]].depth < r.depth) : (at -= 1) order[at] = order[at - 1];
            order[at] = @intCast(k);
        }
        const ids = Engine.u32s(sl.ids, st.max_rows);
        var segs: [st.max_rows]fwd.Seg = undefined;
        var x = e.ctx();
        const enc = sl.encoder();
        var row: u32 = 0;
        for (order[0..reqs.len], segs[0..reqs.len]) |k, *g| {
            const r = reqs[k];
            const slot = &sl.slots[r.slot];
            const n: u32 = @intCast(r.follow.len);
            const h = if (r.prompt) slot.last else e.sc.hidden.at(@as(usize, slot.row0) * D * 2);
            copyWords(&x, enc, h, sl.rows.at(row * D * 2), @intCast(n * D / 2));
            @memcpy(ids[row..][0..n], r.follow);
            g.* = .{ .s = &slot.s, .row0 = row, .rows = n, .pos = slot.s.mtp_pos };
            row += n;
        }
        x.s = segs[0].s;
        x.segs = segs[0..reqs.len];
        mtp.absorb(&x, enc, sl.rows, sl.ids, row, 0); // every row's cache entries; only last rows go on to draft
        var drafting: u32 = 0;
        var lasts: [st.max_rows]fwd.Seg = undefined;
        for (order[0..reqs.len], segs[0..reqs.len], 0..) |k, g, i| {
            const slot = &sl.slots[reqs[k].slot];
            slot.s.mtp_pos += g.rows;
            slot.held_n = reqs[k].depth;
            if (reqs[k].depth == 0) continue;
            const at = (g.row0 + g.rows - 1) * D * 2;
            copyWords(&x, enc, e.sc.m_x.at(at), sl.lasts.at(i * D * 2), @intCast(D / 2));
            copyWords(&x, enc, e.sc.m_xn.at(at), sl.rows.at(i * D * 2), @intCast(D / 2));
            lasts[i] = .{ .s = &slot.s, .row0 = @intCast(i), .rows = 1, .pos = slot.s.mtp_pos - 1 };
            drafting += 1;
        }
        if (drafting == 0) return;
        x.s = lasts[0].s;
        x.segs = lasts[0..drafting];
        mtp.finish(&x, enc, sl.lasts, sl.rows, drafting, 0, true);
        const y = e.ctx();
        mtp.drafts(&y, enc, e.sc.m_x, drafting, sl.picks);
        sl.hold(enc, order[0..drafting], reqs, 0);
        var level: u32 = 1;
        while (true) : (level += 1) {
            var n: u32 = 0; // the streams drafting past this level: a prefix of `order`
            while (n < drafting and reqs[order[n]].depth > level) : (n += 1) {
                const slot = &sl.slots[reqs[order[n]].slot];
                segs[n] = .{ .s = &slot.s, .row0 = n, .rows = 1, .pos = slot.s.mtp_pos + level - 1 };
            }
            if (n == 0) break;
            x.s = segs[0].s;
            x.segs = segs[0..n];
            mtp.layer(&x, enc, e.sc.m_x, sl.picks, n, 0); // the level before's outputs, a prefix
            mtp.drafts(&y, enc, e.sc.m_x, n, sl.picks);
            sl.hold(enc, order[0..n], reqs, level);
        }
    }

    /// Level `level`'s picks (one a stream, in `order`) into each stream's held drafts.
    fn hold(sl: *Slots, enc: mtl.ComputeEncoder, order: []const u32, reqs: []const Draft, level: u32) void {
        const x = sl.e.ctx();
        for (order, 0..) |k, i| copyWords(&x, enc, sl.picks.at(i * 4), sl.slots[reqs[k].slot].held.at(level * 4), 1);
    }

    /// The stream in slot `i` left; once every slot is free, pending work commits on this thread (its pool's owner).
    pub fn release(sl: *Slots, i: u32) void {
        if (i >= sl.slots.len) return;
        const slot = &sl.slots[i];
        slot.used = false;
        slot.rows = 0;
        slot.held_n = 0;
        for (sl.slots) |s| if (s.used) return;
        sl.flush() catch |err| std.log.err("glm: the slots' last command buffer failed: {s}", .{@errorName(err)});
    }
};

/// `n` u32 words from `src` to `dst` on the GPU.
fn copyWords(x: *const fwd.Ctx, enc: mtl.ComputeEncoder, src: Ref, dst: Ref, n: u32) void {
    enc.setPipeline(x.k.copy_u32);
    enc.setBuffer(src.buf, src.off, 0);
    enc.setBuffer(dst.buf, dst.off, 1);
    enc.setValue(n, 2);
    enc.dispatchThreads(mtl.Size.of(n, 1, 1), mtl.Size.of(32, 1, 1));
}
