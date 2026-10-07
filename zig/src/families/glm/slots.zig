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

pub const Slot = struct {
    s: st.State,
    held: Ref, // u32 [max_rows]: the head's drafts for the next window
    next: Ref, // u32 [max_rows]: the token after each row the head absorbs
    last: Ref, // bf16 [hidden]: the prompt's last row, the head's first input
    used: bool = false,
    mtp: bool = false, // the head follows this stream (it drafts)
    held_n: u32 = 0,
    prompt_len: u32 = 0,
    row0: u32 = 0, // the last window's first row
    rows: u32 = 0, // the last window's rows until a keep settles them (0: settled)
    seen: u64 = 0, // the window that last ran this slot's rows
};

pub const Slots = struct {
    e: *Engine,
    slots: []Slot,
    open: ?Open = null, // keeps and drafts encode here; the next window commits them with its forward
    windows: u64 = 0,
    digest: u64 = 0, // the last window's picks hashed: a pair's ranks compare theirs

    const Open = struct { cb: mtl.CommandBuffer, enc: mtl.ComputeEncoder, pool: mtl.objc.Pool };

    /// `n` slots: the engine's caches first (no generate while they serve), the rest sharing its projections.
    pub fn init(gpa: std.mem.Allocator, e: *Engine, n: u32) !Slots {
        const slots = try gpa.alloc(Slot, @max(n, 1));
        errdefer gpa.free(slots);
        for (slots, 0..) |*slot, i| slot.* = .{
            .s = if (i == 0) e.s else try st.initState(&e.arena, &e.c, e.s.cap, &e.s),
            .held = try e.arena.buffer(st.max_rows * 4),
            .next = try e.arena.buffer(st.max_rows * 4),
            .last = try e.arena.buffer(@as(usize, e.c.hidden) * 2),
        };
        return .{ .e = e, .slots = slots };
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
            if (slot.mtp and absorb > 0) mtp.run(&x, b.enc, e.sc.hidden, next, absorb, at, null);
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
            total += rows;
        }
        x.s = segs[0].s;
        x.segs = segs[0..wins.len];
        fwd.backbone(&x, enc, e.sc.ids, total, segs[0].pos);
        fwd.head(&x, enc, e.sc.hidden, e.sc.logits, e.sc.picks, total);
        try sl.flush();
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

    /// The head absorbs the prompt's last row or the last window's first rows with their next tokens, then drafts.
    pub fn draft(sl: *Slots, i: u32, prompt: bool, follow: []const u32, depth: u32) !void {
        const e = sl.e;
        const slot = try sl.slotAt(i);
        const n: u32 = @intCast(follow.len);
        if (!slot.mtp or n == 0 or n > st.max_rows or depth >= st.max_rows or (prompt and n != 1)) return error.DraftOutOfStep;
        if (!prompt and slot.seen != sl.windows) return error.DraftOutOfStep; // its rows left the hidden scratch
        const D = e.c.hidden;
        @memcpy(Engine.u32s(slot.next, n), follow);
        const h = if (prompt) slot.last else e.sc.hidden.at(@as(usize, slot.row0) * D * 2);
        var x = sl.ctx(slot);
        const enc = sl.encoder();
        mtp.run(&x, enc, h, slot.next, n, slot.s.mtp_pos, if (depth > 0) slot.held else null);
        const base = slot.s.mtp_pos + n;
        for (1..@max(depth, 1)) |j| { // depth 0 absorbs only (1..0 is no range)
            const hj = if (j == 1) e.sc.m_x.at(@as(usize, n - 1) * D * 2) else e.sc.m_x;
            mtp.chain(&x, enc, hj, slot.held.at((j - 1) * 4), base + @as(u32, @intCast(j)) - 1, slot.held.at(j * 4));
        }
        slot.s.mtp_pos = base;
        slot.held_n = depth;
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
