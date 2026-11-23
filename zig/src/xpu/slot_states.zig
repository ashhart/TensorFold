//! A slot's states across TP={1,2,4}: allocation on first use, and the prompt cache's snapshots (save, restore, drop).
const q = @import("qwen27.zig");
const dev = @import("dev.zig");
const fwd = @import("forward.zig");
const Runner = q.Runner;
const copy = dev.copy;
const kvBytes = fwd.kvBytes;
const off = dev.off;

/// Per-card bytes a slot's states take: GDN and convolution states, and the K/V cache of `context` positions.
pub fn slotBytes(_: *const Runner) usize {
    const sh = q.tpShape();
    const rec_b = @as(usize, sh.gv) * 128 * 128 * 4;
    const conv_b = 3 * @as(usize, sh.nconv) * 2;
    return 48 * (2 * rec_b + 2 * conv_b) + 16 * 2 * kvBytes();
}

/// A slot's state after `at` prompt tokens (prompt cache): one device buffer per TP rank.
pub const Snap = struct { at: u32, bufs: [4]?*anyopaque = @splat(null), bytes: usize };

pub fn recBytes() usize {
    return @as(usize, q.tpShape().gv) * 128 * 128 * 4;
}

pub fn convBytes() usize {
    return 3 * @as(usize, q.tpShape().nconv) * 2;
}

pub fn kvRow() usize {
    const akv: usize = q.tpShape().akv;
    return if (q.kv_i8) akv * 256 else akv * 256 * 2;
}

pub fn kvScalesRow() usize {
    return @as(usize, q.tpShape().akv) * 4;
}

/// Per-rank bytes saveSlot takes at `at` tokens.
pub fn snapBytes(r: *const Runner, at: u32) usize {
    var n: usize = 0;
    for (&r.model.layers) |*l| n += if (l.attn) 2 * @as(usize, at) * (kvRow() + if (q.kv_i8) kvScalesRow() else 0) else recBytes() + convBytes();
    return n;
}

/// Copies `slot`'s state standing at `at` tokens into new device memory on each TP rank.
pub fn saveSlot(r: *Runner, slot: u32, at: u32) !Snap {
    try r.select(slot);
    if (r.pos != at or r.multi_count != 0) return error.SlotNotThere;
    const bytes = r.snapBytes(at);
    const rb = recBytes();
    const cb = convBytes();
    const kvr = kvRow();
    const kvs = kvScalesRow();
    var snap: Snap = .{ .at = at, .bytes = bytes };
    errdefer r.dropSnap(snap);

    for (0..r.tp) |rk| {
        const d = r.devAt(rk);
        const m = r.modelAt(rk);
        const buf = try d.alloc_(bytes);
        snap.bufs[rk] = buf;
        var o: usize = 0;
        for (&m.layers) |*l| {
            if (l.attn) {
                for ([_]*anyopaque{ l.kc.?, l.vc.? }) |c| {
                    try copy(d, off(buf, o), c, @as(usize, at) * kvr);
                    o += @as(usize, at) * kvr;
                    if (q.kv_i8) {
                        try copy(d, off(buf, o), off(c, @as(usize, q.MAX_CTX) * kvr), @as(usize, at) * kvs);
                        o += @as(usize, at) * kvs;
                    }
                }
                continue;
            }
            try copy(d, off(buf, o), l.rec.?, rb);
            try copy(d, off(buf, o + rb), l.conv[l.conv_cur].?, cb);
            o += rb + cb;
        }
    }
    return snap;
}

/// Makes `slot` stand where `snap` was taken (its prompt pass goes on from snap.at).
pub fn restoreSlot(r: *Runner, slot: u32, snap: Snap) !void {
    try r.select(slot);
    const at = snap.at;
    const rb = recBytes();
    const cb = convBytes();
    const kvr = kvRow();
    const kvs = kvScalesRow();
    for (0..r.tp) |rk| {
        const d = r.devAt(rk);
        const m = r.modelAt(rk);
        const buf = snap.bufs[rk].?;
        var o: usize = 0;
        for (&m.layers) |*l| {
            if (l.attn) {
                for ([_]*anyopaque{ l.kc.?, l.vc.? }) |c| {
                    try copy(d, c, off(buf, o), @as(usize, at) * kvr);
                    o += @as(usize, at) * kvr;
                    if (q.kv_i8) {
                        try copy(d, off(c, @as(usize, q.MAX_CTX) * kvr), off(buf, o), @as(usize, at) * kvs);
                        o += @as(usize, at) * kvs;
                    }
                }
                continue;
            }
            try copy(d, l.rec.?, off(buf, o), rb);
            try copy(d, l.conv[l.conv_cur].?, off(buf, o + rb), cb);
            o += rb + cb;
        }
    }
    r.pos = at;
    r.block_rows = 0;
    r.tree_on = false;
    try r.syncAll();
}

pub fn dropSnap(r: *Runner, snap: Snap) void {
    for (0..r.tp) |rk| if (snap.bufs[rk]) |p| freeAfterQueue(r.devAt(rk), p);
}

/// Frees `p` once the queue has drained; kept (until the device closes) when the queue cannot be waited for.
pub fn freeAfterQueue(d: *dev.Dev, p: *anyopaque) void {
    d.sync() catch return;
    d.g.free(p);
}

/// Slots whose states are not allocated yet (they take slotBytes on first use).
pub fn unallocatedSlots(r: *const Runner, slots: u32) u32 {
    var n: u32 = 0;
    for (0..slots) |i| n += @intFromBool(for (&r.model.layers) |*l| {
        if (l.attn) break l.slot_kc[i] == null;
    } else true);
    return n;
}

/// `slot`'s states, allocated (zero) on its first use across all TP ranks.
pub fn ensure(r: *Runner, slot: u32) !void {
    const rb = recBytes();
    const cb = convBytes();
    for (0..r.tp) |rk| {
        const d = r.devAt(rk);
        const m = r.modelAt(rk);
        var had: [q.LAYERS]bool = undefined;
        for (&m.layers, &had) |*l, *h| h.* = if (l.attn) l.slot_kc[slot] != null else l.slot_rec[slot] != null;
        errdefer {
            for (&m.layers, had) |*l, h| {
                if (h) continue;
                const c = &l.slot_conv[slot];
                for ([_]*?*anyopaque{ &l.slot_kc[slot], &l.slot_vc[slot], &l.slot_rec[slot], &l.slot_save[slot], &c[0], &c[1] }) |pp| if (pp.*) |p| {
                    freeAfterQueue(d, p);
                    pp.* = null;
                };
            }
        }
        for (&m.layers) |*l| {
            if (l.attn) {
                if (l.slot_kc[slot] != null) continue;
                l.slot_kc[slot] = try d.alloc_(kvBytes());
                l.slot_vc[slot] = try d.alloc_(kvBytes());
            } else {
                if (l.slot_rec[slot] != null) continue;
                l.slot_rec[slot] = try d.alloc_(rb);
                l.slot_save[slot] = try d.alloc_(rb);
                try d.fill(l.slot_rec[slot].?, @as(u32, 0), rb);
                for (&l.slot_conv[slot]) |*cv| {
                    cv.* = try d.alloc_(cb);
                    try d.fill(cv.*.?, @as(u16, 0), cb);
                }
                l.slot_cur[slot] = 0;
            }
        }
        try d.sync();
    }
}
