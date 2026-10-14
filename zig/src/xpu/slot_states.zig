//! A slot's states: allocation on first use, and the prompt cache's snapshots (save, restore, drop).
const q = @import("qwen27.zig");
const dev = @import("dev.zig");
const fwd = @import("forward.zig");
const Runner = q.Runner;
const copy = dev.copy;
const kvBytes = fwd.kvBytes;
const off = dev.off;

/// Bytes a slot's states take: GDN and convolution states, and the K/V cache of `context` positions.
pub fn slotBytes(r: *const Runner) usize {
    _ = r;
    return 48 * (2 * 48 * 128 * 128 * 4 + 2 * 3 * 10240 * 2) + 16 * 2 * kvBytes();
}

/// A slot's state after `at` prompt tokens (prompt cache): GDN and convolution states, K/V rows 0 .. at, one buffer.
pub const Snap = struct { at: u32, buf: *anyopaque, bytes: usize };

pub const REC_BYTES = 48 * 128 * 128 * 4;
pub const CONV_BYTES = 3 * 10240 * 2;
pub fn kvRow() usize {
    return if (q.kv_i8) 4 * 256 else 4 * 256 * 2;
}

/// The bytes saveSlot takes at `at` tokens.
pub fn snapBytes(r: *const Runner, at: u32) usize {
    var n: usize = 0;
    for (&r.model.layers) |*l| n += if (l.attn) 2 * @as(usize, at) * (kvRow() + if (q.kv_i8) @as(usize, 16) else 0) else REC_BYTES + CONV_BYTES;
    return n;
}

/// Copies `slot`'s state standing at `at` tokens into new device memory.
pub fn saveSlot(r: *Runner, slot: u32, at: u32) !Snap {
    try r.select(slot);
    if (r.pos != at or r.multi_count != 0) return error.SlotNotThere;
    const d = &r.d;
    const bytes = r.snapBytes(at);
    const buf = try d.alloc_(bytes);
    errdefer d.g.free(buf);
    var o: usize = 0;
    for (&r.model.layers) |*l| {
        if (l.attn) {
            for ([_]*anyopaque{ l.kc.?, l.vc.? }) |c| {
                try copy(d, off(buf, o), c, @as(usize, at) * kvRow());
                o += @as(usize, at) * kvRow();
                if (q.kv_i8) { // the scales after the cap's rows
                    try copy(d, off(buf, o), off(c, @as(usize, q.MAX_CTX) * 1024), @as(usize, at) * 16);
                    o += @as(usize, at) * 16;
                }
            }
            continue;
        }
        try copy(d, off(buf, o), l.rec.?, REC_BYTES);
        try copy(d, off(buf, o + REC_BYTES), l.conv[l.conv_cur].?, CONV_BYTES);
        o += REC_BYTES + CONV_BYTES;
    }
    return .{ .at = at, .buf = buf, .bytes = bytes };
}

/// Makes `slot` stand where `snap` was taken (its prompt pass goes on from snap.at).
pub fn restoreSlot(r: *Runner, slot: u32, snap: Snap) !void {
    try r.select(slot);
    const d = &r.d;
    const at = snap.at;
    var o: usize = 0;
    for (&r.model.layers) |*l| {
        if (l.attn) {
            for ([_]*anyopaque{ l.kc.?, l.vc.? }) |c| {
                try copy(d, c, off(snap.buf, o), @as(usize, at) * kvRow());
                o += @as(usize, at) * kvRow();
                if (q.kv_i8) {
                    try copy(d, off(c, @as(usize, q.MAX_CTX) * 1024), off(snap.buf, o), @as(usize, at) * 16);
                    o += @as(usize, at) * 16;
                }
            }
            continue;
        }
        try copy(d, l.rec.?, off(snap.buf, o), REC_BYTES);
        try copy(d, l.conv[l.conv_cur].?, off(snap.buf, o + REC_BYTES), CONV_BYTES);
        o += REC_BYTES + CONV_BYTES;
    }
    r.pos = at;
    r.block_rows = 0;
    r.tree_on = false;
    try d.sync();
}

pub fn dropSnap(r: *Runner, snap: Snap) void {
    r.d.sync() catch {};
    r.d.g.free(snap.buf);
}

/// Slots whose states are not allocated yet (they take slotBytes on first use).
pub fn unallocatedSlots(r: *const Runner, slots: u32) u32 {
    var n: u32 = 0;
    for (0..slots) |i| n += @intFromBool(for (&r.model.layers) |*l| {
        if (l.attn) break l.slot_kc[i] == null;
    } else true);
    return n;
}

/// `slot`'s states, allocated (zero) on its first use.
pub fn ensure(r: *Runner, slot: u32) !void {
    const d = &r.d;
    for (&r.model.layers) |*l| {
        if (l.attn) {
            if (l.slot_kc[slot] != null) continue;
            l.slot_kc[slot] = try d.alloc_(kvBytes());
            l.slot_vc[slot] = try d.alloc_(kvBytes());
        } else {
            if (l.slot_rec[slot] != null) continue;
            l.slot_rec[slot] = try d.alloc_(48 * 128 * 128 * 4);
            l.slot_save[slot] = try d.alloc_(48 * 128 * 128 * 4);
            try d.fill(l.slot_rec[slot].?, @as(u32, 0), 48 * 128 * 128 * 4);
            for (&l.slot_conv[slot]) |*cv| {
                cv.* = try d.alloc_(3 * 10240 * 2);
                try d.fill(cv.*.?, @as(u16, 0), 3 * 10240 * 2);
            }
            l.slot_cur[slot] = 0;
        }
    }
    try d.sync();
}
