//! Kept prompt states: a slot's cache rows [0, at) in every layer and its last hidden row, by the same id on both Macs.
const std = @import("std");
const mtl = @import("metal");
const ly = @import("layer.zig");
const R = @import("gpu.zig").max_rows;
const Engine = @import("engine.zig").Engine;

pub const Snap = struct { id: u32, at: u32, buf: mtl.Buffer };

const LATENT = 576 * 2;
const INDEX = 128 * 2;

/// The cache bytes a token takes over every layer this Mac holds (the MTP layer's too).
fn rowBytes(e: *const Engine) u64 {
    var n: u64 = 0;
    for (e.m.layers) |*l| n += LATENT + if (l.index_cache != null) @as(u64, INDEX) else 0;
    if (e.m.mtp) |*l| n += LATENT + if (l.index_cache != null) @as(u64, INDEX) else 0;
    return n;
}

pub fn bytes(e: *const Engine, at: u32) u64 {
    return @as(u64, at) * rowBytes(e) + e.m.c.hidden * 4;
}

/// Each layer's cache rows of slot i, [0, at), between the slot and `buf` (`out`: into buf), then the hidden row.
fn copy(e: *Engine, i: u32, at: u32, buf: mtl.Buffer, out: bool) !void {
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const cb = e.queue.commandBuffer();
    const b = cb.blit();
    var off: usize = 0;
    const ctx: usize = e.context;
    const one = struct {
        fn f(bl: mtl.BlitEncoder, cache: mtl.Buffer, row: usize, slot: usize, n: usize, ctx_: usize, buf_: mtl.Buffer, at_: *usize, out_: bool) void {
            const live = slot * ctx_ * row;
            if (out_) bl.copy(cache, live, buf_, at_.*, n * row) else bl.copy(buf_, at_.*, cache, live, n * row);
            at_.* += n * row;
        }
    }.f;
    for (e.m.layers) |*l| {
        one(b, l.cache, LATENT, i, at, ctx, buf, &off, out);
        if (l.index_cache) |c| one(b, c, INDEX, i, at, ctx, buf, &off, out);
    }
    if (e.m.mtp) |*l| {
        one(b, l.cache, LATENT, i, at, ctx, buf, &off, out);
        if (l.index_cache) |c| one(b, c, INDEX, i, at, ctx, buf, &off, out);
    }
    const hid: usize = e.m.c.hidden * 4;
    const tail = (2 * R + @as(usize, i)) * hid;
    if (out) b.copy(e.xs, tail, buf, off, hid) else b.copy(buf, off, e.xs, tail, hid);
    b.end();
    cb.commit();
    cb.wait();
    if (cb.failure()) |msg| {
        std.log.err("glm_dsa: a prompt state copy failed: {s}", .{msg});
        return error.GpuFailure;
    }
}

/// Slot i's state after its first `at` prompt tokens, kept as `id`.
pub fn save(e: *Engine, id: u32, i: u32, at: u32) !*Snap {
    if (i >= e.slots.len or !e.slots[i].used or e.slots[i].len != at or at == 0 or e.snaps.contains(id)) return error.SnapshotOutOfStep;
    const s = try e.gpa.create(Snap);
    errdefer e.gpa.destroy(s);
    s.* = .{ .id = id, .at = at, .buf = try e.device.buffer(bytes(e, at), mtl.ResourceOptions.private) };
    errdefer s.buf.deinit();
    try copy(e, i, at, s.buf, true);
    try e.snaps.put(e.gpa, id, s);
    if (e.resident) |*r| r.add(&.{s.buf});
    return s;
}

/// Kept state `id` into slot i, begun with a prompt that starts with its tokens: the next chunk starts at its end.
pub fn restore(e: *Engine, i: u32, id: u32) !void {
    const s = e.snaps.get(id) orelse return error.SnapshotOutOfStep;
    if (i >= e.slots.len or !e.slots[i].used or e.slots[i].len != 0 or s.at >= e.slots[i].prompt.items.len) return error.SnapshotOutOfStep;
    try copy(e, i, s.at, s.buf, false);
    e.slots[i].len = s.at;
}

pub fn drop(e: *Engine, id: u32) void {
    const kv = e.snaps.fetchRemove(id) orelse return;
    if (e.resident) |*r| r.remove(kv.value.buf);
    kv.value.buf.deinit();
    e.gpa.destroy(kv.value);
}

pub fn deinit(e: *Engine) void {
    var it = e.snaps.valueIterator();
    while (it.next()) |s| {
        if (e.resident) |*r| r.remove(s.*.buf);
        s.*.buf.deinit();
        e.gpa.destroy(s.*);
    }
    e.snaps.deinit(e.gpa);
}
