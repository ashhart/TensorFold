//! The 27B's prompt states for the cache: DeltaNet states, attention K/V rows and the drafter's taps.
const std = @import("std");
const mtl = @import("metal");
const Runner = @import("decode_round.zig").Runner;
const Draft = @import("dflash/runtime_model.zig").Model;

/// One kept state: its bytes in a GPU buffer, its position and the drafter's committed end there.
pub const State = struct { buf: mtl.Buffer, at: u32, bytes: usize, draft_end: u64 };

const Part = struct { buffer: mtl.Buffer, offset: usize, len: usize };
const ROW = 256 * 2; // a key or value row of one head (bf16)
pub const Ring = @import("tap_ring.zig").Ring;

fn parts(r: *const Runner, slot: u32, ring: ?Ring, at: u32, list: *std.ArrayList(Part), a: std.mem.Allocator) !void {
    if (slot >= r.ready) return error.SlotNotReady;
    for (0..r.gdn.budget.layers) |layer| for ([_]bool{ true, false }) |recurrent| {
        const span = try r.gdn.committedSpan(layer, slot, recurrent);
        try list.append(a, .{ .buffer = span.buffer, .offset = span.offset, .len = span.bytes });
    };
    for (0..r.caches.len / r.slots) |layer| {
        const cache = r.caches[layer * r.slots + slot];
        for ([_]@import("projection.zig").Ref{ cache.keys, cache.values }, [_]u32{ cache.key_stride, cache.value_stride }) |ref, stride| {
            const head_stride = @as(usize, if (stride == 0) cache.capacity * 256 else stride) * 2;
            for (0..r.model.config.kv_heads) |head| try list.append(a, .{ .buffer = ref.buffer, .offset = ref.offset + head * head_stride, .len = @as(usize, at) * ROW });
        }
    }
    if (ring) |g| try list.append(a, .{ .buffer = g.buf, .offset = 0, .len = g.bytes(at) });
}

/// Bytes of a state after `at` tokens (every slot's are the same size).
pub fn bytes(r: *const Runner, draft: ?*Draft, at: u32) usize {
    return slotBytes(r, if (draft) |d| d.ring() else null, at);
}

pub fn slotBytes(r: *const Runner, ring: ?Ring, at: u32) usize {
    const pool = &r.gdn;
    const gdn = pool.budget.layers * (pool.budget.shape.state * 4 + pool.budget.shape.conv * 2);
    const kv = (r.caches.len / r.slots) * 2 * r.model.config.kv_heads * @as(usize, at) * ROW;
    return gdn + kv + if (ring) |g| g.bytes(at) else 0;
}

/// Every part between a slot's live state and `st`, in one blit pass on the model's queue.
fn copy(a: std.mem.Allocator, r: *const Runner, slot: u32, ring: ?Ring, st: *const State, to_state: bool) !void {
    var list: std.ArrayList(Part) = .empty;
    defer list.deinit(a);
    try parts(r, slot, ring, st.at, &list, a);
    var total: usize = 0;
    for (list.items) |p| {
        if (p.offset % 4 != 0 or p.len % 4 != 0) return error.SnapshotAlignment; // blits copy 4-byte words
        total += p.len;
    }
    if (total != st.bytes) return error.SnapshotSize;
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const cb = r.model.queue.commandBuffer();
    const e = cb.blit();
    var off: usize = 0;
    for (list.items) |p| {
        if (p.len > 0) if (to_state) e.copy(p.buffer, p.offset, st.buf, off, p.len) else e.copy(st.buf, off, p.buffer, p.offset, p.len);
        off += p.len;
    }
    e.end();
    cb.commit();
    cb.wait();
    if (cb.failure() != null) return error.SnapshotGpuFailure;
}

/// The live state after `at` prompt tokens, copied once the prompt pass stands at `at` with its GPU work done.
pub fn save(a: std.mem.Allocator, r: *const Runner, draft: ?*Draft, at: u32) !*State {
    return saveSlot(a, r, 0, if (draft) |d| d.ring() else null, at);
}

/// Slot `slot`'s state after `at` tokens, its taps ring standing there too when it has one.
pub fn saveSlot(a: std.mem.Allocator, r: *const Runner, slot: u32, ring: ?Ring, at: u32) !*State {
    if (at == 0 or r.failed or r.active != null or slot >= r.slots or r.offsets[slot] != at or (ring != null and ring.?.end.* != at)) return error.NotAtMark;
    const n = slotBytes(r, ring, at);
    const buf = try r.model.device.buffer(n, mtl.ResourceOptions.private | mtl.ResourceOptions.untracked);
    errdefer buf.deinit();
    const st = try a.create(State);
    errdefer a.destroy(st);
    st.* = .{ .buf = buf, .at = at, .bytes = n, .draft_end = if (ring) |g| g.end.* else 0 };
    try copy(a, r, slot, ring, st, true);
    return st;
}

/// The live state becomes `st`'s: the next chunk starts there, and the drafter rebuilds from the taps.
pub fn restore(a: std.mem.Allocator, r: *Runner, draft: ?*Draft, st: *const State) !void {
    if (st.bytes != bytes(r, draft, st.at) or st.at > r.capacity) return error.SnapshotLayout;
    if (draft) |d| try d.reset();
    try restoreSlot(a, r, 0, if (draft) |d| d.ring() else null, st);
}

/// Slot `slot` (reset first) becomes `st`'s state; a ring takes its taps and end.
pub fn restoreSlot(a: std.mem.Allocator, r: *Runner, slot: u32, ring: ?Ring, st: *const State) !void {
    if (st.bytes != slotBytes(r, ring, st.at) or st.at > r.capacity) return error.SnapshotLayout;
    try r.reset(slot);
    try copy(a, r, slot, ring, st, false);
    r.offsets[slot] = st.at;
    if (ring) |g| g.end.* = st.draft_end;
}

/// Slot `from`'s state moves to slot `to` (reset first) in one blit pass: DeltaNet states, K/V rows and taps ring.
pub fn moveSlot(a: std.mem.Allocator, r: *Runner, from: u32, from_ring: ?Ring, to: u32, to_ring: ?Ring) !void {
    const at = r.offsets[from];
    if (from == to or r.failed or r.active != null or from >= r.slots or to >= r.slots or (from_ring == null) != (to_ring == null)) return error.BadSlotMove;
    if (from_ring) |g| if (g.end.* != at) return error.NotAtMark;
    try r.reset(to);
    var src: std.ArrayList(Part) = .empty;
    defer src.deinit(a);
    var dst: std.ArrayList(Part) = .empty;
    defer dst.deinit(a);
    try parts(r, from, from_ring, at, &src, a);
    try parts(r, to, to_ring, at, &dst, a);
    if (src.items.len != dst.items.len) return error.SnapshotLayout;
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const cb = r.model.queue.commandBuffer();
    const e = cb.blit();
    for (src.items, dst.items) |x, y| {
        if (x.len != y.len or x.offset % 4 != 0 or y.offset % 4 != 0 or x.len % 4 != 0) {
            e.end();
            return error.SnapshotAlignment;
        }
        if (x.len > 0) e.copy(x.buffer, x.offset, y.buffer, y.offset, x.len);
    }
    e.end();
    cb.commit();
    cb.wait();
    if (cb.failure() != null) return error.SnapshotGpuFailure;
    r.offsets[to] = at;
    if (to_ring) |g| g.end.* = at;
}

pub fn drop(a: std.mem.Allocator, st: *State) void {
    st.buf.deinit();
    a.destroy(st);
}
