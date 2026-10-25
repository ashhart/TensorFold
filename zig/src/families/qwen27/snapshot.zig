//! The 27B's prompt states for the cache: DeltaNet states, attention K/V rows and the drafter's taps.
const std = @import("std");
const mtl = @import("metal");
const Runner = @import("decode_round.zig").Runner;
const Draft = @import("dflash/runtime_model.zig").Model;

/// One kept state: its bytes in a GPU buffer, its position and the drafter's committed end there.
pub const State = struct {
    buf: mtl.Buffer,
    at: u32,
    bytes: usize,
    draft_end: u64,
    residency: ?mtl.ResidencySet.Follow = null,
    cap: usize = 0, // the buffer's size when it came from (and goes back to) `spares`
    spares: ?*Spares = null,
    pending: ?mtl.CommandBuffer = null, // the save's blit, committed and not yet waited for
    failed: bool = false,

    /// The save's blit has landed (waited for only if it still runs); a failed one leaves the state unusable.
    pub fn landed(st: *State) !void {
        if (st.pending) |cb| {
            st.pending = null;
            cb.wait();
            st.failed = cb.failure() != null;
            cb.release();
        }
        if (st.failed) return error.SnapshotGpuFailure;
    }
};

/// Kept states' buffers by size: a dropped state's buffer waits for a later save it fits, so that save makes none.
pub const Spares = struct {
    bufs: [4]?mtl.Buffer = @splat(null),
    caps: [4]usize = @splat(0),
    follow: ?mtl.ResidencySet.Follow = null, // the residency set a buffer stays in while it is spare

    /// A new buffer's size for a state of `n` bytes: an eighth more in 32 MiB steps, so a later turn's fits it.
    pub fn size(n: usize) usize {
        return std.mem.alignForward(usize, n + n / 8, 32 << 20);
    }

    /// The smallest spare buffer that holds `n` bytes without wasting half again.
    fn fitting(p: *const Spares, n: usize) ?usize {
        var best: ?usize = null;
        for (p.bufs, p.caps, 0..) |b, cap, i| if (b != null and cap >= n and cap <= size(n) + size(n) / 2 and (best == null or cap < p.caps[best.?])) {
            best = i;
        };
        return best;
    }

    pub fn fits(p: *const Spares, n: usize) bool {
        return p.fitting(n) != null;
    }

    pub fn spare(p: *const Spares) u64 {
        var n: u64 = 0;
        for (p.bufs, p.caps) |b, cap| if (b != null) {
            n += cap;
        };
        return n;
    }

    /// Spare buffers go, the largest first, until at most `room` bytes stay spare.
    pub fn trim(p: *Spares, room: u64) void {
        while (p.spare() > room) {
            var big: usize = 0;
            for (p.bufs, p.caps, 0..) |b, cap, i| if (b != null and (p.bufs[big] == null or cap > p.caps[big])) {
                big = i;
            };
            p.free(p.bufs[big].?);
            p.bufs[big] = null;
            p.caps[big] = 0;
        }
    }

    pub fn deinit(p: *Spares) void {
        p.trim(0);
    }

    fn free(p: *const Spares, buf: mtl.Buffer) void {
        if (p.follow) |f| f.leave(f.ctx, buf);
        buf.deinit();
    }

    /// A spare buffer that holds `n` bytes, else a new one of `size(n)` in the residency set.
    fn take(p: *Spares, device: mtl.Device, n: usize) !struct { buf: mtl.Buffer, cap: usize } {
        if (p.fitting(n)) |i| {
            defer p.bufs[i] = null;
            return .{ .buf = p.bufs[i].?, .cap = p.caps[i] };
        }
        const cap = size(n);
        const buf = try device.buffer(cap, mtl.ResourceOptions.private | mtl.ResourceOptions.untracked);
        if (p.follow) |f| f.join(f.ctx, buf);
        return .{ .buf = buf, .cap = cap };
    }

    /// A buffer back: into a free place, else in place of a smaller one, else freed.
    fn give(p: *Spares, buf: mtl.Buffer, cap: usize) void {
        var at: usize = 0; // a free place, else the smallest buffer's
        for (p.bufs, p.caps, 0..) |b, c, i| {
            if (b == null) {
                at = i;
                break;
            }
            if (c < p.caps[at]) at = i;
        }
        if (p.bufs[at]) |old| {
            if (p.caps[at] >= cap) return p.free(buf);
            p.free(old);
        }
        p.bufs[at] = buf;
        p.caps[at] = cap;
    }
};

pub fn blank(a: std.mem.Allocator, r: *const Runner, ring: ?Ring, at: u32) !*State {
    const n = slotBytes(r, ring, at);
    const buf = try r.model.device.buffer(n, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
    errdefer buf.deinit();
    const st = try a.create(State);
    st.* = .{ .buf = buf, .at = at, .bytes = n, .draft_end = 0, .residency = r.residency };
    if (st.residency) |f| f.join(f.ctx, buf);
    return st;
}

const Part = struct { buffer: mtl.Buffer, offset: usize, len: usize };
pub const Ring = @import("tap_ring.zig").Ring;

fn parts(r: *const Runner, slot: u32, ring: ?Ring, at: u32, list: *std.ArrayList(Part), a: std.mem.Allocator, marked: bool) !void {
    if (slot >= r.ready) return error.SlotNotReady;
    for (0..r.gdn.budget.layers) |layer| for ([_]bool{ true, false }) |recurrent| {
        const span: @import("gdn.zig").Span = if (marked and r.offsets[slot] != at) blk: {
            const bank = r.gdn.marks orelse return error.NotAtMark;
            const mark = bank.index(slot, at, r.offsets[slot]) orelse return error.NotAtMark;
            break :blk try bank.span(layer, mark, recurrent);
        } else blk: {
            const live = try r.gdn.committedSpan(layer, slot, recurrent);
            break :blk .{ .buffer = live.buffer, .offset = live.offset, .bytes = live.bytes };
        };
        try list.append(a, .{ .buffer = span.buffer, .offset = span.offset, .len = span.bytes });
    };
    const row_bytes = r.model.config.head_dim * @sizeOf(u16);
    for (0..r.caches.len / r.slots) |layer| {
        const cache = r.caches[layer * r.slots + slot];
        for ([_]@import("projection.zig").Ref{ cache.keys, cache.values }, [_]u32{ cache.key_stride, cache.value_stride }) |ref, stride| {
            const head_stride = @as(usize, if (stride == 0) cache.capacity * r.model.config.head_dim else stride) * 2;
            for (0..r.model.config.kv_heads) |head| try list.append(a, .{ .buffer = ref.buffer, .offset = ref.offset + head * head_stride, .len = @as(usize, at) * row_bytes });
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
    const row_bytes = r.model.config.head_dim * @sizeOf(u16);
    const kv = (r.caches.len / r.slots) * 2 * r.model.config.kv_heads * @as(usize, at) * row_bytes;
    return gdn + kv + if (ring) |g| g.bytes(at) else 0;
}

/// Every part between a slot's live state and `st`, in one blit pass on the model's queue, waited for.
fn copy(a: std.mem.Allocator, r: *const Runner, slot: u32, ring: ?Ring, st: *const State, to_state: bool) !void {
    const cb = try encode(a, r, slot, ring, st, to_state);
    defer cb.release();
    cb.wait();
    if (cb.failure() != null) return error.SnapshotGpuFailure;
}

/// That blit pass committed and not waited for: the caller holds the returned (retained) command buffer.
fn encode(a: std.mem.Allocator, r: *const Runner, slot: u32, ring: ?Ring, st: *const State, to_state: bool) !mtl.CommandBuffer {
    var list: std.ArrayList(Part) = .empty;
    defer list.deinit(a);
    try parts(r, slot, ring, st.at, &list, a, to_state);
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
    return cb.retain();
}

/// The live state after `at` prompt tokens, copied once the prompt pass stands at `at` with its GPU work done.
pub fn save(a: std.mem.Allocator, r: *Runner, draft: ?*Draft, at: u32) !*State {
    return saveSlot(a, r, 0, if (draft) |d| d.ring() else null, at, null);
}

/// Slot `slot`'s state after `at` tokens (and its taps ring), its blit not waited for: see settleSlot and landed.
pub fn saveSlot(a: std.mem.Allocator, r: *Runner, slot: u32, ring: ?Ring, at: u32, spares: ?*Spares) !*State {
    if (at == 0 or r.failed or r.active != null or slot >= r.ready) return error.NotAtMark;
    const marked = if (r.gdn.marks) |m| m.index(slot, at, r.offsets[slot]) != null else false;
    if ((r.offsets[slot] != at and !marked) or (ring != null and (marked or ring.?.end.* != at))) return error.NotAtMark;
    const n = slotBytes(r, ring, at);
    const st = try a.create(State);
    errdefer a.destroy(st);
    if (spares) |p| {
        const got = try p.take(r.model.device, n);
        st.* = .{ .buf = got.buf, .at = at, .bytes = n, .draft_end = if (ring) |g| g.end.* else 0, .cap = got.cap, .spares = p };
    } else {
        const buf = try r.model.device.buffer(n, mtl.ResourceOptions.private | mtl.ResourceOptions.untracked);
        st.* = .{ .buf = buf, .at = at, .bytes = n, .draft_end = if (ring) |g| g.end.* else 0, .residency = r.residency };
        if (st.residency) |f| f.join(f.ctx, st.buf);
    }
    errdefer giveBack(st);
    const cb = try encode(a, r, slot, ring, st, true);
    st.pending = cb;
    r.readingSlot(slot, cb);
    return st;
}

/// A state's buffer to its spares, or out of the residency set and freed.
fn giveBack(st: *State) void {
    if (st.spares) |p| return p.give(st.buf, st.cap);
    if (st.residency) |f| f.leave(f.ctx, st.buf);
    st.buf.deinit();
}

/// The live state becomes `st`'s: the next chunk starts there, and the drafter rebuilds from the taps.
pub fn restore(a: std.mem.Allocator, r: *Runner, draft: ?*Draft, st: *State) !void {
    if (st.bytes != bytes(r, draft, st.at) or st.at > r.capacity) return error.SnapshotLayout;
    if (draft) |d| try d.reset();
    try restoreSlot(a, r, 0, if (draft) |d| d.ring() else null, st);
}

/// Slot `slot` (reset first) becomes `st`'s state; a ring takes its taps and end.
pub fn restoreSlot(a: std.mem.Allocator, r: *Runner, slot: u32, ring: ?Ring, st: *State) !void {
    if (st.bytes != slotBytes(r, ring, st.at) or st.at > r.capacity) return error.SnapshotLayout;
    try st.landed();
    try r.reset(slot);
    try copy(a, r, slot, ring, st, false);
    r.offsets[slot] = st.at;
    if (ring) |g| g.end.* = st.draft_end;
}

/// Slot `from`'s state moves to slot `to` (reset first) in one blit pass: DeltaNet states, K/V rows and taps ring.
pub fn moveSlot(a: std.mem.Allocator, r: *Runner, from: u32, from_ring: ?Ring, to: u32, to_ring: ?Ring) !void {
    const at = r.offsets[from];
    r.settleSlot(from);
    if (from == to or r.failed or r.active != null or from >= r.slots or to >= r.slots or (from_ring == null) != (to_ring == null)) return error.BadSlotMove;
    if (from_ring) |g| if (g.end.* != at) return error.NotAtMark;
    try r.reset(to);
    var src: std.ArrayList(Part) = .empty;
    defer src.deinit(a);
    var dst: std.ArrayList(Part) = .empty;
    defer dst.deinit(a);
    try parts(r, from, from_ring, at, &src, a, false);
    try parts(r, to, to_ring, at, &dst, a, false);
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
    st.landed() catch {};
    giveBack(st);
    a.destroy(st);
}
