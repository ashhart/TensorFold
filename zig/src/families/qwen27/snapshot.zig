//! The 27B's prompt states for the cache: DeltaNet states, attention K/V rows and the drafter's taps.
const std = @import("std");
const mtl = @import("metal");
const Runner = @import("decode_round.zig").Runner;
const Draft = @import("dflash/runtime_model.zig").Model;

/// One kept state: its bytes in a GPU buffer, its position and the drafter's committed end there.
pub const State = struct { buf: mtl.Buffer, at: u32, bytes: usize, draft_end: u64 };

const Part = struct { buffer: mtl.Buffer, offset: usize, len: usize };
const ROW = 256 * 2; // a key or value row of one head (bf16)

fn parts(r: *const Runner, draft: ?*const Draft, at: u32, list: *std.ArrayList(Part), a: std.mem.Allocator) !void {
    for (0..r.gdn.budget.layers) |layer| for ([_]bool{ true, false }) |recurrent| {
        const span = try r.gdn.committedSpan(layer, 0, recurrent);
        try list.append(a, .{ .buffer = span.buffer, .offset = span.offset, .len = span.bytes });
    };
    for (r.caches) |cache| for ([_]@import("projection.zig").Ref{ cache.keys, cache.values }, [_]u32{ cache.key_stride, cache.value_stride }) |ref, stride| {
        const head_stride = @as(usize, if (stride == 0) cache.capacity * 256 else stride) * 2;
        for (0..r.model.config.kv_heads) |head| try list.append(a, .{ .buffer = ref.buffer, .offset = ref.offset + head * head_stride, .len = @as(usize, at) * ROW });
    };
    if (draft) |d| try list.append(a, .{ .buffer = d.pending, .offset = 0, .len = ringBytes(d, at) });
}

/// The drafter's committed taps a state needs: ring slots [0, at) until the ring wraps, then all of it.
fn ringBytes(d: *const Draft, at: u32) usize {
    return @as(usize, @min(at, d.graph.config.window)) * d.graph.config.tapWidth() * 2;
}

/// Bytes of a state after `at` tokens.
pub fn bytes(r: *const Runner, draft: ?*const Draft, at: u32) usize {
    const pool = &r.gdn;
    const gdn = pool.budget.layers * (pool.budget.shape.state * 4 + pool.budget.shape.conv * 2);
    const kv = r.caches.len * 2 * r.model.config.kv_heads * @as(usize, at) * ROW;
    return gdn + kv + if (draft) |d| ringBytes(d, at) else 0;
}

/// Every part between the live state and `st`, in one blit pass on the model's queue.
fn copy(a: std.mem.Allocator, r: *const Runner, draft: ?*const Draft, st: *const State, to_state: bool) !void {
    var list: std.ArrayList(Part) = .empty;
    defer list.deinit(a);
    try parts(r, draft, st.at, &list, a);
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
pub fn save(a: std.mem.Allocator, r: *const Runner, draft: ?*const Draft, at: u32) !*State {
    if (at == 0 or r.failed or r.active != null or r.offsets[0] != at or (draft != null and draft.?.committed_end != at)) return error.NotAtMark;
    const n = bytes(r, draft, at);
    const buf = try r.model.device.buffer(n, mtl.ResourceOptions.private | mtl.ResourceOptions.untracked);
    errdefer buf.deinit();
    const st = try a.create(State);
    errdefer a.destroy(st);
    st.* = .{ .buf = buf, .at = at, .bytes = n, .draft_end = if (draft) |d| d.committed_end else 0 };
    try copy(a, r, draft, st, true);
    return st;
}

/// The live state becomes `st`'s: the next chunk starts there, and the drafter rebuilds from the taps.
pub fn restore(a: std.mem.Allocator, r: *Runner, draft: ?*Draft, st: *const State) !void {
    if (st.bytes != bytes(r, draft, st.at) or st.at > r.capacity) return error.SnapshotLayout;
    try r.reset(0);
    if (draft) |d| try d.reset();
    try copy(a, r, draft, st, false);
    r.offsets[0] = st.at;
    if (draft) |d| d.committed_end = st.draft_end;
}

pub fn drop(a: std.mem.Allocator, st: *State) void {
    st.buf.deinit();
    a.destroy(st);
}
