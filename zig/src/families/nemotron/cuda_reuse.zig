//! Nemotron CUDA prompt states for the shared cache: a prompt's key and value rows, the Mamba states, the head's rows.

const std = @import("std");
const cuda = @import("cuda");
const config = @import("config.zig");
const Config = config.Config;
const state = @import("cuda_state.zig");
const Engine = @import("cuda_engine.zig").Engine;
const Head = @import("cuda_mtp.zig").Head;
const marks = @import("cuda_marks.zig");

/// One kept state: its Mamba states in a snapshot slot, its key and value rows in a buffer, where the engine stood.
pub const Copy = struct {
    mamba: u64, // its slot's address (marks.Slots)
    slot: u32,
    kv: cuda.DeviceBuffer,
    head: bool,
    pos: usize,
    parity: usize,
    rows: usize,
    whole: bool = true, // false while an armed mark's chunk has not written all its parts
};

/// The engine and head a cache save copies. The native host keeps one of these for the loaded model.
pub const Target = struct {
    e: *Engine,
    head: ?*Head,
    ctx: ?*const cuda.Context = null, // made current before a disk copy, which may run before any backend call
};

/// A kept state's two buffers as the disk sees them: the key and value rows (`kv` bytes), then the Mamba states.
pub const Shape = struct { layers: usize, row: usize, head: bool, kv: usize, mamba: usize };

pub fn shape(t: *const Target, at: usize) Shape {
    const m = state.mambaPlanes(t.e.c);
    return .{ .layers = t.e.c.count(.attention), .row = state.kvRowBytes(t.e.c), .head = t.head != null, .kv = kvBytes(t.e.c, at, t.head != null), .mamba = m[0] + m[1] };
}

/// A whole kept state of `at` tokens, its buffers empty for a disk read to fill (`fill`).
pub fn blank(t: *const Target, at: usize) !*Copy {
    return alloc(t, at, true);
}

/// `host` to or from (`out`) bytes [off, off + host.len) of a kept state's key and value buffer or its Mamba buffer.
pub fn hostCopy(t: *const Target, saved: *anyopaque, mamba: bool, off: usize, host: []u8, out: bool) !void {
    const copy: *Copy = @ptrCast(@alignCast(saved));
    const base = (if (mamba) copy.mamba else copy.kv.ptr) + off;
    if (out) try t.e.ops().download(host, base) else try t.e.ops().upload(base, host);
}

/// Copies queued by `hostCopy` finished (the host's bytes may be read or reused).
pub fn settle(t: *const Target) !void {
    try t.e.stream.synchronize();
}

/// A kept state's engine words for the disk: the parity and row count where it stood.
pub fn words(saved: *anyopaque) [2]u64 {
    const copy: *Copy = @ptrCast(@alignCast(saved));
    return .{ copy.parity, copy.rows };
}

pub fn setWords(saved: *anyopaque, w: [2]u64) void {
    const copy: *Copy = @ptrCast(@alignCast(saved));
    copy.parity = @intCast(w[0]);
    copy.rows = @intCast(w[1]);
}

/// `len` bytes at `live` in the bound sequence's planes; `mamba`: the SSM or conv state, which an armed chunk writes.
const Part = struct { live: u64, len: usize, mamba: bool = false };

/// Where the bound sequence keeps what a state copies: KV planes (`stride` bytes a layer), SSM and conv, the head's.
const Planes = struct { k: u64, v: u64, ssm: u64, conv: u64, head: ?[2]u64 = null, stride: usize };

const max_parts = 2 * config.max_layers + 4;

/// cuMemAlloc gives a buffer this size whole 2 MiB pages: a state allocates, and the budget counts, whole pages.
const page: usize = 2 << 20;

fn allocBytes(n: usize) usize {
    return std.mem.alignForward(usize, n, page);
}

/// Bytes of a state at `at` tokens: the Mamba states, then a key and a value row a token a layer (the head's too).
pub fn stateBytes(c: Config, at: usize, head: bool) usize {
    const m = state.mambaPlanes(c);
    return m[0] + m[1] + kvBytes(c, at, head);
}

fn kvBytes(c: Config, at: usize, head: bool) usize {
    return (c.count(.attention) + @intFromBool(head)) * 2 * at * state.kvRowBytes(c);
}

/// A state's Mamba buffer, the same for every state: one snapshot slot.
pub fn mambaBytes(c: Config) usize {
    const m = state.mambaPlanes(c);
    return allocBytes(m[0] + m[1]);
}

/// A state's parts in copy order: every layer's first `at` key rows, its value rows, the Mamba states, the head's.
fn parts(c: Config, p: Planes, at: usize, out: *[max_parts]Part) []const Part {
    const row = state.kvRowBytes(c);
    const m = state.mambaPlanes(c);
    var n: usize = 0;
    for ([2]u64{ p.k, p.v }) |plane| for (0..c.count(.attention)) |l| {
        out[n] = .{ .live = plane + l * p.stride, .len = at * row };
        n += 1;
    };
    out[n] = .{ .live = p.ssm, .len = m[0], .mamba = true };
    out[n + 1] = .{ .live = p.conv, .len = m[1], .mamba = true };
    n += 2;
    if (p.head) |h| for (h) |plane| {
        out[n] = .{ .live = plane, .len = at * row };
        n += 1;
    };
    return out[0..n];
}

fn planes(t: *const Target) Planes {
    const b = &t.e.b;
    const row = state.kvRowBytes(t.e.c);
    const stride = if (b.table != 0) 2 * state.chunk_keys * row else t.e.max_len * row;
    return .{ .k = b.k_cache, .v = b.v_cache, .ssm = b.ssm, .conv = b.conv_base, .head = if (t.head) |h| .{ h.k_cache, h.v_cache } else null, .stride = stride };
}

/// Copies between the bound sequence and `copy` on the engine's stream (`into`: live to copy); the bytes moved.
fn move(t: *const Target, at: usize, copy: *const Copy, into: bool, mamba: bool) !usize {
    const o = t.e.ops();
    var store: [max_parts]Part = undefined;
    var off = [2]usize{ 0, 0 }; // into the key and value buffer, into the Mamba buffer
    for (parts(t.e.c, planes(t), at, &store)) |p| {
        if (p.mamba) {
            if (mamba) if (into) try o.copy(copy.mamba + off[1], p.live, p.len) else try o.copy(p.live, copy.mamba + off[1], p.len);
        } else try rows(t, p, copy.kv.ptr + off[0], into);
        off[@intFromBool(p.mamba)] += p.len;
    }
    return off[0] + off[1];
}

/// A layer's first rows between the bound sequence and `buf`: one copy, or one a page its table names.
fn rows(t: *const Target, p: Part, buf: u64, into: bool) !void {
    const o = t.e.ops();
    const pages = t.e.bound.pages;
    if (t.e.b.table == 0) return if (into) o.copy(buf, p.live, p.len) else o.copy(p.live, buf, p.len);
    const step = state.chunk_keys * state.kvRowBytes(t.e.c);
    var done: usize = 0;
    for (pages) |id| {
        if (done == p.len) break;
        const n = @min(step, p.len - done);
        const live = p.live + id * t.e.pageBytes();
        if (into) try o.copy(buf + done, live, n) else try o.copy(live, buf + done, n);
        done += n;
    }
    if (done != p.len) return error.PastPages;
}

/// Bytes a save at `at` takes from the copied-KV budget: its key and value rows (its Mamba states take a slot).
pub fn bytes(ptr: *anyopaque, at: u32) u64 {
    const t: *Target = @ptrCast(@alignCast(ptr));
    return allocBytes(kvBytes(t.e.c, at, t.head != null));
}

/// `n` snapshot slots made now, in one device buffer: the saves' Mamba states never allocate later; their bytes.
pub fn slots(t: *Target, n: usize) !u64 {
    try t.e.snaps.init(t.e.ctx.d, n, mambaBytes(t.e.c));
    return n * mambaBytes(t.e.c);
}

/// Copy the live state after `at` prompt tokens. The pass is standing there.
pub fn save(ptr: *anyopaque, owner: ?*anyopaque, at: u32) anyerror!*anyopaque {
    _ = owner;
    const t: *Target = @ptrCast(@alignCast(ptr));
    const e = t.e;
    // a prompt mark: no window's rows wait to replay, so raw, xc and dt hold nothing a later pass reads
    if (e.prev_keep != 0) return error.NotAtMark;
    if (e.armed.open) |span| if (at > span[0] and at < span[1]) return arm(t, at);
    if (e.pos != at) return error.NotAtMark;
    if (t.head) |h| if (h.pos != at) return error.NotAtMark;
    const copy = try alloc(t, at, true);
    errdefer drop(ptr, copy);
    if (try move(t, at, copy, true, true) != stateBytes(e.c, at, t.head != null)) return error.SnapshotSize;
    return copy;
}

/// A copy with a free snapshot slot, and its key and value buffer now (`kv`) or at `finish`.
fn alloc(t: *const Target, at: usize, kv: bool) !*Copy {
    const e = t.e;
    const copy = try e.gpa.create(Copy);
    errdefer e.gpa.destroy(copy);
    const slot = e.snaps.take() orelse return error.NoSnapshotSlot;
    errdefer e.snaps.give(slot);
    const buf = try cuda.DeviceBuffer.alloc(e.ctx.d, if (kv) allocBytes(kvBytes(e.c, at, t.head != null)) else 0);
    copy.* = .{ .mamba = e.snaps.at(slot), .slot = slot, .kv = buf, .head = t.head != null, .pos = at, .parity = e.parity, .rows = e.rows };
    return copy;
}

/// A row inside the chunk about to run: the chunk writes the copy's Mamba parts there, `finish` its key and value rows.
fn arm(t: *const Target, at: usize) !*anyopaque {
    const e = t.e;
    if (e.armed.n == marks.max) return error.TooManyMarks;
    const copy = try alloc(t, at, false); // nothing allocated before the chunk when a spare waits
    copy.whole = false;
    e.armed.list[e.armed.n] = .{ .at = at, .ssm = copy.mamba, .conv = copy.mamba + state.mambaPlanes(e.c)[0], .copy = copy };
    e.armed.n += 1;
    return copy;
}

/// Once a chunk is queued, while the GPU runs it: the armed rows' key and value rows (the head's too).
pub fn finish(t: Target) !void {
    const e = t.e;
    for (e.armed.marks()) |m| {
        const copy: *Copy = @ptrCast(@alignCast(m.copy));
        copy.kv = cuda.DeviceBuffer.alloc(e.ctx.d, allocBytes(kvBytes(e.c, m.at, copy.head))) catch continue; // stays not whole
        _ = try move(&t, m.at, copy, true, false);
        copy.whole = true;
    }
}

/// What a kept state holds on the device: its key and value buffer, and its slot.
pub fn held(t: *const Target, saved: *anyopaque) usize {
    const copy: *Copy = @ptrCast(@alignCast(saved));
    return mambaBytes(t.e.c) + copy.kv.len;
}

/// A kept state's bytes in `out` (its key and value rows, then its Mamba states), after the engine's stream.
pub fn download(t: *const Target, saved: *anyopaque, out: []u8) !void {
    const copy: *Copy = @ptrCast(@alignCast(saved));
    const kv = kvBytes(t.e.c, copy.pos, copy.head);
    try t.e.ops().download(out[0..kv], copy.kv.ptr);
    try t.e.ops().download(out[kv..], copy.mamba);
    try t.e.stream.synchronize();
}

/// Make the bound sequence `saved`'s; rows past its position stay, since every kernel writes a key before reading it.
pub fn restore(ptr: *anyopaque, owner: ?*anyopaque, saved: *anyopaque) anyerror!void {
    _ = owner;
    const t: *Target = @ptrCast(@alignCast(ptr));
    const copy: *Copy = @ptrCast(@alignCast(saved));
    const e = t.e;
    if (copy.head != (t.head != null)) return error.NoHead; // checked before any copy: nothing half restored
    if (!copy.whole) return error.Incomplete; // its chunk failed before `finish`
    _ = try move(t, copy.pos, copy, false, true);
    e.pos = copy.pos;
    e.parity = copy.parity;
    e.prev_keep = 0;
    e.rows = copy.rows;
    if (t.head) |h| h.pos = copy.pos;
}

/// Free a kept state's device copy; its slot goes back to the free ones.
pub fn drop(ptr: *anyopaque, saved: *anyopaque) void {
    const t: *Target = @ptrCast(@alignCast(ptr));
    const copy: *Copy = @ptrCast(@alignCast(saved));
    t.e.armed.forget(saved); // a copy evicted before its chunk ran: the chunk must not write it
    t.e.snaps.give(copy.slot);
    copy.kv.free();
    t.e.gpa.destroy(copy);
}

fn testConfig() Config {
    var c: Config = undefined;
    c.mamba_heads = 64;
    c.mamba_head_dim = 64;
    c.groups = 8;
    c.state = 128;
    c.head_dim = 128;
    c.kv_heads = 2;
    c.layers = 5;
    c.kinds[0..5].* = .{ .mamba, .attention, .moe, .mamba, .attention };
    return c;
}

test "a kept state holds its prompt's key and value rows beside the whole Mamba states" {
    const c = testConfig();
    const fixed = 2 * (64 * 64 * 128 * 4 + 3 * 6144 * 2);
    try std.testing.expectEqual(@as(usize, fixed), stateBytes(c, 0, true));
    try std.testing.expectEqual(@as(usize, 3 * 2 * 2100 * 512), stateBytes(c, 2100, true) - fixed); // two layers, the head
    try std.testing.expectEqual(@as(usize, 2 * 2 * 2100 * 512), stateBytes(c, 2100, false) - fixed);
}

test "a state's Mamba parts fill its Mamba buffer, SSM first, where an armed row's chunk writes them" {
    const c = testConfig();
    const m = state.mambaPlanes(c);
    const p: Planes = .{ .k = 1 << 40, .v = 2 << 40, .ssm = 3 << 40, .conv = 4 << 40, .head = .{ 5 << 40, 6 << 40 }, .stride = 70144 * 512 };
    var store: [max_parts]Part = undefined;
    var off = [2]usize{ 0, 0 };
    for (parts(c, p, 1500, &store)) |x| {
        if (x.live == p.ssm) try std.testing.expectEqual(@as(usize, 0), off[1]);
        if (x.live == p.conv) try std.testing.expectEqual(m[0], off[1]);
        off[@intFromBool(x.mamba)] += x.len;
    }
    try std.testing.expectEqual(kvBytes(c, 1500, true), off[0]);
    try std.testing.expect(off[1] <= mambaBytes(c));
}

test "a state is charged the whole pages it allocates" {
    try std.testing.expectEqual(@as(usize, 48 << 20), allocBytes(48 << 20));
    try std.testing.expectEqual(@as(usize, 50 << 20), allocBytes((48 << 20) + 1));
    try std.testing.expectEqual(@as(usize, 48 << 20), allocBytes(49082368)); // Nemotron 3.5 Lightning's Mamba buffer
    try std.testing.expectEqual(@as(usize, 14 << 20), allocBytes(7 * 2 * 2048 * 512)); // and its rows at 2,048 tokens
}

test "a state's parts start at each layer's row 0 and add up to its bytes" {
    const c = testConfig();
    const max_len: usize = 70144;
    const row: usize = 512;
    const m = state.mambaPlanes(c);
    const p: Planes = .{ .k = 1 << 40, .v = 2 << 40, .ssm = 3 << 40, .conv = 4 << 40, .head = .{ 5 << 40, 6 << 40 }, .stride = max_len * row };
    var store: [max_parts]Part = undefined;
    const want = [_]Part{
        .{ .live = 1 << 40, .len = 2100 * row },          .{ .live = (1 << 40) + max_len * row, .len = 2100 * row },
        .{ .live = 2 << 40, .len = 2100 * row },          .{ .live = (2 << 40) + max_len * row, .len = 2100 * row },
        .{ .live = 3 << 40, .len = m[0], .mamba = true }, .{ .live = 4 << 40, .len = m[1], .mamba = true },
        .{ .live = 5 << 40, .len = 2100 * row },          .{ .live = 6 << 40, .len = 2100 * row },
    };
    const got = parts(c, p, 2100, &store);
    try std.testing.expectEqualSlices(Part, &want, got);
    var total: usize = 0;
    for (got) |x| total += x.len;
    try std.testing.expectEqual(stateBytes(c, 2100, true), total);
    total = 0;
    var p2 = p;
    p2.head = null;
    for (parts(c, p2, 2100, &store)) |x| total += x.len;
    try std.testing.expectEqual(stateBytes(c, 2100, false), total);
}
