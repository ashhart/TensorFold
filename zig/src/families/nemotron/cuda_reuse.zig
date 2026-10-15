//! Nemotron CUDA prompt states for the shared cache: a prompt's key and value rows, the Mamba states, the head's rows.
//! On the device, or (a discrete card's `host`) in pinned host memory: the key and value rows, then the Mamba states.

const std = @import("std");
const cuda = @import("cuda");
const config = @import("config.zig");
const Config = config.Config;
const state = @import("cuda_state.zig");
const Engine = @import("cuda_engine.zig").Engine;
const Head = @import("cuda_mtp.zig").Head;
const marks = @import("cuda_marks.zig");

/// One kept state: its Mamba states in one device buffer, its key and value rows in another, where the engine stood.
/// A host state keeps both in `host` (key and value rows first); `mamba` is then an armed row's staging until `finish`.
pub const Copy = struct {
    mamba: cuda.DeviceBuffer,
    kv: cuda.DeviceBuffer,
    host: ?cuda.HostBuffer = null,
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

    /// Kept states in pinned host memory (Engine.snap_host, a discrete card): the budget is the host's.
    fn host(t: *const Target) bool {
        return t.e.snap_host;
    }
};

/// `len` bytes at `live` in the bound sequence's planes; `mamba`: the SSM or conv state, which an armed chunk writes.
const Part = struct { live: u64, len: usize, mamba: bool = false };

/// Where the bound sequence keeps what a state copies: KV planes, SSM and conv planes, the head's KV planes.
const Planes = struct { k: u64, v: u64, ssm: u64, conv: u64, head: ?[2]u64 = null };

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

/// A state's Mamba buffer, the same for every state: a spare one takes any.
fn mambaBytes(c: Config) usize {
    const m = state.mambaPlanes(c);
    return allocBytes(m[0] + m[1]);
}

/// A state's parts in copy order: every layer's first `at` key rows, its value rows, the Mamba states, the head's.
fn parts(c: Config, max_len: usize, p: Planes, at: usize, out: *[max_parts]Part) []const Part {
    const row = state.kvRowBytes(c);
    const m = state.mambaPlanes(c);
    var n: usize = 0;
    for ([2]u64{ p.k, p.v }) |plane| for (0..c.count(.attention)) |l| {
        out[n] = .{ .live = plane + l * max_len * row, .len = at * row };
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
    return .{ .k = b.k_cache, .v = b.v_cache, .ssm = b.ssm, .conv = b.conv_base, .head = if (t.head) |h| .{ h.k_cache, h.v_cache } else null };
}

/// Copies between the bound sequence and `copy` on the engine's stream (`into`: live to copy); the bytes moved.
fn move(t: *const Target, at: usize, copy: *const Copy, into: bool, mamba: bool) !usize {
    const o = t.e.ops();
    var store: [max_parts]Part = undefined;
    var off = [2]usize{ 0, 0 }; // into the key and value buffer, into the Mamba buffer
    for (parts(t.e.c, t.e.max_len, planes(t), at, &store)) |p| {
        const buf = if (p.mamba) copy.mamba.ptr + off[1] else copy.kv.ptr + off[0];
        if (mamba or !p.mamba) if (into) try o.copy(buf, p.live, p.len) else try o.copy(p.live, buf, p.len);
        off[@intFromBool(p.mamba)] += p.len;
    }
    return off[0] + off[1];
}

/// Copies between the bound sequence and a host copy (`into`: live to host), key and value rows first; `mamba`: the
/// Mamba states too, else from `staging` (an armed row's) when given. Queued on the engine's stream.
fn hostMove(t: *const Target, at: usize, host: []u8, into: bool, mamba: bool, staging: ?u64) !void {
    const o = t.e.ops();
    var store: [max_parts]Part = undefined;
    var off = [2]usize{ 0, kvBytes(t.e.c, at, t.head != null) };
    for (parts(t.e.c, t.e.max_len, planes(t), at, &store)) |p| {
        const at_host = off[@intFromBool(p.mamba)];
        const buf = host[at_host..][0..p.len];
        if (!p.mamba) {
            if (into) try o.download(buf, p.live) else try o.upload(p.live, buf);
        } else if (mamba) {
            if (into) try o.download(buf, p.live) else try o.upload(p.live, buf);
        } else if (staging) |st| try o.download(buf, st + (at_host - kvBytes(t.e.c, at, t.head != null))); // SSM, then conv
        off[@intFromBool(p.mamba)] += p.len;
    }
}

/// Bytes a save at `at` takes, as the budget counts them: what `save` allocates, spare or new (host: its bytes).
pub fn bytes(ptr: *anyopaque, at: u32) u64 {
    const t: *Target = @ptrCast(@alignCast(ptr));
    if (t.host()) return stateBytes(t.e.c, at, t.head != null);
    return mambaBytes(t.e.c) + allocBytes(kvBytes(t.e.c, at, t.head != null));
}

/// Sets spare Mamba buffers aside in a cache of `budget` bytes, a quarter of it at most; the bytes they take.
pub fn ready(t: *Target, budget: u64) !u64 {
    const e = t.e;
    const len = mambaBytes(e.c);
    // host states: one device staging buffer for an armed row, out of the device's room rather than the host budget
    e.spares.want = if (t.host()) @min(marks.spares_max, 1) else @min(marks.spares_max, budget / 4 / len);
    e.spares.fill(e.ctx.d, len);
    return if (t.host()) 0 else e.spares.want * len;
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
    if (t.host()) {
        const copy = try e.gpa.create(Copy);
        errdefer e.gpa.destroy(copy);
        copy.* = .{ .mamba = .{ .ptr = 0, .len = 0, .d = e.ctx.d }, .kv = .{ .ptr = 0, .len = 0, .d = e.ctx.d }, .head = t.head != null, .pos = at, .parity = e.parity, .rows = e.rows };
        copy.host = try cuda.HostBuffer.alloc(e.ctx.d, stateBytes(e.c, at, copy.head));
        errdefer copy.host.?.free();
        try hostMove(t, at, copy.host.?.bytes, true, true, null);
        return copy;
    }
    const copy = try alloc(t, at, true);
    errdefer drop(ptr, copy);
    if (try move(t, at, copy, true, true) != stateBytes(e.c, at, t.head != null)) return error.SnapshotSize;
    return copy;
}

/// A copy with a spare Mamba buffer when one waits, and its key and value buffer now (`kv`) or at `finish`.
fn alloc(t: *const Target, at: usize, kv: bool) !*Copy {
    const e = t.e;
    const copy = try e.gpa.create(Copy);
    errdefer e.gpa.destroy(copy);
    var m = e.spares.take() orelse try cuda.DeviceBuffer.alloc(e.ctx.d, mambaBytes(e.c));
    errdefer if (!e.spares.give(m)) m.free();
    const rows = try cuda.DeviceBuffer.alloc(e.ctx.d, if (kv and !t.host()) allocBytes(kvBytes(e.c, at, t.head != null)) else 0);
    copy.* = .{ .mamba = m, .kv = rows, .head = t.head != null, .pos = at, .parity = e.parity, .rows = e.rows };
    return copy;
}

/// A row inside the chunk about to run: the chunk writes the copy's Mamba parts there, `finish` its key and value rows.
fn arm(t: *const Target, at: usize) !*anyopaque {
    const e = t.e;
    if (e.armed.n == marks.max) return error.TooManyMarks;
    const copy = try alloc(t, at, false); // nothing allocated before the chunk when a spare waits
    copy.whole = false;
    e.armed.list[e.armed.n] = .{ .at = at, .ssm = copy.mamba.ptr, .conv = copy.mamba.ptr + state.mambaPlanes(e.c)[0], .copy = copy };
    e.armed.n += 1;
    return copy;
}

/// Once a chunk is queued, while the GPU runs it: the armed rows' key and value rows (the head's too), then new spares.
pub fn finish(t: Target) !void {
    const e = t.e;
    for (e.armed.marks()) |m| {
        const copy: *Copy = @ptrCast(@alignCast(m.copy));
        if (t.host()) {
            // the chunk wrote this row's Mamba states into the staging buffer: rows and states to the host, then the
            // staging back (the stream orders its next writer after these copies)
            var h = cuda.HostBuffer.alloc(e.ctx.d, stateBytes(e.c, m.at, copy.head)) catch continue; // stays not whole
            hostMove(&t, m.at, h.bytes, true, false, copy.mamba.ptr) catch {
                h.free();
                continue;
            };
            copy.host = h;
            if (!e.spares.give(copy.mamba)) copy.mamba.free();
            copy.mamba = .{ .ptr = 0, .len = 0, .d = e.ctx.d };
            copy.whole = true;
            continue;
        }
        copy.kv = cuda.DeviceBuffer.alloc(e.ctx.d, allocBytes(kvBytes(e.c, m.at, copy.head))) catch continue; // stays not whole
        _ = try move(&t, m.at, copy, true, false);
        copy.whole = true;
    }
    e.spares.fill(e.ctx.d, mambaBytes(e.c));
}

/// What a kept state holds on the device, both buffers.
pub fn held(saved: *anyopaque) usize {
    const copy: *Copy = @ptrCast(@alignCast(saved));
    return copy.mamba.len + copy.kv.len + if (copy.host) |h| h.bytes.len else 0;
}

/// A kept state's bytes in `out` (its key and value rows, then its Mamba states), after the engine's stream.
pub fn download(t: *const Target, saved: *anyopaque, out: []u8) !void {
    const copy: *Copy = @ptrCast(@alignCast(saved));
    if (copy.host) |h| {
        try t.e.stream.synchronize();
        @memcpy(out[0..h.bytes.len], h.bytes);
        return;
    }
    const kv = kvBytes(t.e.c, copy.pos, copy.head);
    try t.e.ops().download(out[0..kv], copy.kv.ptr);
    try t.e.ops().download(out[kv..], copy.mamba.ptr);
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
    if (copy.host) |h| {
        try hostMove(t, copy.pos, h.bytes, false, true, null);
        try e.stream.synchronize(); // the store may drop the host copy once this returns
    } else _ = try move(t, copy.pos, copy, false, true);
    e.pos = copy.pos;
    e.parity = copy.parity;
    e.prev_keep = 0;
    e.rows = copy.rows;
    if (t.head) |h| h.pos = copy.pos;
}

/// Free a kept state's device copy; its Mamba buffer waits as a spare while fewer than the cache set aside do.
pub fn drop(ptr: *anyopaque, saved: *anyopaque) void {
    const t: *Target = @ptrCast(@alignCast(ptr));
    const copy: *Copy = @ptrCast(@alignCast(saved));
    t.e.armed.forget(saved); // a copy evicted before its chunk ran: the chunk must not write it
    if (copy.host) |*h| {
        // queued copies may still read or write it: they finish before the host memory goes
        t.e.stream.synchronize() catch {};
        h.free();
    }
    if (copy.mamba.len > 0 and !t.e.spares.give(copy.mamba)) copy.mamba.free();
    if (copy.kv.len > 0) copy.kv.free();
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
    const p: Planes = .{ .k = 1 << 40, .v = 2 << 40, .ssm = 3 << 40, .conv = 4 << 40, .head = .{ 5 << 40, 6 << 40 } };
    var store: [max_parts]Part = undefined;
    var off = [2]usize{ 0, 0 };
    for (parts(c, 70144, p, 1500, &store)) |x| {
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
    const p: Planes = .{ .k = 1 << 40, .v = 2 << 40, .ssm = 3 << 40, .conv = 4 << 40, .head = .{ 5 << 40, 6 << 40 } };
    var store: [max_parts]Part = undefined;
    const want = [_]Part{
        .{ .live = 1 << 40, .len = 2100 * row },          .{ .live = (1 << 40) + max_len * row, .len = 2100 * row },
        .{ .live = 2 << 40, .len = 2100 * row },          .{ .live = (2 << 40) + max_len * row, .len = 2100 * row },
        .{ .live = 3 << 40, .len = m[0], .mamba = true }, .{ .live = 4 << 40, .len = m[1], .mamba = true },
        .{ .live = 5 << 40, .len = 2100 * row },          .{ .live = 6 << 40, .len = 2100 * row },
    };
    const got = parts(c, max_len, p, 2100, &store);
    try std.testing.expectEqualSlices(Part, &want, got);
    var total: usize = 0;
    for (got) |x| total += x.len;
    try std.testing.expectEqual(stateBytes(c, 2100, true), total);
    total = 0;
    var p2 = p;
    p2.head = null;
    for (parts(c, max_len, p2, 2100, &store)) |x| total += x.len;
    try std.testing.expectEqual(stateBytes(c, 2100, false), total);
}
