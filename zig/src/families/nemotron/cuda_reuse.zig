//! Nemotron CUDA prompt states for the shared cache: a prompt's key and value rows, the Mamba states, the head's rows.

const std = @import("std");
const cuda = @import("cuda");
const config = @import("config.zig");
const Config = config.Config;
const state = @import("cuda_state.zig");
const Engine = @import("cuda_engine.zig").Engine;
const Head = @import("cuda_mtp.zig").Head;
const marks = @import("cuda_marks.zig");

/// One kept state: its parts packed in one device copy, and where the engine stood.
pub const Copy = struct {
    buf: cuda.DeviceBuffer,
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
    return m[0] + m[1] + (c.count(.attention) + @intFromBool(head)) * 2 * at * state.kvRowBytes(c);
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

/// Where a state at `at` tokens keeps its SSM part (its conv part follows): after every layer's key and value rows.
fn mambaAt(c: Config, at: usize) usize {
    return 2 * c.count(.attention) * at * state.kvRowBytes(c);
}

/// Copies between the bound sequence and `buf` on the engine's stream (`into`: live to buf); the bytes moved.
fn move(t: *const Target, at: usize, buf: u64, into: bool, mamba: bool) !usize {
    const o = t.e.ops();
    var store: [max_parts]Part = undefined;
    var off: usize = 0;
    for (parts(t.e.c, t.e.max_len, planes(t), at, &store)) |p| {
        if (mamba or !p.mamba) if (into) try o.copy(buf + off, p.live, p.len) else try o.copy(p.live, buf + off, p.len);
        off += p.len;
    }
    return off;
}

/// Bytes a save at `at` takes, as the budget counts them: what `save` allocates.
pub fn bytes(ptr: *anyopaque, at: u32) u64 {
    const t: *Target = @ptrCast(@alignCast(ptr));
    return allocBytes(stateBytes(t.e.c, at, t.head != null));
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
    const copy = try alloc(t, at);
    errdefer drop(ptr, copy);
    if (try move(t, at, copy.buf.ptr, true, true) != stateBytes(e.c, at, t.head != null)) return error.SnapshotSize;
    return copy;
}

fn alloc(t: *const Target, at: usize) !*Copy {
    const e = t.e;
    const copy = try e.gpa.create(Copy);
    errdefer e.gpa.destroy(copy);
    copy.* = .{ .buf = try cuda.DeviceBuffer.alloc(e.ctx.d, allocBytes(stateBytes(e.c, at, t.head != null))), .head = t.head != null, .pos = at, .parity = e.parity, .rows = e.rows };
    return copy;
}

/// A row inside the chunk about to run: the chunk writes the copy's Mamba parts there, `finish` its key and value rows.
fn arm(t: *const Target, at: usize) !*anyopaque {
    const e = t.e;
    if (e.armed.n == marks.max) return error.TooManyMarks;
    const copy = try alloc(t, at);
    copy.whole = false;
    const ssm = copy.buf.ptr + mambaAt(e.c, at);
    e.armed.list[e.armed.n] = .{ .at = at, .ssm = ssm, .conv = ssm + state.mambaPlanes(e.c)[0], .copy = copy };
    e.armed.n += 1;
    return copy;
}

/// The armed rows' key and value rows, the head's too, once their chunk is queued: their copies are then whole.
pub fn finish(t: Target) !void {
    for (t.e.armed.marks()) |m| {
        const copy: *Copy = @ptrCast(@alignCast(m.copy));
        _ = try move(&t, m.at, copy.buf.ptr, true, false);
        copy.whole = true;
    }
}

/// Make the bound sequence `saved`'s; rows past its position stay, since every kernel writes a key before reading it.
pub fn restore(ptr: *anyopaque, owner: ?*anyopaque, saved: *anyopaque) anyerror!void {
    _ = owner;
    const t: *Target = @ptrCast(@alignCast(ptr));
    const copy: *Copy = @ptrCast(@alignCast(saved));
    const e = t.e;
    if (copy.head != (t.head != null)) return error.NoHead; // checked before any copy: nothing half restored
    if (!copy.whole) return error.Incomplete; // its chunk failed before `finish`
    _ = try move(t, copy.pos, copy.buf.ptr, false, true);
    e.pos = copy.pos;
    e.parity = copy.parity;
    e.prev_keep = 0;
    e.rows = copy.rows;
    if (t.head) |h| h.pos = copy.pos;
}

/// Free a kept state's device copy.
pub fn drop(ptr: *anyopaque, saved: *anyopaque) void {
    const t: *Target = @ptrCast(@alignCast(ptr));
    const copy: *Copy = @ptrCast(@alignCast(saved));
    t.e.armed.forget(saved); // a copy evicted before its chunk ran: the chunk must not write it
    copy.buf.free();
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

test "an armed row's Mamba parts sit where a whole copy puts them" {
    const c = testConfig();
    const p: Planes = .{ .k = 1 << 40, .v = 2 << 40, .ssm = 3 << 40, .conv = 4 << 40, .head = .{ 5 << 40, 6 << 40 } };
    var store: [max_parts]Part = undefined;
    var off: usize = 0;
    for (parts(c, 70144, p, 1500, &store)) |x| {
        if (x.live == p.ssm) try std.testing.expectEqual(mambaAt(c, 1500), off);
        if (x.live == p.conv) try std.testing.expectEqual(mambaAt(c, 1500) + state.mambaPlanes(c)[0], off);
        off += x.len;
    }
}

test "a state is charged the whole pages it allocates" {
    try std.testing.expectEqual(@as(usize, 62 << 20), allocBytes(64135168)); // 2,100 tokens of Nemotron 3.5 Lightning
    try std.testing.expectEqual(@as(usize, 48 << 20), allocBytes(48 << 20));
    try std.testing.expectEqual(@as(usize, 50 << 20), allocBytes((48 << 20) + 1));
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
