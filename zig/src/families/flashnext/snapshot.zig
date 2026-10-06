//! Flash Next's prompt states for the prompt cache: cache rows, DeltaNet states and the n-gram tail at a mark.
const std = @import("std");
const mtl = @import("metal");
const fz = @import("replay.zig");
const Engine = @import("engine.zig").Engine;

const LAYERS = fz.LAYERS;
const CAP = fz.CAP;
const CS_ROW = fz.CS_ROW;
const SO_ROW = fz.SO_ROW;
const KEY_ROW = 256 * 2; // a key or value row of one head (bf16)
const RAW_ROW = 128 * 2; // an indexer key row (bf16)
const TAIL = fz.PLE_TAIL * fz.WIDE * 2; // the n-gram conv tail's rows

/// Where a state's DeltaNet rows live: the prompt pass's slot pair, the one layout snapshots are taken in today.
pub const Layout = enum { slots };

/// One kept state: its bytes, its position, the n-gram history there and the layout that wrote it.
pub const State = struct { buf: mtl.Buffer, at: usize, hist: [2]i64, bytes: usize, layout: Layout };

/// The layout the engine's prompt pass keeps DeltaNet states in (a decode path with in-place states adds its own).
fn layoutOf(_: *const Engine) Layout {
    return .slots;
}

const Part = struct { live: fz.Buf, len: usize };
const MAX_PARTS = 36 * 2 + 13 * 5 + 1;

/// Bytes of a state after `at` tokens: 36 DeltaNet states, 13 layers of keys, values and indexer keys, the tail.
pub fn bytes(at: usize) usize {
    return 36 * (CS_ROW + SO_ROW) + 13 * at * (4 * KEY_ROW + RAW_ROW) + TAIL;
}

/// The pieces in one order: DeltaNet states (from Engine.Passed past a mark), each attention layer's rows [0, at), the head's, the tail.
fn parts(m: *fz.Model, at: usize, passed: ?Engine.Passed, out: *[MAX_PARTS]Part) usize {
    var n: usize = 0;
    var li: usize = 0;
    for (&m.layers) |*L| if (L.linear) {
        if (passed) |ps| {
            out[n] = .{ .live = m.marks.?.cs_at(li, ps.slot), .len = CS_ROW };
            out[n + 1] = .{ .live = m.marks.?.so_at(li, ps.slot), .len = SO_ROW };
        } else {
            out[n] = .{ .live = .{ .b = L.cs[m.state].b, .off = L.cs[m.state].off + m.state_row * CS_ROW }, .len = CS_ROW };
            out[n + 1] = .{ .live = .{ .b = L.so[m.state].b, .off = L.so[m.state].off + m.state_row * SO_ROW }, .len = SO_ROW };
        }
        n += 2;
        li += 1;
    };
    for (&m.layers) |*L| if (!L.linear) {
        n += rows(L.keys, L.vals, L.raw, at, out[n..]);
    };
    n += rows(m.mtp.keys, m.mtp.vals, m.mtp.raw, at, out[n..]);
    out[n] = .{ .live = if (passed) |ps| ps.tail else .{ .b = m.ple.cin.b, .off = m.ple.cin.off }, .len = TAIL };
    return n + 1;
}

fn rows(keys: fz.Buf, vals: fz.Buf, raw: fz.Buf, at: usize, out: []Part) usize {
    for (0..2) |hd| {
        out[hd] = .{ .live = .{ .b = keys.b, .off = keys.off + hd * CAP * KEY_ROW }, .len = at * KEY_ROW };
        out[2 + hd] = .{ .live = .{ .b = vals.b, .off = vals.off + hd * CAP * KEY_ROW }, .len = at * KEY_ROW };
    }
    out[4] = .{ .live = raw, .len = at * RAW_ROW };
    return 5;
}

/// Copies each part between the live buffers and `buf` (to_buf: save) on the engine's queue, and waits.
fn copy(e: *Engine, buf: mtl.Buffer, at: usize, to_buf: bool, passed: ?Engine.Passed) !void {
    const r = e.r;
    var list: [MAX_PARTS]Part = undefined;
    const n = parts(e.m, at, passed, &list);
    const cb = r.queue.commandBuffer();
    r.enc = cb.compute(.serial);
    var off: usize = 0;
    for (list[0..n]) |p| {
        if (p.len == 0) continue;
        const kept: fz.Buf = .{ .b = buf, .off = off };
        if (to_buf) e.pr.copyWords(p.live, kept, p.len / 4) else e.pr.copyWords(kept, p.live, p.len / 4);
        off += p.len;
    }
    try e.m.finish(cb);
    if (off != bytes(at)) return error.SnapshotSize;
}

/// The live state after `at` prompt tokens into a new buffer; the prompt pass stands at `at`, or its last call passed it.
pub fn save(e: *Engine, gpa: std.mem.Allocator, at: usize) !*State {
    const passed: ?Engine.Passed = if (e.passed) |ps| (if (ps.at == at) ps else null) else null;
    if (at == 0 or (passed == null and e.m.pos != at)) return error.NotAtMark;
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const n = bytes(at);
    const buf = try e.r.device.buffer(n, fz.opts);
    errdefer buf.deinit();
    try copy(e, buf, at, true, passed);
    const st = try gpa.create(State);
    st.* = .{ .buf = buf, .at = at, .hist = if (passed) |ps| ps.hist else e.m.ple.hist, .bytes = n, .layout = layoutOf(e) };
    return st;
}

/// The live state becomes `st`'s: the next prompt chunk starts at its position (generateFrom with from = st.at).
pub fn restore(e: *Engine, st: *const State) !void {
    if (st.layout != layoutOf(e)) return error.SnapshotLayout; // the store drops it and the pass starts at 0
    const m = e.m;
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    e.hostMode();
    m.reset(); // a clean model if a copy fails; indexer blocks pool again from the restored keys (same kernel, same bits)
    try copy(e, st.buf, st.at, false, null);
    m.pos = st.at;
    m.ple.hist = st.hist;
}

pub fn drop(gpa: std.mem.Allocator, st: *State) void {
    st.buf.deinit();
    gpa.destroy(st);
}

test "a state's parts add up to its bytes" {
    try std.testing.expectEqual(@as(usize, 36 * (CS_ROW + SO_ROW) + 184_320), bytes(0));
    try std.testing.expectEqual(@as(usize, 13 * 2304), bytes(1) - bytes(0));
}
