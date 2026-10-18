//! Actual GLM ring admission, copies and restored bases across rejected MTP proposals, without CUDA execution.
const std = @import("std");
const cuda = @import("cuda");
const glm = @import("glm");
const State = glm.forward.State;
const t = std.testing;

fn copy(dst: u64, src: u64, n: usize, _: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
    const a: [*]u8 = @ptrFromInt(dst);
    const b: [*]const u8 = @ptrFromInt(src);
    std.mem.copyForwards(u8, a[0..n], b[0..n]);
    return 0;
}

fn set(dst: u64, fill: c_uint, n: usize, _: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
    const p: [*]u32 = @ptrFromInt(dst);
    @memset(p[0..n], fill);
    return 0;
}

fn value(pos: usize) u8 {
    return @intCast(pos % 251);
}

fn check(boundary: usize, resumed: bool, rewind: usize) !void {
    var d: cuda.Driver = undefined;
    d.api.cuMemcpyDtoDAsync_v2 = copy;
    d.api.cuMemsetD32Async = set;
    const stream: cuda.Stream = .{ .d = &d, .handle = null };
    var keys: [4096]u8 = undefined;
    var gates: [4096]u8 = undefined;
    const base = boundary - keys.len;
    for (&keys, &gates, 0..) |*k, *g, i| {
        k.* = value(base + i);
        g.* = k.* ^ 0x55;
    }
    var rings = [_]u64{ @intFromPtr(&keys), @intFromPtr(&gates) };
    var pos: u32 = 0;
    var mtp: u32 = 0;
    var st: State = undefined;
    st.arena = .{ .buf = .{ .d = &d, .ptr = 0, .len = 0 } };
    st.capacity = boundary + 8192;
    st.ring_rows = keys.len;
    st.ring_row = 1;
    st.rings = &rings;
    st.caches = &.{};
    st.cur = &.{};
    st.pos_dev = @intFromPtr(&pos);
    st.mtp_pos_dev = @intFromPtr(&mtp);
    st.mtp_caches = .{ .lc = 0, .ik = rings[0], .ig = rings[1], .pk = 0, .pk_rows = 0 };
    try st.restoreHost(stream, boundary - 1, boundary - 1, &.{}, .{ 0, base });
    if (resumed) {
        const saved_keys = keys;
        const saved_gates = gates;
        const saved_base = st.mtp_base;
        @memset(&keys, 0);
        @memset(&gates, 0);
        keys = saved_keys;
        gates = saved_gates;
        try st.restoreHost(stream, boundary - 1, boundary - 1, &.{}, .{ 0, saved_base });
    }
    for (0..rewind) |_| {
        try st.readyMtp(stream, 1);
        const dst = st.mtp_caches.ik + st.mtp_len;
        try t.expect(dst >= rings[0] and dst < rings[0] + keys.len);
        const p: *u8 = @ptrFromInt(dst);
        p.* = value(st.mtp_len);
        try st.setMtpLen(stream, st.mtp_len + 1);
    }
    try st.setMtpLen(stream, st.mtp_len - rewind);
    try st.readyMtp(stream, 1);
    try t.expect(st.mtp_len >= st.mtp_base);
    const pool = st.mtp_len / 4 * 4;
    for (pool..st.mtp_len) |i| {
        const k: *const u8 = @ptrFromInt(st.mtp_caches.ik + i);
        const g: *const u8 = @ptrFromInt(st.mtp_caches.ig + i);
        try t.expectEqual(value(i), k.*);
        try t.expectEqual(value(i) ^ 0x55, g.*);
    }
}

test "glm ring: rejected MTP proposals retain their rollback pool at every ring boundary" {
    for ([_]usize{ 4096, 8192, 12288 }) |boundary| for ([_]usize{ 2, 7 }) |rewind| try check(boundary, false, rewind);
}

test "glm ring: restored bases retain the same rollback pool" {
    for ([_]usize{ 4096, 8192, 12288 }) |boundary| for ([_]usize{ 2, 7 }) |rewind| try check(boundary, true, rewind);
}
