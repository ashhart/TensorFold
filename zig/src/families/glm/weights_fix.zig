//! GLM-5.3-Flash weights rewritten as they load: packed mixes and router, conv taps, fp32 norms, column slices.
const std = @import("std");
const st = @import("../../core/safetensors.zig");

/// One tensor's bytes on disk.
pub const Src = struct { shard: u32, off: u64, len: u64, dtype: st.DType, shape: [st.max_rank]usize, rank: u8 };
pub const Fix = enum { hc_pack, router_pack, conv, to_f32, transpose_bf16, cols };
/// A tensor read whole into host memory and rewritten into its place (small: mixes, router, conv taps, norms).
pub const Transform = struct { src: Src, dst: [*]u8, fix: Fix, extra: [3]?Src = .{ null, null, null }, arg: [3]usize = .{ 0, 0, 0 } };

pub fn readAll(fd: std.c.fd_t, dest: []u8, at: u64) !void {
    var done: usize = 0;
    while (done < dest.len) {
        const n = std.c.pread(fd, dest.ptr + done, dest.len - done, @intCast(at + done));
        if (n <= 0) return error.ShortRead;
        done += @intCast(n);
    }
}

pub fn transform(gpa: std.mem.Allocator, fds: []std.c.fd_t, t: Transform) !void {
    const raw = try gpa.alloc(u8, t.src.len);
    defer gpa.free(raw);
    try readAll(fds[t.src.shard], raw, t.src.off);
    const in16: []const u16 = @alignCast(std.mem.bytesAsSlice(u16, raw));
    switch (t.fix) {
        .hc_pack => { // [og 6][tm 4][i 16][sgn 8][lane 32][tn 4] -> [og][sgn][lane][i][tm][tn]
            const out: [*]u16 = @ptrCast(@alignCast(t.dst));
            var o: usize = 0;
            for (0..6) |og| for (0..8) |sgn| for (0..32) |lane| for (0..16) |i| for (0..4) |tm| for (0..4) |tn| {
                out[o] = in16[(og * 4 + tm) * 16384 + i * 1024 + sgn * 128 + lane * 4 + tn];
                o += 1;
            };
        },
        .router_pack => { // packed[q][thrM][i][tm][tn] = router[4q + tn][32i + 4thrM + tm]
            const out: [*]u16 = @ptrCast(@alignCast(t.dst));
            const E = t.src.shape[0];
            const K = t.src.shape[1];
            var o: usize = 0;
            for (0..E / 4) |q| for (0..8) |thr| for (0..K / 32) |i| for (0..4) |tm| for (0..4) |tn| {
                out[o] = in16[(4 * q + tn) * K + 32 * i + 4 * thr + tm];
                o += 1;
            };
        },
        .conv => { // [q | k | v][channel][tap] bf16 -> [tap][3 * channels] fp32, channels [arg 0, arg 0 + arg 1) in TP2
            const out: [*]f32 = @ptrCast(@alignCast(t.dst));
            const C = if (t.arg[1] != 0) t.arg[1] else t.src.shape[0];
            const c0 = t.arg[0];
            const T = t.src.shape[2];
            const parts = [_]?Src{ t.src, t.extra[0], t.extra[1] };
            for (parts, 0..) |ps, part| {
                const p = ps orelse return error.MissingTensor;
                const buf = if (part == 0) raw else blk: {
                    const b = try gpa.alloc(u8, p.len);
                    try readAll(fds[p.shard], b, p.off);
                    break :blk b;
                };
                defer if (part != 0) gpa.free(buf);
                const v: []const u16 = @alignCast(std.mem.bytesAsSlice(u16, buf));
                for (0..C) |ch| for (0..T) |tap| {
                    out[tap * 3 * C + part * C + ch] = @bitCast(@as(u32, v[(c0 + ch) * T + tap]) << 16);
                };
            }
        },
        .to_f32 => {
            const out: [*]f32 = @ptrCast(@alignCast(t.dst));
            for (in16, 0..) |x, i| out[i] = @bitCast(@as(u32, x) << 16);
        },
        .cols => { // each stored row's bytes [arg 1, arg 1 + arg 2) of its arg 0
            const rows = t.src.len / t.arg[0];
            for (0..rows) |rr| @memcpy(t.dst[rr * t.arg[2] ..][0..t.arg[2]], raw[rr * t.arg[0] + t.arg[1] ..][0..t.arg[2]]);
        },
        .transpose_bf16 => { // [r][c] -> [c][r]
            const out: [*]u16 = @ptrCast(@alignCast(t.dst));
            const R = t.src.shape[0];
            const C = t.src.shape[1];
            for (0..R) |r| for (0..C) |cc| {
                out[cc * R + r] = in16[r * C + cc];
            };
        },
    }
}
