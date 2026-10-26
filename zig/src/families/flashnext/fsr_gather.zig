//! The prompt's unchanged gather launches with explicit layer recipes and persistent pool addresses.
const std = @import("std");
const mtl = @import("metal");
const fz = @import("replay.zig");
const Runtime = @import("fsr_runtime.zig").Runtime;

pub fn gather(p: *fz.Prompt, s: *Runtime, bm: usize, x: fz.Buf, w: []const fz.Buf, k: usize, n: usize, pairs: usize, y: fz.Buf) void {
    const part: usize = if (w[0].b.id == s.pool.?.buffers[6].b.id) 6 else if (w[0].b.id == s.pool.?.buffers[3].b.id) 3 else 0;
    const width = if (part == 6) p.fsr_width.down else p.fsr_width.gu;
    const pipe = if (width[0] == 6 and width[1] == 32)
        (if (bm == 128) p.gather128 else if (bm == 64) p.gather64 else p.gather32)
    else
        p.wide.?.of(@intCast(width[0]), @intCast(width[1])).gather[if (bm == 128) 2 else if (bm == 64) 1 else 0];
    p.bind(pipe, &.{ x, w[0], w[1], w[2], p.b.off });
    s.bind(p.r, 27, 28, part);
    const prm = [4]i32{ @intCast(pairs), @intCast(n), @intCast(k), 512 };
    p.r.enc.setBytes(std.mem.asBytes(&prm), 5);
    p.r.enc.setBuffer(y.b, y.off, 6);
    p.r.enc.dispatchThreads(mtl.Size.of(((n + 63) / 64) * 128, pairs / bm + 512, 1), mtl.Size.of(128, 1, 1));
    p.barrier();
}

pub fn gate(p: *fz.Prompt, s: *Runtime, bm: usize, x: fz.Buf, wg: []const fz.Buf, wu: []const fz.Buf, k: usize, n: usize, pairs: usize, y: fz.Buf) void {
    const width = p.fsr_width.gu;
    const pipe = if (width[0] == 6 and width[1] == 32) (if (bm == 64) p.gu64 else p.gu32) else p.wide.?.of(@intCast(width[0]), @intCast(width[1])).gu[if (bm == 64) 1 else 0];
    p.bind(pipe, &.{ x, wg[0], wg[1], wg[2], wu[0], wu[1], wu[2], p.b.off });
    s.bind(p.r, 27, 28, 0);
    const prm = [4]i32{ @intCast(pairs), @intCast(n), @intCast(k), 512 };
    p.r.enc.setBytes(std.mem.asBytes(&prm), 8);
    p.r.enc.setBuffer(y.b, y.off, 9);
    p.r.enc.dispatchThreads(mtl.Size.of(((n + 63) / 64) * 128, pairs / bm + 512, 1), mtl.Size.of(128, 1, 1));
    p.barrier();
}
