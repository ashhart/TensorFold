//! GLM-5.3-Flash weights as they load: parallel reads into place, and rewrites (mixes, router, conv, norms, columns).
const std = @import("std");
const st = @import("../../core/safetensors.zig");
const paced_read = @import("../../core/paced_read.zig");

/// One tensor's bytes on disk.
pub const Src = struct { shard: u32, off: u64, len: u64, dtype: st.DType, shape: [st.max_rank]usize, rank: u8 };
pub const Fix = enum { hc_pack, router_pack, conv, to_f32, transpose_bf16, cols };
/// A tensor read whole into host memory and rewritten into its place (small: mixes, router, conv taps, norms).
pub const Transform = struct { src: Src, dst: [*]u8, fix: Fix, extra: [3]?Src = .{ null, null, null }, arg: [3]usize = .{ 0, 0, 0 } };

/// One tensor's bytes read straight into their place in a weights buffer.
pub const Copy = struct { shard: u32, off: u64, len: u64, dst: [*]u8 };

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
    const narrow = t.src.dtype == .f32 and (t.fix == .hc_pack or t.fix == .router_pack); // mlx-lm's fp32 mixes, router
    if (narrow) bf16Exact(raw) catch |err| {
        @import("quant.zig").say("glm: an fp32 {t} tensor (shard {d}, at byte {d}, shape {any}) is not exact in bf16", .{ t.fix, t.src.shard, t.src.off, t.src.shape[0..t.src.rank] });
        return err;
    };
    const in16: []const u16 = @alignCast(std.mem.bytesAsSlice(u16, raw[0..if (narrow) raw.len / 2 else raw.len]));
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

/// fp32 values as bf16 in place (the first half of `raw`), refused unless each is a bf16 (0.6.6's bf16_if_exact).
fn bf16Exact(raw: []u8) error{InexactBf16}!void {
    const n = raw.len / 4;
    for (0..n) |i| {
        const v = std.mem.readInt(u32, raw[4 * i ..][0..4], .little);
        if (v & 0xffff != 0) return error.InexactBf16;
        std.mem.writeInt(u16, raw[2 * i ..][0..2], @truncate(v >> 16), .little);
    }
}

test "fp32 values that are bf16 narrow in place; any other value is refused" {
    var raw: [12]u8 = undefined;
    for ([_]f32{ 1.5, -0.0078125, 3.0 }, 0..) |v, i| std.mem.writeInt(u32, raw[4 * i ..][0..4], @bitCast(v), .little);
    try bf16Exact(&raw);
    try std.testing.expectEqual(@as(u16, 0x3fc0), std.mem.readInt(u16, raw[0..2], .little));
    try std.testing.expectEqual(@as(u16, 0x4040), std.mem.readInt(u16, raw[4..6], .little));
    std.mem.writeInt(u32, raw[0..4], @bitCast(@as(f32, 1.0000001)), .little);
    try std.testing.expectError(error.InexactBf16, bf16Exact(&raw));
}

const Pool = struct {
    fds: []std.c.fd_t,
    jobs: []const Copy,
    next: std.atomic.Value(usize) = .init(0),
    failed: std.atomic.Value(bool) = .init(false),

    fn run(p: *Pool) void {
        const buf = paced_read.stage(std.heap.page_allocator, 16 << 20) catch {
            p.failed.store(true, .release);
            return;
        };
        defer std.heap.page_allocator.free(buf);
        while (true) {
            const i = p.next.fetchAdd(1, .monotonic);
            if (i >= p.jobs.len) return;
            const j = p.jobs[i];
            paced_read.read(p.fds[j.shard], j.dst[0..j.len], j.off, buf) catch p.failed.store(true, .release);
        }
    }
};

/// Every copy on `threads` threads, uncached, paced by free memory (a large load otherwise fills the page cache too).
pub fn runCopies(gpa: std.mem.Allocator, shards: []const [:0]const u8, copies: []const Copy, transforms: []const Transform, threads: usize) !void {
    const fds = try gpa.alloc(std.c.fd_t, shards.len);
    defer gpa.free(fds);
    for (shards, fds) |path, *fd| {
        fd.* = std.c.open(path, .{ .ACCMODE = .RDONLY });
        if (fd.* < 0) return error.OpenFailed;
        _ = std.c.fcntl(fd.*, 48, @as(c_int, 1)); // F_NOCACHE
    }
    defer for (fds) |fd| {
        _ = std.c.close(fd);
    };
    var pool: Pool = .{ .fds = fds, .jobs = copies };
    const workers = try gpa.alloc(?std.Thread, threads);
    defer gpa.free(workers);
    for (workers) |*t| t.* = std.Thread.spawn(.{}, Pool.run, .{&pool}) catch null;
    pool.run();
    for (workers) |t| if (t) |th| th.join();
    if (pool.failed.load(.acquire)) return error.ShortRead;
    // the transforms on the same threads (each reads its own tensors whole)
    const Tx = struct {
        gpa: std.mem.Allocator,
        jobs: []const Transform,
        fds: []std.c.fd_t,
        next: std.atomic.Value(usize) = .init(0),
        failed: std.atomic.Value(bool) = .init(false),
        fn run(x: *@This()) void {
            while (true) {
                const i = x.next.fetchAdd(1, .monotonic);
                if (i >= x.jobs.len) return;
                transform(x.gpa, x.fds, x.jobs[i]) catch x.failed.store(true, .release);
            }
        }
    };
    var tx: Tx = .{ .gpa = gpa, .jobs = transforms, .fds = fds };
    for (workers) |*t| t.* = std.Thread.spawn(.{}, Tx.run, .{&tx}) catch null;
    tx.run();
    for (workers) |t| if (t) |th| th.join();
    if (tx.failed.load(.acquire)) return error.TransformFailed;
}
