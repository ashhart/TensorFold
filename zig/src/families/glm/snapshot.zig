//! A slot's prompt state at a chunk end: every KDA state and conv window, every MLA cache's prefix.
const std = @import("std");
const mtl = @import("metal");
const cfg = @import("config.zig");
const st = @import("state.zig");
const fwd = @import("forward.zig");
const Ref = @import("weights.zig").Ref;

/// One kept state: `at` prompt tokens of one stream, in one buffer (an id both Macs of a pair name it by).
pub const Snap = struct { id: u32, at: u32, buf: mtl.Buffer, bytes: usize };

fn stBytes(c: *const cfg.Config) usize {
    return @as(usize, c.kda_heads) * c.kda_dim * c.kda_dim * 4;
}

fn csBytes(c: *const cfg.Config) usize {
    return @as(usize, c.conv - 1) * 3 * c.kdaWidth() * 2;
}

fn mlaCount(c: *const cfg.Config) usize {
    return c.countKind(.mla) + @as(usize, if (c.mtp > 0) 1 else 0);
}

/// The pooled index blocks `at` tokens hold (a partial last block is copied, never read until whole).
fn blocks(c: *const cfg.Config, at: u32) usize {
    return at / c.kpool + 1;
}

/// The bytes a state at `at` tokens takes.
pub fn bytes(c: *const cfg.Config, at: u32) usize {
    const mla = @as(usize, at) * (c.kv_lora + 2 * c.i_dim) * 2 + blocks(c, at) * c.i_dim * 2;
    return c.countKind(.kda) * (stBytes(c) + csBytes(c)) + mlaCount(c) * mla;
}

/// `n` bytes (a multiple of 4) from `src` to `dst` on the GPU.
fn words(x: *const fwd.Ctx, e: mtl.ComputeEncoder, src: Ref, dst: Ref, n: usize) void {
    e.setPipeline(x.k.copy_u32);
    e.setBuffer(src.buf, src.off, 0);
    e.setBuffer(dst.buf, dst.off, 1);
    e.setValue(@as(u32, @intCast(n / 4)), 2);
    e.dispatchThreads(mtl.Size.of(n / 4, 1, 1), mtl.Size.of(256, 1, 1));
}

/// Encode the copies between `s` and the snapshot at `snap` (`into`: the state into the snapshot, else back).
pub fn copy(x: *const fwd.Ctx, e: mtl.ComputeEncoder, s: *st.State, snap: Ref, at: u32, into: bool) void {
    const c = x.c;
    var off: usize = 0;
    for (s.kda[0..c.countKind(.kda)]) |*L| { // the state the next chunk reads: the current slot's, restored into slot 0
        const live = [2]Ref{ L.st[if (into) L.cur else 0], L.cs[if (into) L.cur else 0] };
        for (live, [2]usize{ stBytes(c), csBytes(c) }) |r, n| {
            if (into) words(x, e, r, snap.at(off), n) else words(x, e, snap.at(off), r, n);
            off += n;
        }
    }
    for (s.mla[0..mlaCount(c)]) |*C| {
        const parts = [4]Ref{ C.keys, C.ik, C.ig, C.pool };
        const sizes = [4]usize{ @as(usize, at) * c.kv_lora * 2, @as(usize, at) * c.i_dim * 2, @as(usize, at) * c.i_dim * 2, blocks(c, at) * c.i_dim * 2 };
        for (parts, sizes) |r, n| {
            if (into) words(x, e, r, snap.at(off), n) else words(x, e, snap.at(off), r, n);
            off += n;
        }
    }
    std.debug.assert(off == bytes(c, at));
    if (into) return;
    for (s.kda[0..c.countKind(.kda)]) |*L| L.cur = 0;
    s.pos = at;
    s.mtp_pos = at;
}

const FILE_MAGIC: u32 = 0x474c4d53; // "GLMS"

/// A learned state's file for one rank of the pair: <dir>/<key>.r<rank>.bin.
pub fn path(buf: []u8, dir: []const u8, key: u64, rank: u32) ![:0]const u8 {
    return std.fmt.bufPrintSentinel(buf, "{s}/{x:0>16}.r{d}.bin", .{ dir, key, rank }, 0);
}

/// `snap` to `file` (its position and size, then its bytes), through a temporary file renamed into place.
pub fn writeFile(snap: *const Snap, file: [:0]const u8) !void {
    var tmp_buf: [1100]u8 = undefined;
    const tmp = try std.fmt.bufPrintSentinel(&tmp_buf, "{s}.part", .{file}, 0);
    const fd = std.c.open(tmp, .{ .ACCMODE = .WRONLY, .CREAT = true, .TRUNC = true }, @as(std.c.mode_t, 0o600));
    if (fd < 0) return error.SnapshotWrite;
    errdefer _ = std.c.unlink(tmp);
    {
        defer _ = std.c.close(fd);
        const head = [4]u32{ FILE_MAGIC, snap.at, @truncate(snap.bytes), @truncate(snap.bytes >> 32) };
        try put(fd, std.mem.sliceAsBytes(&head));
        try put(fd, snap.buf.contents()[0..snap.bytes]);
        if (std.c.fsync(fd) != 0) return error.SnapshotWrite;
    }
    if (std.c.rename(tmp, file) != 0) return error.SnapshotWrite;
}

/// `file`'s state into `snap`, whose buffer holds `bytes(c, snap.at)`; a file of another position or size is refused.
pub fn readFile(snap: *Snap, file: [:0]const u8) !void {
    const fd = std.c.open(file, .{ .ACCMODE = .RDONLY }, @as(std.c.mode_t, 0));
    if (fd < 0) return error.SnapshotRead;
    defer _ = std.c.close(fd);
    var head: [4]u32 = undefined;
    try get(fd, std.mem.sliceAsBytes(&head), 0);
    if (head[0] != FILE_MAGIC or head[1] != snap.at or (@as(u64, head[2]) | @as(u64, head[3]) << 32) != snap.bytes) return error.SnapshotRead;
    try get(fd, snap.buf.contents()[0..snap.bytes], @sizeOf(@TypeOf(head)));
}

fn put(fd: c_int, b: []const u8) !void {
    var done: usize = 0;
    while (done < b.len) {
        const n = std.c.write(fd, b.ptr + done, @min(b.len - done, 1 << 30));
        if (n <= 0) return error.SnapshotWrite;
        done += @intCast(n);
    }
}

fn get(fd: c_int, b: []u8, at: u64) !void {
    var done: usize = 0;
    while (done < b.len) {
        const n = std.c.pread(fd, b.ptr + done, @min(b.len - done, 1 << 30), @intCast(at + done));
        if (n <= 0) return error.SnapshotRead;
        done += @intCast(n);
    }
}
