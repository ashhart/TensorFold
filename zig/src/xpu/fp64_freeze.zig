//! Frozen GEMM input capture/replay for FP64 probes (XPU_FP64_SAVE_X / XPU_FP64_LOAD_X directories).
const std = @import("std");
const q = @import("qwen27.zig");
const fwd = @import("forward.zig");
const Dev = q.Dev;
const alloc = q.alloc;
extern "c" fn open(path: [*:0]const u8, flags: c_int, ...) c_int;
extern "c" fn close(fd: c_int) c_int;
extern "c" fn lseek(fd: c_int, off: i64, whence: c_int) i64;
extern "c" fn pread(fd: c_int, buf: [*]u8, n: usize, off: i64) isize;
extern "c" fn write(fd: c_int, buf: [*]const u8, n: usize) isize;

const MAGIC: u64 = 0x46505a495f463634; // FPZI_F64
const HEADER: usize = 32;
const O_RDONLY: c_int = 0;
const O_WRONLY_CREAT_TRUNC: c_int = 1 | 64 | 512;

pub fn corrBytes() usize {
    const sh = q.tpShape();
    const h_cols: usize = @max(@as(usize, q.D), @max(@as(usize, sh.ff), @as(usize, sh.od)));
    return @as(usize, q.maxRows()) * fwd.gStride(h_cols / 64) * 2;
}

fn pathOf(dir: []const u8, layer: usize, site: u32) ![:0]u8 {
    return std.fmt.allocPrintSentinel(alloc, "{s}/fp64_l{d}_s{d}.bin", .{ dir, layer, site }, 0);
}

fn writeAll(fd: c_int, bytes: []const u8) !void {
    var at: usize = 0;
    while (at < bytes.len) {
        const n = write(fd, bytes.ptr + at, bytes.len - at);
        if (n <= 0) return error.FrozenInputWriteFailed;
        at += @intCast(n);
    }
}

pub const Replay = struct {
    x: *anyopaque,
    hi: *anyopaque,
    lo: *anyopaque,
    rows: usize,
    k: usize,

    pub fn deinit(r: Replay, d: *Dev) void {
        d.g.free(r.x);
        d.g.free(r.hi);
        d.g.free(r.lo);
    }
};

fn readFile(path: [:0]u8) ![]u8 {
    const fd = open(path.ptr, O_RDONLY);
    if (fd < 0) return error.FrozenInputMissing;
    defer _ = close(fd);
    const len: usize = @intCast(lseek(fd, 0, 2));
    const bytes = try alloc.alloc(u8, len);
    errdefer alloc.free(bytes);
    var at: usize = 0;
    while (at < len) {
        const n = pread(fd, bytes.ptr + at, len - at, @intCast(at));
        if (n <= 0) return error.FrozenInputReadFailed;
        at += @intCast(n);
    }
    return bytes;
}

fn save(d: *Dev, a: *fwd.Acts, x: *anyopaque, T: usize, K: usize, layer: usize, site: u32, dir: []const u8) !void {
    const xb = T * K * 2;
    const cb = corrBytes();
    const bytes = try alloc.alloc(u8, HEADER + xb + 2 * cb);
    defer alloc.free(bytes);
    std.mem.writeInt(u64, bytes[0..8], MAGIC, .little);
    std.mem.writeInt(u32, bytes[8..12], @intCast(T), .little);
    std.mem.writeInt(u32, bytes[12..16], @intCast(K), .little);
    std.mem.writeInt(u64, bytes[16..24], @intCast(cb), .little);
    std.mem.writeInt(u64, bytes[24..32], 0, .little);
    try q.ze.check(d.g.api.zeCommandListAppendMemoryCopy(d.list, bytes[HEADER..].ptr, x, xb, null, 0, null));
    try q.ze.check(d.g.api.zeCommandListAppendMemoryCopy(d.list, bytes[HEADER + xb ..].ptr, a.ahi, cb, null, 0, null));
    try q.ze.check(d.g.api.zeCommandListAppendMemoryCopy(d.list, bytes[HEADER + xb + cb ..].ptr, a.alo, cb, null, 0, null));
    try d.sync();
    const path = try pathOf(dir, layer, site);
    defer alloc.free(path);
    const fd = open(path.ptr, O_WRONLY_CREAT_TRUNC, @as(c_int, 0o644));
    if (fd < 0) return error.FrozenInputWriteFailed;
    defer _ = close(fd);
    try writeAll(fd, bytes);
    std.debug.print("fp64 ref: froze layer {d} site {d} GEMM input ({d} bytes)\n", .{ layer, site, bytes.len });
}

/// Save this probe input, or return device-resident replay buffers and overwrite xa with the frozen fp16 rows.
pub fn prepare(d: *Dev, a: *fwd.Acts, x: *anyopaque, T: usize, K: usize, layer: usize, site: u32, xa: []f16) !?Replay {
    const save_dir = std.c.getenv("XPU_FP64_SAVE_X");
    const load_dir = std.c.getenv("XPU_FP64_LOAD_X");
    if (save_dir != null and load_dir != null) return error.FrozenInputModeConflict;
    if (save_dir) |p| {
        try save(d, a, x, T, K, layer, site, std.mem.span(p));
        return null;
    }
    const p = load_dir orelse return null;
    const path = try pathOf(std.mem.span(p), layer, site);
    defer alloc.free(path);
    const bytes = try readFile(path);
    defer alloc.free(bytes);
    if (bytes.len < HEADER or std.mem.readInt(u64, bytes[0..8], .little) != MAGIC) return error.BadFrozenInput;
    const ft = std.mem.readInt(u32, bytes[8..12], .little);
    const fk = std.mem.readInt(u32, bytes[12..16], .little);
    const cb: usize = @intCast(std.mem.readInt(u64, bytes[16..24], .little));
    const xb = T * K * 2;
    if (ft != T or fk != K or cb != corrBytes() or bytes.len != HEADER + xb + 2 * cb) return error.BadFrozenInput;
    @memcpy(std.mem.sliceAsBytes(xa)[0..xb], bytes[HEADER..][0..xb]);
    const xd = try d.g.deviceAlloc(xb);
    errdefer d.g.free(xd);
    const hi = try d.g.deviceAlloc(cb);
    errdefer d.g.free(hi);
    const lo = try d.g.deviceAlloc(cb);
    errdefer d.g.free(lo);
    try d.copyIn(xd, bytes[HEADER..][0..xb]);
    try d.copyIn(hi, bytes[HEADER + xb ..][0..cb]);
    try d.copyIn(lo, bytes[HEADER + xb + cb ..][0..cb]);
    std.debug.print("fp64 ref: replaying frozen layer {d} site {d} GEMM input\n", .{ layer, site });
    return .{ .x = xd, .hi = hi, .lo = lo, .rows = T, .k = K };
}
