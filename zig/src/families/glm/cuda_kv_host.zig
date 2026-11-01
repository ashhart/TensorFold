//! TF_GLM_KV_HOST=1: the DSA layers' latent rows on pinned host pages, each call's picked rows gathered to the device.
const std = @import("std");
const cuda = @import("cuda");
const posix = std.posix;

pub fn onHost() bool {
    return set("TF_GLM_KV_HOST");
}

/// TF_GLM_KV_HOST_SHARED=1: the ranks on this host share one copy of the pages (their latent rows are identical).
pub fn shared() bool {
    return set("TF_GLM_KV_HOST_SHARED");
}

fn set(name: [*:0]const u8) bool {
    const v = std.c.getenv(name) orelse return false;
    return std.mem.eql(u8, std.mem.span(v), "1");
}

/// Host pages of `page` bytes for a `capacity`-row window: `gib` GiB of them, else one full window and 16.
fn pagesOf(gib: ?f64, page: u64, capacity: usize) usize {
    const g = gib orelse return cdiv(capacity, page_rows) + 16;
    return @intFromFloat(g * (1 << 30) / @as(f64, @floatFromInt(page)));
}

/// TF_GLM_KV_HOST_GIB's host pages, else one full window's.
pub fn hostPages(page: u64, capacity: usize) usize {
    const v = std.c.getenv("TF_GLM_KV_HOST_GIB") orelse return pagesOf(null, page, capacity);
    return pagesOf(std.fmt.parseFloat(f64, std.mem.span(v)) catch null, page, capacity);
}

/// The plan's `pages` (counted by their device keys alone), held to the host pages when the pages are in host memory.
pub fn pagesFor(page: u64, capacity: usize, pages: usize) usize {
    if (!onHost()) return pages;
    return @min(pages, hostPages(page, capacity));
}

/// Before the ranks measure their room: the gather's device buffers, so the budget counts them.
pub fn prepare(d: *const cuda.Driver, page: u64, capacity: usize, row_bytes: usize) !void {
    if (!onHost() or active != null) return;
    active = try Gather.init(d, hostPages(page, capacity) * page / row_bytes, capacity, row_bytes);
}

/// The engine's gather, set when its pages go to host memory.
pub var active: ?Gather = null;

/// Rows a KV page holds; the row cache's layers (11 DSA and MTP, with room), rows a layer, and its largest call.
const page_rows = 512;
const layers = 16;
const cache_rows = 8192;
const cached_call = 16;

/// The gather's device buffers (one engine stream runs every call, so one set serves all layers and sequences).
pub const Gather = struct {
    slot_of: cuda.DeviceBuffer, // i32 per arena row: its slot in this call, -1 unless marked
    list: cuda.DeviceBuffer, // i32 [cap]: the call's rows by slot
    slots: cuda.DeviceBuffer, // bf16 [cap][lw]
    n: cuda.DeviceBuffer, // i32: the call's rows
    prev_of: cuda.DeviceBuffer, // i32 per arena row: its slot in the layer's cache, -1 unless marked
    prev_rows: cuda.DeviceBuffer, // i32 [layers][cache_rows]: each layer's last decode call's rows
    prev_n: cuda.DeviceBuffer, // i32 [layers]
    cache: cuda.DeviceBuffer, // bf16 [layers][cache_rows][lw]
    cap: usize,
    row_bytes: usize,
    tags: [layers]u64 = @splat(0),

    pub fn init(d: *const cuda.Driver, arena_rows: usize, cap: usize, row_bytes: usize) !Gather {
        var slot_of = try cuda.DeviceBuffer.alloc(d, arena_rows * 4);
        errdefer slot_of.free();
        var list = try cuda.DeviceBuffer.alloc(d, cap * 4);
        errdefer list.free();
        var slots = try cuda.DeviceBuffer.alloc(d, cap * row_bytes);
        errdefer slots.free();
        var n = try cuda.DeviceBuffer.alloc(d, 16);
        errdefer n.free();
        var prev_of = try cuda.DeviceBuffer.alloc(d, arena_rows * 4);
        errdefer prev_of.free();
        var prev_rows = try cuda.DeviceBuffer.alloc(d, layers * cache_rows * 4);
        errdefer prev_rows.free();
        var prev_n = try cuda.DeviceBuffer.alloc(d, layers * 4);
        errdefer prev_n.free();
        const cache = try cuda.DeviceBuffer.alloc(d, layers * cache_rows * row_bytes);
        try d.check(d.api.cuMemsetD32_v2(slot_of.ptr, 0xFFFF_FFFF, arena_rows), "cuMemsetD32");
        try d.check(d.api.cuMemsetD32_v2(prev_of.ptr, 0xFFFF_FFFF, arena_rows), "cuMemsetD32");
        try d.check(d.api.cuMemsetD32_v2(prev_n.ptr, 0, layers), "cuMemsetD32");
        std.log.info("kv host: {d} gather slots ({d} MiB), {d} MiB of row marks and a {d} MiB row cache on the device", .{ cap, cap * row_bytes >> 20, 2 * arena_rows * 4 >> 20, layers * cache_rows * row_bytes >> 20 });
        return .{ .slot_of = slot_of, .list = list, .slots = slots, .n = n, .prev_of = prev_of, .prev_rows = prev_rows, .prev_n = prev_n, .cache = cache, .cap = cap, .row_bytes = row_bytes };
    }

    pub fn deinit(g: *Gather) void {
        inline for (.{ &g.slot_of, &g.list, &g.slots, &g.n, &g.prev_of, &g.prev_rows, &g.prev_n, &g.cache }) |b| b.free();
    }

    /// Pages changed hands: no layer's cached rows may be used again.
    pub fn invalidate(g: *const Gather, stream: cuda.Stream) !void {
        try stream.d.check(stream.d.api.cuMemsetD32Async(g.prev_n.ptr, 0, layers, stream.handle), "cuMemsetD32Async");
    }

    /// The cache index of the layer `tag` names (numbered on first sight); null past `layers`.
    fn layerOf(g: *Gather, tag: u64) ?usize {
        for (&g.tags, 0..) |*t, i| {
            if (t.* == tag) return i;
            if (t.* == 0) {
                t.* = tag;
                return i;
            }
        }
        return null;
    }

    /// A sparse call's picked rows of `lc` (tokens [rows][width], counts) into the slots; the lists then name slots.
    pub fn gather(g: *Gather, ops: cuda.Module, stream: cuda.Stream, tag: u64, lc: u64, pt: u64, page_rows_at: u64, pos: u64, tokens: u64, counts: u64, rows: usize, width: usize) !u64 {
        const layer = if (rows <= cached_call) g.layerOf(tag) else null; // decode and verify calls use the row cache
        const total = rows * width;
        const upper = @min(total, g.cap);
        const cap: c_int = @intCast(g.cap);
        const rv: c_int = @intCast(g.row_bytes / 16);
        try stream.d.check(stream.d.api.cuMemsetD32Async(g.n.ptr, 0, 1, stream.handle), "cuMemsetD32Async");
        try launch(ops, stream, "tf_glm_kvh_mark", cdiv(total, 256), .{ tokens, counts, @as(c_int, @intCast(width)), @as(i64, @intCast(total)), g.slot_of.ptr, g.list.ptr, g.n.ptr, cap });
        const l = layer orelse {
            try launch(ops, stream, "tf_glm_kvh_copy", cdiv(upper, 8), .{ lc, g.list.ptr, g.n.ptr, g.slots.ptr, rv, cap });
            try launch(ops, stream, "tf_glm_kvh_remap", cdiv(total, 256), .{ tokens, counts, @as(c_int, @intCast(width)), @as(i64, @intCast(total)), g.slot_of.ptr });
            try launch(ops, stream, "tf_glm_kvh_reset", cdiv(upper, 256), .{ g.list.ptr, g.n.ptr, g.slot_of.ptr, cap });
            return g.slots.ptr;
        };
        const prev_rows = g.prev_rows.ptr + l * cache_rows * 4;
        const prev_n = g.prev_n.ptr + l * 4;
        const cache = g.cache.ptr + l * cache_rows * g.row_bytes;
        try launch(ops, stream, "tf_glm_kvh_prev_mark", cdiv(cache_rows, 256), .{ prev_rows, prev_n, g.prev_of.ptr, pt, @as(i64, @intCast(page_rows_at)), pos, @as(c_int, @intCast(rows)) });
        try launch(ops, stream, "tf_glm_kvh_copy_cached", cdiv(upper, 8), .{ lc, g.list.ptr, g.n.ptr, g.slots.ptr, rv, cap, g.prev_of.ptr, cache });
        try launch(ops, stream, "tf_glm_kvh_remap", cdiv(total, 256), .{ tokens, counts, @as(c_int, @intCast(width)), @as(i64, @intCast(total)), g.slot_of.ptr });
        try launch(ops, stream, "tf_glm_kvh_unmark", cdiv(@max(upper, cache_rows), 256), .{ prev_rows, prev_n, g.prev_of.ptr, g.list.ptr, g.n.ptr, g.slot_of.ptr, cap });
        try launch(ops, stream, "tf_glm_kvh_keep", cdiv(cache_rows, 8), .{ g.slots.ptr, g.list.ptr, g.n.ptr, cache, prev_rows, prev_n, rv, @as(c_int, cache_rows) });
        return g.slots.ptr;
    }

    /// A dense call's visible rows 0 .. pos + rows (at most `max_rows`) into the slots, in position order.
    pub fn visible(g: *const Gather, ops: cuda.Module, stream: cuda.Stream, lc: u64, pt: u64, page_rows_at: u64, pos: u64, rows: usize, max_rows: usize) !u64 {
        const n = @min(max_rows, g.cap);
        try launch(ops, stream, "tf_glm_kvh_rows", cdiv(n, 8), .{ lc, pt, @as(i64, @intCast(page_rows_at)), pos, @as(c_int, @intCast(rows)), g.slots.ptr, @as(c_int, @intCast(g.row_bytes / 16)), @as(c_int, @intCast(n)) });
        return g.slots.ptr;
    }
};

/// One glm_ops kernel of 256 threads a block over `blocks` blocks.
fn launch(ops: cuda.Module, stream: cuda.Stream, name: [:0]const u8, blocks: usize, args: anytype) !void {
    var a: cuda.Args = .{};
    inline for (args) |v| a.add(v);
    try cuda.launch.launch(try ops.function(name), .{ .grid = .{ .x = @intCast(blocks) }, .block = .{ .x = 256 } }, stream, &a);
}

/// The shared pages' file for an NCCL id (`host:port`), the same name on every rank.
fn shmName(buf: *[96:0]u8, id: []const u8) ![]u8 {
    buf.* = @splat(0);
    const name = std.fmt.bufPrint(buf, "/dev/shm/tf-glm-kv-{s}", .{id}) catch return error.SharedMemoryUnavailable;
    for (name["/dev/shm/".len..]) |*c| if (c.* == ':' or c.* == '/') {
        c.* = '-';
    };
    return name;
}

/// The shared file's tail past the pages: a count of the ranks that have mapped it.
const mark_bytes = 4096;

/// Counts this rank's mapping; true for the last of `world` ranks, which takes the file's name away.
fn lastToMap(count: *u32, world: usize) bool {
    return @atomicRmw(u32, count, .Add, 1, .seq_cst) + 1 == world;
}

/// The pages as one /dev/shm file every rank maps: rank 0 makes it, the others wait for it at its full length.
pub const SharedPages = struct {
    region: []align(std.heap.page_size_min) u8,
    dev: u64,
    lib: std.DynLib,
    unregister: *const fn (?*anyopaque) callconv(.c) cuda.abi.Result,
    leader: bool,
    world: usize, // ranks that map the file; the last to map it unlinks it, so nothing outlives the ranks
    path: [96:0]u8,

    pub fn open(d: *const cuda.Driver, len: usize, leader: bool, world: usize) !SharedPages {
        const file_len = len + mark_bytes;
        var path: [96:0]u8 = undefined;
        const name = try shmName(&path, if (std.c.getenv("NCCL_COMM_ID")) |v| std.mem.span(v) else "local");
        const flags: std.c.O = if (leader) .{ .ACCMODE = .RDWR, .CREAT = true, .TRUNC = true, .CLOEXEC = true } else .{ .ACCMODE = .RDWR, .CLOEXEC = true };
        var fd: c_int = -1;
        var tries: usize = 0;
        while (true) : (tries += 1) { // the others may get rank 0's pool before it has made the file
            fd = std.c.open(&path, flags, @as(std.c.mode_t, 0o600));
            if (fd >= 0 and leader) break;
            if (fd >= 0 and std.c.lseek(fd, 0, std.c.SEEK.END) == @as(std.c.off_t, @intCast(file_len))) break;
            if (fd >= 0) _ = std.c.close(fd);
            if (tries == 600) return error.SharedMemoryUnavailable;
            const ts: std.c.timespec = .{ .sec = 0, .nsec = 50_000_000 };
            _ = std.c.nanosleep(&ts, null);
        }
        defer _ = std.c.close(fd);
        if (leader and std.c.ftruncate(fd, @intCast(file_len)) != 0) return error.SharedMemoryUnavailable;
        const region = try posix.mmap(null, file_len, .{ .READ = true, .WRITE = true }, .{ .TYPE = .SHARED }, fd, 0);
        errdefer posix.munmap(region);
        var lib = std.DynLib.open("libcuda.so.1") catch return error.DriverUnavailable;
        errdefer lib.close();
        const register = lib.lookup(*const fn (?*anyopaque, usize, c_uint) callconv(.c) cuda.abi.Result, "cuMemHostRegister_v2") orelse return error.MissingSymbol;
        const unregister = lib.lookup(*const fn (?*anyopaque) callconv(.c) cuda.abi.Result, "cuMemHostUnregister") orelse return error.MissingSymbol;
        try d.check(register(region.ptr, len, 1 | 2), "cuMemHostRegister"); // PORTABLE | DEVICEMAP
        errdefer _ = unregister(region.ptr);
        var dev: u64 = 0;
        try d.check(d.api.cuMemHostGetDevicePointer_v2(&dev, region.ptr, 0), "cuMemHostGetDevicePointer");
        const last = lastToMap(@ptrCast(@alignCast(region.ptr + len)), world);
        if (last) _ = std.c.unlink(&path);
        std.log.info("kv host: pages shared through {s} ({d} MiB, {s}{s})", .{ name, len >> 20, if (leader) "made" else "mapped", if (last) ", every rank mapped: unlinked" else "" });
        return .{ .region = region, .dev = dev, .lib = lib, .unregister = unregister, .leader = leader, .world = world, .path = path };
    }

    pub fn close(s: *SharedPages) void {
        const count: *u32 = @ptrCast(@alignCast(s.region.ptr + s.region.len - mark_bytes));
        const named = s.leader and @atomicLoad(u32, count, .seq_cst) < s.world; // a rank never mapped it: still named
        _ = s.unregister(s.region.ptr);
        posix.munmap(s.region);
        if (named) _ = std.c.unlink(&s.path);
        s.lib.close();
    }
};

fn cdiv(a: usize, b: usize) usize {
    return (a + b - 1) / b;
}

test "kv host pages: one full window and 16 by default, TF_GLM_KV_HOST_GIB's worth when set" {
    const page: u64 = 13_056 * page_rows; // a GLM-5.3-Flash page: 12 DSA and MTP caches' rows
    try std.testing.expectEqual(@as(usize, 2048 + 16), pagesOf(null, page, 1 << 20));
    try std.testing.expectEqual(@as(usize, 1 + 16), pagesOf(null, page, 300));
    try std.testing.expectEqual(@as(usize, 2088), pagesOf(13, page, 1 << 20));
    try std.testing.expectEqual(@as(usize, 0), pagesOf(0, page, 1 << 20));
}

test "kv host shared pages: the last of the ranks to map the file takes its name away" {
    var count: u32 = 0;
    try std.testing.expect(!lastToMap(&count, 2));
    try std.testing.expect(lastToMap(&count, 2));
    count = 0;
    try std.testing.expect(lastToMap(&count, 1));
    count = 0;
    for (0..2) |_| try std.testing.expect(!lastToMap(&count, 3));
    try std.testing.expect(lastToMap(&count, 3));
}

test "kv host shared pages: one file name per NCCL id, with no separators past /dev/shm" {
    var buf: [96:0]u8 = undefined;
    try std.testing.expectEqualStrings("/dev/shm/tf-glm-kv-127.0.0.1-29583", try shmName(&buf, "127.0.0.1:29583"));
    try std.testing.expectEqualStrings("/dev/shm/tf-glm-kv-a-b-1", try shmName(&buf, "a/b:1"));
    const long: [90]u8 = @splat('x');
    try std.testing.expectError(error.SharedMemoryUnavailable, shmName(&buf, &long));
}
