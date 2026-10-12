//! The mailbox backend: two-rank exchanges in one kernel through pinned host memory both GPUs map (host side).
const std = @import("std");
const log = @import("log.zig");
const cuda = @import("cuda");
const collective = @import("collective.zig");
const Bootstrap = @import("bootstrap.zig").Bootstrap;
const sock = @import("sock.zig");
const posix = std.posix;
const DevicePtr = collective.DevicePtr;
const DType = collective.DType;
const Error = collective.Error;

pub const max_blocks = 16;
const threads = 512;
const flag_bytes = 2 * max_blocks * 128;
const ctrl_bytes = 4096;
const ctrl_abort = 0;
const ctrl_rank = 16;

const Mode = enum(u32) { gather = 0, sum_f32 = 1, sum_bf16 = 2, exchange = 3 };

/// mailbox.cu's `MailArgs`, field for field.
const MailArgs = extern struct {
    in: DevicePtr,
    out: DevicePtr,
    peer_slots: DevicePtr,
    peer_flags: DevicePtr,
    my_slots: DevicePtr,
    my_flags: DevicePtr,
    state: DevicePtr,
    ctrl: DevicePtr,
    nbytes: u64,
    slot_bytes: u64,
    timeout_ns: u64,
    rank: u32,
    mode: u32,
};

/// Driver calls the runtime's table does not carry (page-locking memory this process mapped itself).
pub const HostRegister = struct {
    lib: std.DynLib,
    register: *const fn (?*anyopaque, usize, c_uint) callconv(.c) cuda.abi.Result,
    unregister: *const fn (?*anyopaque) callconv(.c) cuda.abi.Result,

    pub fn open() !HostRegister {
        var lib = std.DynLib.open("libcuda.so.1") catch return error.DriverUnavailable;
        errdefer lib.close();
        return .{
            .register = lib.lookup(@FieldType(HostRegister, "register"), "cuMemHostRegister_v2") orelse return error.MissingSymbol,
            .unregister = lib.lookup(@FieldType(HostRegister, "unregister"), "cuMemHostUnregister") orelse return error.MissingSymbol,
            .lib = lib,
        };
    }
};

/// Grid for `n` bytes a rank: one block per 16 KiB, a power of two, at most `max_blocks` (both ranks pick the same).
pub fn blocksFor(n: usize) u32 {
    const want = std.math.divCeil(usize, @max(n, 1), 16 * 1024) catch 1;
    return @intCast(@min(std.math.ceilPowerOfTwoAssert(usize, want), max_blocks));
}

pub const Mailbox = struct {
    d: *const cuda.Driver,
    me: u32,
    slot_bytes: usize,
    timeout_ns: u64,
    region: []align(std.heap.page_size_min) u8,
    dev: DevicePtr,
    hr: HostRegister,
    state: cuda.DeviceBuffer,
    module: cuda.Module,
    func: cuda.Function,
    aborted: std.atomic.Value(bool) = .init(false),

    /// Both ranks call it after the bootstrap (two ranks, one host); `image` is mailbox.cu's fatbin.
    pub fn init(d: *const cuda.Driver, b: *Bootstrap, image: []const u8, max_bytes: usize, timeout_ns: u64) !Mailbox {
        if (b.world != 2) return error.Unsupported;
        if (image.len == 0) return error.NoKernel;
        const slot = std.mem.alignForward(usize, max_bytes, 4096);
        const len = ctrl_bytes + 2 * (flag_bytes + 2 * slot);
        // rank 0 creates and sizes the file before anyone learns its name, the others open it, then rank 0 unlinks it
        var name: [64]u8 = undefined;
        var name_len: u8 = 0;
        var fd: std.c.fd_t = -1;
        defer if (fd >= 0) {
            _ = std.c.close(fd);
        };
        if (b.rank == 0) blk: {
            const s = std.fmt.bufPrint(&name, "/dev/shm/tf-tp-{d}-{x}", .{ std.c.getpid(), sock.nowNs() }) catch break :blk;
            name[s.len] = 0;
            fd = std.c.open(name[0..s.len :0], .{ .ACCMODE = .RDWR, .CREAT = true, .EXCL = true, .CLOEXEC = true }, @as(std.c.mode_t, 0o600));
            if (fd >= 0 and std.c.ftruncate(fd, @intCast(len)) == 0) name_len = @intCast(s.len);
        }
        // an empty name tells the other rank that rank 0 failed (it must not wait for a name that never comes)
        try b.broadcast(std.mem.asBytes(&name_len), std.mem.asBytes(&name_len));
        if (name_len == 0) return error.SharedMemoryUnavailable;
        try b.broadcast(name[0..name_len], name[0..name_len]);
        name[name_len] = 0;
        const path = name[0..name_len :0];
        if (b.rank != 0) {
            fd = std.c.open(path, .{ .ACCMODE = .RDWR, .CLOEXEC = true }, @as(std.c.mode_t, 0));
            if (fd < 0) return error.SharedMemoryUnavailable;
        }
        try b.agree("mapped", "mailbox file");
        const region = try posix.mmap(null, len, .{ .READ = true, .WRITE = true }, .{ .TYPE = .SHARED }, fd, 0);
        errdefer posix.munmap(region);
        try b.agree("open", "mailbox mapping");
        if (b.rank == 0) _ = std.c.unlink(path);
        var hr = try HostRegister.open();
        errdefer hr.lib.close();
        try d.check(hr.register(region.ptr, len, 1 | 2), "cuMemHostRegister"); // PORTABLE | DEVICEMAP
        errdefer _ = hr.unregister(region.ptr);
        var dev: DevicePtr = 0;
        try d.check(d.api.cuMemHostGetDevicePointer_v2(&dev, region.ptr, 0), "cuMemHostGetDevicePointer");
        var state = try cuda.DeviceBuffer.alloc(d, 16);
        errdefer state.free();
        try state.fill8(0, null);
        var module = try cuda.Module.load(d, image);
        errdefer module.unload();
        const func = try module.function("tp_mailbox");
        try b.agree("ready", "mailbox setup");
        return .{ .d = d, .me = b.rank, .slot_bytes = slot, .timeout_ns = timeout_ns, .region = region, .dev = dev, .hr = hr, .state = state, .module = module, .func = func };
    }

    pub fn deinit(m: *Mailbox) void {
        m.module.unload();
        m.state.free();
        _ = m.hr.unregister(m.region.ptr);
        m.hr.lib.close();
        posix.munmap(m.region);
    }

    fn inbox(m: *const Mailbox, r: u32) usize {
        return ctrl_bytes + r * (flag_bytes + 2 * m.slot_bytes);
    }

    fn ctrlWord(m: *const Mailbox, i: usize) *std.atomic.Value(u32) {
        return @ptrCast(@alignCast(m.region[4 * i ..].ptr));
    }

    /// Whether `n` bytes a rank fit one slot.
    pub fn fits(m: *const Mailbox, n: usize) bool {
        return n <= m.slot_bytes;
    }

    fn launch(m: *Mailbox, mode: Mode, in: DevicePtr, out: DevicePtr, n: usize, stream: collective.Stream) Error!void {
        if (m.aborted.load(.acquire)) return error.Aborted;
        if (n > m.slot_bytes) return error.Unsupported;
        if (n == 0) return;
        const peer = 1 - m.me;
        var a: MailArgs = .{
            .in = in,
            .out = out,
            .peer_slots = m.dev + m.inbox(peer) + flag_bytes,
            .peer_flags = m.dev + m.inbox(peer),
            .my_slots = m.dev + m.inbox(m.me) + flag_bytes,
            .my_flags = m.dev + m.inbox(m.me),
            .state = m.state.ptr,
            .ctrl = m.dev,
            .nbytes = n,
            .slot_bytes = m.slot_bytes,
            .timeout_ns = m.timeout_ns,
            .rank = m.me,
            .mode = @backingInt(mode),
        };
        var params = [1]?*anyopaque{@ptrCast(&a)};
        const r = m.d.api.cuLaunchKernel(m.func.handle, blocksFor(n), 1, 1, threads, 1, 1, 0, stream, &params, null);
        m.d.check(r, "cuLaunchKernel tp_mailbox") catch return error.BackendFailed;
    }

    pub fn allGatherBytes(m: *Mailbox, s: DevicePtr, r: DevicePtr, n: usize, stream: collective.Stream) Error!void {
        return m.launch(.gather, s, r, n, stream);
    }

    pub fn exchangeBytes(m: *Mailbox, s: DevicePtr, r: DevicePtr, n: usize, stream: collective.Stream) Error!void {
        return m.launch(.exchange, s, r, n, stream);
    }

    /// Sums only, f32 or bf16 (what the model's partials are); the rest is `error.Unsupported`.
    pub fn sum(m: *Mailbox, s: DevicePtr, r: DevicePtr, count: usize, t: DType, stream: collective.Stream) Error!void {
        const mode: Mode = switch (t) {
            .f32 => .sum_f32,
            .bf16 => .sum_bf16,
            else => return error.Unsupported,
        };
        return m.launch(mode, s, r, count * t.size(), stream);
    }

    /// Ends pending and later exchanges on both ranks (the abort word is shared); this rank's launches stop.
    pub fn abort(m: *Mailbox) void {
        m.aborted.store(true, .release);
        m.ctrlWord(ctrl_abort).store(1, .release);
    }

    /// A wait that timed out or was aborted on either rank.
    pub fn check(m: *Mailbox) Error!void {
        for (0..2) |r| {
            const rec = ctrl_rank + 8 * r;
            if (m.ctrlWord(rec).load(.acquire) != 0) {
                log.warn("tp mailbox: rank {d}'s wait for seq {d} failed ({s} after {d} us; block {d}, flag held {d})", .{ r, m.ctrlWord(rec + 1).load(.acquire), if (m.ctrlWord(rec + 4).load(.acquire) == 1) "aborted" else "timed out", m.ctrlWord(rec + 5).load(.acquire), m.ctrlWord(rec + 3).load(.acquire), m.ctrlWord(rec + 2).load(.acquire) });
                return error.PeerFailed;
            }
        }
        if (m.aborted.load(.acquire) or m.ctrlWord(ctrl_abort).load(.acquire) != 0) return error.Aborted;
    }
};

test "grids grow with the shard and stay powers of two" {
    try std.testing.expectEqual(@as(u32, 1), blocksFor(4096));
    try std.testing.expectEqual(@as(u32, 1), blocksFor(16 * 1024));
    try std.testing.expectEqual(@as(u32, 2), blocksFor(16 * 1024 + 1));
    try std.testing.expectEqual(@as(u32, 4), blocksFor(64 * 1024));
    try std.testing.expectEqual(@as(u32, 16), blocksFor(4 << 20));
    try std.testing.expectEqual(@as(usize, 96), @sizeOf(MailArgs));
}
