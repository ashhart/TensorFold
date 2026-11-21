//! GB10 display memory as device memory: a DRM dumb buffer or a daemon's lent slice, carved into cache spans.

const std = @import("std");
const builtin = @import("builtin");
const linux = std.os.linux;
const abi = @import("abi.zig");
const Driver = @import("driver.zig").Driver;
const DriverError = @import("driver.zig").Error;
const DeviceBuffer = @import("memory.zig").DeviceBuffer;
const env = @import("sys").environment;

pub const width = 4096; // pixels a framebuffer row
pub const bpp = 32;
pub const row_bytes = width * bpp / 8; // the unit a carveout's size rounds down to
pub const default_bytes: usize = 1792 << 20; // what GB10's display reservation holds with room to spare
pub const default_card = "/dev/dri/card0";
pub const alignment = 256; // cuMemAlloc's own

/// struct drm_mode_create_dumb (drm_mode.h): height, width and bpp in; handle, pitch and size out.
pub const CreateDumb = extern struct { height: u32, width: u32, bpp: u32, flags: u32 = 0, handle: u32 = 0, pitch: u32 = 0, size: u64 = 0 };
/// struct drm_mode_map_dumb: the handle in, the offset to mmap the card at out.
pub const MapDumb = extern struct { handle: u32, pad: u32 = 0, offset: u64 = 0 };
/// struct drm_mode_destroy_dumb.
pub const DestroyDumb = extern struct { handle: u32 };

pub const ioctl_create_dumb = linux.IOCTL.IOWR('d', 0xB2, CreateDumb);
pub const ioctl_map_dumb = linux.IOCTL.IOWR('d', 0xB3, MapDumb);
pub const ioctl_destroy_dumb = linux.IOCTL.IOWR('d', 0xB4, DestroyDumb);

pub const Error = DriverError || error{ CardUnavailable, DumbBufferRefused, MapRefused, RegisterRefused, NoDaemon, DaemonSilent, DaemonRefused, SliceTooSmall, VmmUnavailable, MapFailed, KeyFileRefused };

/// Which way the display memory is obtained: a DRM dumb buffer, or a display-memory daemon lending its reservation.
pub const Backend = enum { drm, daemon };

/// What the settings asked for: the source, the size in bytes (whole framebuffer rows for drm), and the source's own.
pub const Request = struct {
    backend: Backend,
    bytes: usize,
    card: []const u8 = default_card,
    socket: []const u8 = default_socket,
    key: []const u8 = default_key,
    /// Set when the key comes from a file; the file is opened only for the daemon source, as the claim is made.
    key_file: ?[]const u8 = null,
};

/// `bytes` in whole framebuffer rows; zero below one row.
pub fn roundBytes(bytes: usize) usize {
    return bytes / row_bytes * row_bytes;
}

/// The carveout's settings as their texts: `flag` is the command line's, the rest platform/environment.zig's rows.
pub const Settings = struct {
    flag: bool = false,
    on: ?[]const u8 = null,
    source: ?[]const u8 = null,
    socket: ?[]const u8 = null,
    key: ?[]const u8 = null,
    key_file: ?[]const u8 = null,
    mib: ?[]const u8 = null,
    card: ?[]const u8 = null,

    /// Every row through `environment`'s `get`: a process's environ map, or platform/environment.zig's Process.
    pub fn read(flag: bool, environment: anytype) Settings {
        return .{
            .flag = flag,
            .on = environment.get("TF_CUDA_CARVEOUT"),
            .source = environment.get("TF_CUDA_CARVEOUT_SOURCE"),
            .socket = environment.get("TF_CUDA_CARVEOUT_SOCKET"),
            .key = environment.get("TF_CUDA_CARVEOUT_KEY"),
            .key_file = environment.get("TF_CUDA_CARVEOUT_KEY_FILE"),
            .mib = environment.get("TF_CUDA_CARVEOUT_MIB"),
            .card = environment.get("TF_DRM_CARD"),
        };
    }
};

/// What the settings ask for; null means off. `flag` or `on` = "1" turns it on, and a mistyped row is then refused.
pub fn requested(s: Settings) error{ InvalidCharacter, Overflow, CarveoutTooSmall, InvalidEnvironment }!?Request {
    if (!s.flag) _ = try env.parse("TF_CUDA_CARVEOUT", s.on);
    const wanted = s.flag or (if (s.on) |v| std.mem.eql(u8, v, "1") else false);
    if (!wanted) return null;
    const lend = ((try env.parse("TF_CUDA_CARVEOUT_SOURCE", s.source)) orelse 0) == 1;
    var r: Request = .{ .backend = if (lend) .daemon else .drm, .bytes = default_bytes };
    if (lend) {
        if (try env.path("TF_CUDA_CARVEOUT_SOCKET", s.socket)) |text| r.socket = text;
        // the key's own row first, else its file (read when a daemon's claim is made), else the default
        if (try env.path("TF_CUDA_CARVEOUT_KEY", s.key)) |text| r.key = text else r.key_file = try env.path("TF_CUDA_CARVEOUT_KEY_FILE", s.key_file);
    } else {
        if (try env.path("TF_DRM_CARD", s.card)) |text| r.card = text;
    }
    if (try env.parse("TF_CUDA_CARVEOUT_MIB", s.mib)) |mib| {
        const n = std.math.cast(usize, mib) orelse return error.Overflow;
        r.bytes = try std.math.mul(usize, n, 1 << 20);
    }
    if (lend) {
        if (r.bytes < gran) return error.CarveoutTooSmall;
        return r;
    }
    r.bytes = roundBytes(r.bytes);
    if (r.bytes == 0) return error.CarveoutTooSmall;
    return r;
}

/// The most bytes of a key file that are read: a key's line is shorter than this.
pub const key_max = 256;

/// The daemon's key: `path`'s first line, kept in `held`. A refusal names the setting, never the path or the bytes.
pub fn keyFile(path: []const u8, held: *[key_max]u8, problem: *[]const u8) error{KeyFileRefused}![]const u8 {
    const name = "TF_CUDA_CARVEOUT_KEY_FILE";
    var z: [4096]u8 = undefined;
    if (path.len >= z.len) return refuse(problem, name ++ " names a path of 4096 bytes or more");
    @memcpy(z[0..path.len], path);
    z[path.len] = 0;
    const rc = linux.open(@ptrCast(&z), .{ .ACCMODE = .RDONLY, .CLOEXEC = true, .NONBLOCK = true }, 0); // a FIFO must not block the open
    switch (linux.errno(rc)) {
        .SUCCESS => {},
        .NOENT => return refuse(problem, name ++ " names a file that does not exist"),
        .ACCES => return refuse(problem, name ++ " names a file this user may not read"),
        else => return refuse(problem, name ++ " names a file that cannot be opened"),
    }
    const fd: linux.fd_t = @intCast(rc);
    defer _ = linux.close(fd);
    var sx: linux.Statx = undefined;
    if (linux.errno(linux.statx(fd, "", linux.AT.EMPTY_PATH, .{ .TYPE = true }, &sx)) != .SUCCESS or (sx.mode & linux.S.IFMT) != linux.S.IFREG)
        return refuse(problem, name ++ " names something that is not a regular file");
    const n = linux.read(fd, held, held.len);
    if (linux.errno(n) != .SUCCESS) return refuse(problem, name ++ " names a file that cannot be read");
    const first = std.mem.sliceTo(held[0..n], '\n');
    if (first.len == held.len) return refuse(problem, name ++ " names a file whose first line is 256 bytes or more");
    const line = std.mem.trimEnd(u8, first, "\r");
    if (line.len == 0) return refuse(problem, name ++ " names a file whose first line is empty");
    return line;
}

fn refuse(problem: *[]const u8, comptime why: []const u8) error{KeyFileRefused} {
    problem.* = why;
    return error.KeyFileRefused;
}

/// First-fit placement of 256-byte aligned spans in `size` bytes; with nothing given back it is a bump allocator.
pub const Spans = struct {
    pub const max = 64;
    pub const Range = struct { at: usize, len: usize };

    size: usize,
    live: [max]Range = undefined, // sorted by offset
    n: usize = 0,

    /// The offset of a new span of `len` bytes, or null when no gap (or no slot) holds it.
    pub fn take(s: *Spans, len: usize) ?usize {
        if (len == 0 or s.n == max) return null;
        var at: usize = 0;
        for (0..s.n + 1) |i| {
            const end = if (i < s.n) s.live[i].at else s.size;
            if (at <= end and end - at >= len) {
                std.mem.copyBackwards(Range, s.live[i + 1 .. s.n + 1], s.live[i..s.n]);
                s.live[i] = .{ .at = at, .len = len };
                s.n += 1;
                return at;
            }
            if (i < s.n) at = std.mem.alignForward(usize, s.live[i].at + s.live[i].len, alignment);
        }
        return null;
    }

    /// Gives back the span at `at`; an offset no span starts at is ignored.
    pub fn release(s: *Spans, at: usize) void {
        for (s.live[0..s.n], 0..) |r, i| if (r.at == at) {
            std.mem.copyForwards(Range, s.live[i .. s.n - 1], s.live[i + 1 .. s.n]);
            s.n -= 1;
            return;
        };
    }

    /// Bytes no live span holds (alignment gaps count as free).
    pub fn free(s: *const Spans) usize {
        var used: usize = 0;
        for (s.live[0..s.n]) |r| used += r.len;
        return s.size - used;
    }
};

/// The reference daemon's socket and key (kindling spark-os's dispramd); another daemon sets both.
pub const default_socket = "/run/dispram/dispram.sock";
pub const default_key = "kindlingai_1";
pub const gran = @import("carveout_lend.zig").gran; // one owner for the slice size
/// A daemon's refusal in its own words, for the line that reports it.
const Reason = @import("carveout_lend.zig").Reason;

/// The display memory held for the KV planes: a dumb buffer's mapping, or a lent slice's with its daemon connection.
pub const Carveout = struct {
    d: *const Driver,
    backend: Backend = .drm,
    fd: linux.fd_t,
    handle: u32,
    host: []align(std.heap.page_size_min) u8,
    dev: abi.DevicePtr,
    flags: c_uint, // the registration flags the driver took
    spans: Spans,
    vmm: ?@import("arena.zig").Vmm = null, // the daemon slice's mapping; the card path has none

    /// Opens `want` from its source. A refusal is logged once: the cause, a daemon's reason, the setting to change.
    pub fn from(d: *const Driver, want: Request) Error!Carveout {
        const mib = want.bytes >> 20;
        switch (want.backend) {
            .drm => {
                var z: [4096]u8 = undefined;
                const card = std.fmt.bufPrintSentinel(&z, "{s}", .{want.card}, 0) catch return error.CardUnavailable;
                return open(d, card, want.bytes) catch |err| {
                    if (!builtin.is_test) std.log.err("display carveout of {d} MiB from {s}: {t} (needs nvidia_drm modeset=1, the card's device node and no display in use)", .{ mib, want.card, err });
                    return err;
                };
            },
            .daemon => {
                var line: [key_max]u8 = undefined;
                var problem: []const u8 = "";
                const key = if (want.key_file) |file| keyFile(file, &line, &problem) catch |err| {
                    if (!builtin.is_test) std.log.err("{s}", .{problem});
                    return err;
                } else want.key;
                var said: Reason = .{};
                return openDaemon(d, want.socket, key, want.bytes, &said) catch |err| {
                    if (!builtin.is_test) std.log.err("display carveout of {d} MiB from the daemon at {s}: {t}{f} (run the daemon, set TF_CUDA_CARVEOUT_SOCKET and TF_CUDA_CARVEOUT_KEY or TF_CUDA_CARVEOUT_KEY_FILE, or TF_CUDA_CARVEOUT_SOURCE=drm)", .{ mib, want.socket, err, said });
                    return err;
                };
            },
        }
    }

    /// How the memory is held, for a caller's start-up line.
    pub fn how(c: *const Carveout) []const u8 {
        return switch (c.backend) {
            .drm => if (c.flags & abi.host_register_iomemory != 0) "a DRM dumb buffer, registered DEVICEMAP|IOMEMORY" else "a DRM dumb buffer, registered DEVICEMAP",
            .daemon => "a daemon's lent slice, mapped as device memory",
        };
    }

    /// Maps `want` bytes (whole rows) of `card`'s display memory into the current context.
    pub fn open(d: *const Driver, card: [*:0]const u8, want: usize) Error!Carveout {
        const size = roundBytes(want);
        if (size == 0) return error.Invalid;
        const fd_rc = linux.open(card, .{ .ACCMODE = .RDWR, .CLOEXEC = true }, 0);
        if (linux.errno(fd_rc) != .SUCCESS) return error.CardUnavailable;
        const fd: linux.fd_t = @intCast(fd_rc);
        errdefer _ = linux.close(fd);
        var create: CreateDumb = .{ .height = @intCast(size / row_bytes), .width = width, .bpp = bpp };
        if (linux.errno(linux.ioctl(fd, ioctl_create_dumb, @intFromPtr(&create))) != .SUCCESS) return error.DumbBufferRefused;
        errdefer destroy(fd, create.handle);
        if (create.pitch != row_bytes or create.size < size) return error.DumbBufferRefused;
        var map: MapDumb = .{ .handle = create.handle };
        if (linux.errno(linux.ioctl(fd, ioctl_map_dumb, @intFromPtr(&map))) != .SUCCESS) return error.MapRefused;
        const addr = linux.mmap(null, size, .{ .READ = true, .WRITE = true }, .{ .TYPE = .SHARED }, fd, @intCast(map.offset));
        if (linux.errno(addr) != .SUCCESS) return error.MapRefused;
        const base: [*]align(std.heap.page_size_min) u8 = @ptrFromInt(addr);
        errdefer _ = linux.munmap(base, size);
        const flags = try register(d, base, size);
        errdefer _ = d.api.cuMemHostUnregister(base);
        var dev: abi.DevicePtr = 0;
        try d.check(d.api.cuMemHostGetDevicePointer_v2(&dev, base, 0), "cuMemHostGetDevicePointer");
        return .{ .d = d, .fd = fd, .handle = create.handle, .host = base[0..size], .dev = dev, .flags = flags, .spans = .{ .size = size } };
    }

    /// DEVICEMAP first; a driver that sees the range as I/O memory wants IOMEMORY said too.
    fn register(d: *const Driver, base: [*]u8, len: usize) Error!c_uint {
        const plain = abi.host_register_devicemap;
        if (d.api.cuMemHostRegister_v2(base, len, plain) == abi.success) return plain;
        const io = plain | abi.host_register_iomemory;
        d.check(d.api.cuMemHostRegister_v2(base, len, io), "cuMemHostRegister") catch return error.RegisterRefused;
        return io;
    }

    fn destroy(fd: linux.fd_t, handle: u32) void {
        var x: DestroyDumb = .{ .handle = handle };
        _ = linux.ioctl(fd, ioctl_destroy_dumb, @intFromPtr(&x));
    }

    /// Releases in reverse order of open; every span taken from it must be out of use.
    pub fn close(c: *Carveout) void {
        switch (c.backend) {
            .drm => {
                _ = c.d.api.cuMemHostUnregister(c.host.ptr);
                _ = linux.munmap(c.host.ptr, c.host.len);
                destroy(c.fd, c.handle);
                _ = linux.close(c.fd);
            },
            .daemon => @import("carveout_lend.zig").release(.{ .fd = c.fd, .dev = c.dev, .size = c.spans.size, .vmm = c.vmm.? }),
        }
        c.* = undefined;
    }

    /// The daemon backend: claims what a lending daemon holds, over its socket with its key, keeping the socket open.
    pub fn openDaemon(d: *const Driver, socket: []const u8, key: []const u8, want: usize, said: *Reason) Error!Carveout {
        const claimed = try @import("carveout_lend.zig").claim(d, socket, key, want, said);
        return .{ .d = d, .backend = .daemon, .fd = claimed.fd, .handle = 0, .host = undefined, .dev = claimed.dev, .flags = 0, .spans = .{ .size = claimed.size }, .vmm = claimed.vmm };
    }

    /// A zeroed span of `len` bytes, 256-byte aligned; null when it does not fit (the caller falls back).
    pub fn take(c: *Carveout, len: usize) Error!?DeviceBuffer {
        const at = c.spans.take(len) orelse return null;
        errdefer c.spans.release(at);
        const b: DeviceBuffer = .{ .d = c.d, .ptr = c.dev + at, .len = len, .borrowed = true };
        try b.fill8(0, null);
        return b;
    }

    /// Returns a span `take` gave out.
    pub fn give(c: *Carveout, b: DeviceBuffer) void {
        c.spans.release(b.ptr - c.dev);
    }

    pub fn bytes(c: *const Carveout) usize {
        return c.spans.size;
    }

    pub fn freeBytes(c: *const Carveout) usize {
        return c.spans.free();
    }
};

test "DRM ioctl numbers are drm_mode.h's" {
    try std.testing.expectEqual(@as(u32, 0xC02064B2), ioctl_create_dumb);
    try std.testing.expectEqual(@as(u32, 0xC01064B3), ioctl_map_dumb);
    try std.testing.expectEqual(@as(u32, 0xC00464B4), ioctl_destroy_dumb);
}

test "dumb buffer structs have drm_mode.h's layout" {
    try std.testing.expectEqual(32, @sizeOf(CreateDumb));
    try std.testing.expectEqual(16, @offsetOf(CreateDumb, "handle"));
    try std.testing.expectEqual(20, @offsetOf(CreateDumb, "pitch"));
    try std.testing.expectEqual(24, @offsetOf(CreateDumb, "size"));
    try std.testing.expectEqual(16, @sizeOf(MapDumb));
    try std.testing.expectEqual(8, @offsetOf(MapDumb, "offset"));
    try std.testing.expectEqual(4, @sizeOf(DestroyDumb));
}

test "spans are 256-byte aligned and refused once full" {
    var s: Spans = .{ .size = 4096 };
    try std.testing.expectEqual(@as(?usize, 0), s.take(100));
    try std.testing.expectEqual(@as(?usize, 256), s.take(1000));
    try std.testing.expectEqual(@as(?usize, 1280), s.take(2816));
    try std.testing.expectEqual(4096 - 3916, s.free());
    try std.testing.expectEqual(@as(?usize, null), s.take(1));
    try std.testing.expectEqual(@as(?usize, null), s.take(0));
    var big: Spans = .{ .size = 1 << 20 };
    try std.testing.expectEqual(@as(?usize, null), big.take((1 << 20) + 1));
    try std.testing.expectEqual(@as(?usize, 0), big.take(1 << 20));
}

test "a given-back span's room is taken again, first fit" {
    var s: Spans = .{ .size = 4096 };
    _ = s.take(512).?;
    const mid = s.take(1024).?;
    _ = s.take(512).?;
    s.release(mid);
    s.release(12345); // no span starts there
    try std.testing.expectEqual(@as(?usize, 512), s.take(768));
    try std.testing.expectEqual(@as(?usize, 2048), s.take(512)); // 1280..1536 is too small
    try std.testing.expectEqual(@as(?usize, 1280), s.take(256));
    try std.testing.expectEqual(@as(usize, 1536), s.free());
}

test "a span list holds at most max spans" {
    var s: Spans = .{ .size = 1 << 20 };
    for (0..Spans.max) |_| _ = s.take(1).?;
    try std.testing.expectEqual(@as(?usize, null), s.take(1));
}

test "the carveout is opt-in, sized in whole rows, with a source a value selects" {
    const t = std.testing;
    try t.expectEqual(@as(?Request, null), try requested(.{}));
    try t.expectEqual(@as(?Request, null), try requested(.{ .on = "0", .mib = "64" }));
    try t.expectEqual(@as(?Request, null), try requested(.{ .source = "daemon", .mib = "64" })); // off: the source alone does nothing
    try t.expectEqual(@as(?Request, null), try requested(.{ .source = "magic", .key_file = "\x1b" })); // off: its rows are not read
    try t.expectEqual(@as(?Request, .{ .backend = .drm, .bytes = default_bytes }), try requested(.{ .flag = true })); // drm is the default source
    try t.expectEqual(@as(?Request, .{ .backend = .drm, .bytes = default_bytes }), try requested(.{ .on = "1", .source = "drm", .mib = "" }));
    try t.expectEqual(@as(?Request, .{ .backend = .drm, .bytes = 64 << 20 }), try requested(.{ .on = "1", .mib = "64" }));
    try t.expectEqual(@as(?Request, .{ .backend = .daemon, .bytes = default_bytes }), try requested(.{ .flag = true, .source = "daemon" }));
    try t.expectEqual(@as(?Request, .{ .backend = .daemon, .bytes = 64 << 20 }), try requested(.{ .on = "1", .source = "daemon", .mib = "64" }));
    try t.expectError(error.InvalidEnvironment, requested(.{ .on = "true" })); // the switch is 0 or 1
    try t.expectError(error.InvalidEnvironment, requested(.{ .flag = true, .mib = "lots" }));
    try t.expectError(error.InvalidEnvironment, requested(.{ .flag = true, .source = "magic" })); // a third source is refused
    try t.expectError(error.CarveoutTooSmall, requested(.{ .flag = true, .source = "daemon", .mib = "1" })); // under one slice
    try t.expectEqual(0, roundBytes(row_bytes - 1));
    try t.expectEqual(row_bytes, roundBytes(row_bytes + 1));
}

test "a source's own settings: blank means the default, and the key's row comes before its file" {
    const t = std.testing;
    try t.expectEqualStrings(default_card, (try requested(.{ .flag = true })).?.card);
    try t.expectEqualStrings(default_card, (try requested(.{ .flag = true, .card = " " })).?.card);
    try t.expectEqualStrings("/dev/dri/card1", (try requested(.{ .flag = true, .card = "/dev/dri/card1" })).?.card);
    const lent = (try requested(.{ .flag = true, .source = "daemon", .socket = "", .key = "", .card = "\x1b" })).?; // no card is opened
    try t.expectEqualStrings(default_socket, lent.socket);
    try t.expectEqualStrings(default_key, lent.key);
    try t.expectEqual(@as(?[]const u8, null), lent.key_file);
    const attic = (try requested(.{ .flag = true, .source = "daemon", .socket = "/run/spark-attic/attic.sock", .key_file = "/etc/spark-attic/key" })).?;
    try t.expectEqualStrings("/run/spark-attic/attic.sock", attic.socket);
    try t.expectEqualStrings("/etc/spark-attic/key", attic.key_file.?);
    const both = (try requested(.{ .flag = true, .source = "daemon", .key = "per-box", .key_file = "/etc/spark-attic/key" })).?;
    try t.expectEqualStrings("per-box", both.key);
    try t.expectEqual(@as(?[]const u8, null), both.key_file);
    try t.expectError(error.InvalidEnvironment, requested(.{ .flag = true, .card = "/dev/dri/card\x1b" }));
    try t.expectError(error.InvalidEnvironment, requested(.{ .flag = true, .source = "daemon", .key_file = "key\x7f" }));
    // the daemon's rows belong to the daemon source: a stale one does not stop a drm start
    try t.expectEqual(@as(?Request, .{ .backend = .drm, .bytes = default_bytes }), try requested(.{ .flag = true, .socket = "\x1b", .key = "\x1b", .key_file = "\x1b" }));
}

/// A path under the test's tmp dir, relative to the working directory as tmpDir's is; `data` writes the file first.
fn keyPath(buf: []u8, tmp: *std.testing.TmpDir, name: []const u8, data: ?[]const u8) ![]const u8 {
    if (data) |bytes| try tmp.dir.writeFile(std.testing.io, .{ .sub_path = name, .data = bytes });
    return std.fmt.bufPrint(buf, ".zig-cache/tmp/{s}/{s}", .{ tmp.sub_path, name });
}

test "the key file's first line is the key, with its line ending dropped" {
    if (comptime builtin.os.tag != .linux) return error.SkipZigTest;
    const t = std.testing;
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    var path: [256]u8 = undefined;
    var held: [key_max]u8 = undefined;
    var problem: []const u8 = "";
    try t.expectEqualStrings("box-key-1", try keyFile(try keyPath(&path, &tmp, "lf", "box-key-1\nnot the key\n"), &held, &problem));
    try t.expectEqualStrings("box-key-2", try keyFile(try keyPath(&path, &tmp, "crlf", "box-key-2\r\n"), &held, &problem));
    try t.expectEqualStrings("box-key-3", try keyFile(try keyPath(&path, &tmp, "bare", "box-key-3"), &held, &problem));
    try t.expectEqualStrings("", problem);
}

test "a key file that is missing, empty, too long or not a file is refused by the setting's name, its path and bytes unechoed" {
    if (comptime builtin.os.tag != .linux) return error.SkipZigTest;
    const t = std.testing;
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    var path: [256]u8 = undefined;
    var held: [key_max]u8 = undefined;
    const long: [key_max]u8 = @splat('k');
    const cases = [_]struct { name: []const u8, data: ?[]const u8, why: []const u8 }{
        .{ .name = "missing", .data = null, .why = "does not exist" },
        .{ .name = "empty", .data = "", .why = "first line is empty" },
        .{ .name = "blank-first", .data = "\nSECRET-on-line-two\n", .why = "first line is empty" },
        .{ .name = "cr-only", .data = "\r\nSECRET-on-line-two\n", .why = "first line is empty" },
        .{ .name = "long", .data = &long, .why = "256 bytes or more" },
        .{ .name = ".", .data = null, .why = "not a regular file" },
    };
    for (cases) |c| {
        var problem: []const u8 = "";
        const p = try keyPath(&path, &tmp, c.name, c.data);
        try t.expectError(error.KeyFileRefused, keyFile(p, &held, &problem));
        try t.expect(std.mem.startsWith(u8, problem, "TF_CUDA_CARVEOUT_KEY_FILE "));
        try t.expect(std.mem.indexOf(u8, problem, c.why) != null);
        try t.expect(std.mem.indexOf(u8, problem, p) == null and std.mem.indexOf(u8, problem, "SECRET") == null);
    }
}

test "the key file is read for the daemon source alone, and before any driver call" {
    if (comptime builtin.os.tag != .linux) return error.SkipZigTest;
    const t = std.testing;
    var d: Driver = undefined; // no refusal below reaches it
    _ = &d;
    const absent = ".zig-cache/tmp/absent/key";
    const nowhere = ".zig-cache/tmp/absent/d.sock";
    // drm: the card is what stops it, so the missing key file was never opened
    try t.expectError(error.CardUnavailable, Carveout.from(&d, .{ .backend = .drm, .bytes = row_bytes, .card = ".zig-cache/tmp/absent/card", .key_file = absent }));
    try t.expectError(error.KeyFileRefused, Carveout.from(&d, .{ .backend = .daemon, .bytes = gran, .socket = nowhere, .key_file = absent }));
    // the key's own row: no file is opened, and the missing socket is what stops it
    try t.expectError(error.NoDaemon, Carveout.from(&d, .{ .backend = .daemon, .bytes = gran, .socket = nowhere }));
}

test "the driver-facing code compiles on Linux" {
    if (comptime builtin.os.tag == .linux) std.testing.refAllDecls(Carveout);
}
