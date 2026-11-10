//! cuda_source.zig's checkpoint reads against a fake driver: every byte a slot copies on is the file's, on both paths.
const std = @import("std");
const builtin = @import("builtin");
const cuda = @import("cuda");
const core = @import("core");
const kern = @import("cuda_kernels.zig");
const source = @import("cuda_source.zig");
const abi = cuda.abi;
const dio = core.direct_io;
const t = std.testing;
const linux = builtin.os.tag == .linux;
const gap = 13; // device bytes left between two tensors' copies, so a stray byte shows

/// A driver whose copies land at once in `device` (`base` is its byte 0), logged in order with the waits between.
const Fake = struct {
    const base: abi.DevicePtr = 1 << 40;
    var d: cuda.Driver = undefined;
    var device: []u8 = &.{};
    var calls: std.ArrayList(Call) = .empty;
    var hosts: std.ArrayList([]align(16) u8) = .empty;
    var stray: usize = 0; // copies outside `device`
    const Call = union(enum) { copy: struct { dst: abi.DevicePtr, len: usize }, wait };

    fn log(c: Call) abi.Result {
        calls.append(t.allocator, c) catch return abi.error_not_ready;
        return abi.success;
    }
    fn hostAlloc(out: *?*anyopaque, len: usize, _: c_uint) callconv(.c) abi.Result {
        const b = t.allocator.alignedAlloc(u8, .@"16", len) catch return abi.error_not_ready;
        hosts.append(t.allocator, b) catch {
            t.allocator.free(b);
            return abi.error_not_ready;
        };
        out.* = b.ptr;
        return abi.success;
    }
    fn hostFree(ptr: ?*anyopaque) callconv(.c) abi.Result {
        for (hosts.items, 0..) |b, i| if (@intFromPtr(b.ptr) == @intFromPtr(ptr)) {
            t.allocator.free(b);
            _ = hosts.swapRemove(i);
            return abi.success;
        };
        return abi.error_not_ready;
    }
    fn copy(dst: abi.DevicePtr, src: ?*const anyopaque, len: usize, _: abi.Stream) callconv(.c) abi.Result {
        if (dst < base or dst - base > device.len or len > device.len - (dst - base)) {
            stray += 1;
            return abi.success;
        }
        @memcpy(device[@intCast(dst - base)..][0..len], @as([*]const u8, @ptrCast(src.?))[0..len]);
        return log(.{ .copy = .{ .dst = dst, .len = len } });
    }
    fn streamWait(_: abi.Stream) callconv(.c) abi.Result {
        return log(.wait);
    }
    fn contextWait() callconv(.c) abi.Result {
        return abi.success;
    }
    fn eventCreate(out: *abi.Event, _: c_uint) callconv(.c) abi.Result {
        out.* = @ptrFromInt(0x1000);
        return abi.success;
    }
    fn eventRecord(_: abi.Event, _: abi.Stream) callconv(.c) abi.Result {
        return abi.success;
    }
    fn eventDone(_: abi.Event) callconv(.c) abi.Result {
        return abi.success;
    }

    /// The driver over `bytes` of device memory, nothing logged yet.
    fn init(bytes: []u8) *const cuda.Driver {
        device = bytes;
        stray = 0;
        d = undefined;
        d.api.cuMemHostAlloc = hostAlloc;
        d.api.cuMemFreeHost = hostFree;
        d.api.cuMemcpyHtoDAsync_v2 = copy;
        d.api.cuStreamSynchronize = streamWait;
        d.api.cuCtxSynchronize = contextWait;
        d.api.cuEventCreate = eventCreate;
        d.api.cuEventRecord = eventRecord;
        d.api.cuEventQuery = eventDone;
        d.api.cuEventSynchronize = eventDone;
        d.api.cuEventDestroy_v2 = eventDone;
        return &d;
    }
    fn deinit() void {
        calls.deinit(t.allocator);
        hosts.deinit(t.allocator);
        calls = .empty;
        hosts = .empty;
        device = &.{};
    }
};

/// A safetensors shard of U8 tensors `names`, `lens` bytes each, back to back; every byte comes from its file offset.
fn shard(a: std.mem.Allocator, names: []const []const u8, lens: []const usize) ![]u8 {
    var header: std.ArrayList(u8) = .empty;
    defer header.deinit(a);
    try header.append(a, '{');
    var end: usize = 0;
    for (names, lens, 0..) |n, len, i| {
        if (i > 0) try header.append(a, ',');
        try header.print(a, "\"{s}\":{{\"dtype\":\"U8\",\"shape\":[{d}],\"data_offsets\":[{d},{d}]}}", .{ n, len, end, end + len });
        end += len;
    }
    try header.append(a, '}');
    const out = try a.alloc(u8, 8 + header.items.len + end);
    for (out, 0..) |*b, i| b.* = @truncate((@as(u64, i) *% 0x9e3779b97f4a7c15) >> 56);
    std.mem.writeInt(u64, out[0..8], header.items.len, .little);
    @memcpy(out[8..][0..header.items.len], header.items);
    return out;
}

/// A tensor's mapped bytes and their offset in its file (the mapping itself is not read).
fn find(ck: *const core.Checkpoint, name: []const u8) struct { bytes: []const u8, offset: usize } {
    for (ck.files.items) |*f| if (f.get(name)) |x| return .{ .bytes = x.bytes, .offset = @intFromPtr(x.bytes.ptr) - @intFromPtr(f.map.memory.ptr) };
    unreachable;
}

/// Whether every file still reads with O_DIRECT (FILE_FLAG_NO_BUFFERING on Windows): a refused open or read clears it.
fn direct(s: *const source.Source) bool {
    for (s.files.items) |m| switch (builtin.os.tag) {
        .linux => {
            const flag: c_int = @bitCast(@as(u32, @bitCast(std.c.O{ .DIRECT = true })));
            if (std.c.fcntl(m.file.fd, std.c.F.GETFL) & flag == 0) return false;
        },
        .windows => if (!m.file.direct) return false,
        else => return false,
    };
    return true;
}

/// `ring` false: the thread fallback's reads, as where io_uring_setup is refused; true: the ring's, if it started.
fn usePath(s: *source.Source, ring: bool) !void {
    if (!ring) {
        if (linux) if (s.ring) |*r| {
            r.deinit();
            s.ring = null;
        };
        return;
    }
    if (!linux) {
        std.debug.print("nemotron checkpoint reads: io_uring path skipped, no io_uring on {t}\n", .{builtin.os.tag});
        return error.SkipZigTest;
    }
    if (s.ring == null) {
        std.debug.print("nemotron checkpoint reads: io_uring path skipped, io_uring_setup was refused (seccomp?)\n", .{});
        return error.SkipZigTest;
    }
}

fn send(s: *source.Source, dst: abi.DevicePtr, bytes: []const u8) !void {
    try s.upload(dst, bytes);
    try s.flush();
}

const order = [_][]const u8{ "a0", "b0", "a1", "a2", "b1", "a3", "a4", "b2" };

/// Uploads `order` to device addresses `gap` apart, then flushes: each copy waited for, every byte the file's.
fn roundTrip(s: *source.Source, ck: *const core.Checkpoint) !void {
    @memset(Fake.device, 0xee);
    Fake.calls.clearRetainingCapacity();
    var at: usize = gap;
    var spans: usize = 0;
    for (order) |name| {
        const x = find(ck, name);
        try t.expect(x.offset % dio.alignment != 0 and x.bytes.len % dio.alignment != 0);
        try s.upload(Fake.base + at, x.bytes);
        at += x.bytes.len + gap;
        spans += std.math.divCeil(usize, x.bytes.len, dio.fits(source.slot_bytes)) catch unreachable;
    }
    try s.flush();
    try t.expectEqual(@as(usize, 0), s.in_flight);
    try t.expectEqual(@as(usize, 0), Fake.stray);
    try t.expectEqual(@as(usize, 0), Fake.hosts.items.len);
    // a pageable slot's copy is waited for before its slot takes another read
    try t.expectEqual(2 * spans, Fake.calls.items.len);
    for (Fake.calls.items, 0..) |c, i| try t.expectEqual(i % 2 == 1, c == .wait);
    at = 0;
    for (order) |name| {
        const x = find(ck, name);
        try t.expect(std.mem.allEqual(u8, Fake.device[at..][0..gap], 0xee));
        try t.expectEqualSlices(u8, x.bytes, Fake.device[at + gap ..][0..x.bytes.len]);
        at += gap + x.bytes.len;
    }
    try t.expect(std.mem.allEqual(u8, Fake.device[at..], 0xee));
}

/// Two shards' tensors through one path as opened, then by O_DIRECT's aligned rule where the files opened without it.
fn reads(ring: bool) !void {
    const gpa = t.allocator;
    const io = t.io;
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    const shards = [_]struct { []const u8, []const []const u8, []const usize }{
        .{ "model-00001-of-00002.safetensors", &.{ "a0", "a1", "a2", "a3", "a4" }, &.{ 1, 4095, 4097, dio.fits(source.slot_bytes) + 5003, 12345 } },
        .{ "model-00002-of-00002.safetensors", &.{ "b0", "b1", "b2" }, &.{ 100003, 8191, 3 } },
    };
    var total: usize = gap;
    for (shards) |sh| {
        const image = try shard(gpa, sh[1], sh[2]);
        defer gpa.free(image);
        try tmp.dir.writeFile(io, .{ .sub_path = sh[0], .data = image });
        for (sh[2]) |n| total += n + gap;
    }
    var path: [64]u8 = undefined;
    var ck = try core.Checkpoint.openModel(gpa, io, try std.fmt.bufPrint(&path, ".zig-cache/tmp/{s}", .{tmp.sub_path}));
    defer ck.close();
    const device = try gpa.alloc(u8, total);
    defer gpa.free(device);
    defer Fake.deinit();
    var k: kern.Kernels = undefined;
    k.d = Fake.init(device);
    k.discrete = false;
    var s = try source.Source.init(gpa, .{ .k = &k, .s = cuda.Stream.borrowed(k.d, null) });
    defer s.deinit();
    try s.add(&ck);
    try usePath(&s, ring);
    try roundTrip(&s, &ck);
    try t.expectEqual(ring, s.ring != null);
    if (direct(&s)) return;
    // a page-cache descriptor read by O_DIRECT's rule: aligned spans, short reads retried from a 4 KiB boundary
    for (s.files.items) |*m| m.file.direct = true;
    try roundTrip(&s, &ck);
    if (linux or builtin.os.tag == .windows) {
        const how = if (ring) "io_uring" else "thread";
        std.debug.print("nemotron checkpoint reads: O_DIRECT skipped, the file system refused it; {s} reads checked through the page cache\n", .{how});
        return error.SkipZigTest;
    }
}

test "nemotron checkpoint reads: the thread fallback copies every slot's file bytes to its device address" {
    try reads(false);
}

test "nemotron checkpoint reads: the io_uring ring copies every slot's file bytes to its device address" {
    try reads(true);
}

// by the aligned rule a short read's retry re-reads the same tail: that is the end of the file, not a failed read
test "nemotron checkpoint reads: a tensor past a shortened shard's end fails with EndOfFile and copies nothing" {
    if (builtin.os.tag == .windows) {
        std.debug.print("nemotron checkpoint reads: shortened shard skipped, Windows can't shorten a mapped file\n", .{});
        return error.SkipZigTest;
    }
    const gpa = t.allocator;
    const io = t.io;
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    const image = try shard(gpa, &.{ "kept", "cut" }, &.{ 5000, 20000 });
    defer gpa.free(image);
    var path: [64]u8 = undefined;
    const dir = try std.fmt.bufPrint(&path, ".zig-cache/tmp/{s}", .{tmp.sub_path});
    // a header that says more than its shard holds is refused at open (safetensors.zig:120), before any read
    try tmp.dir.writeFile(io, .{ .sub_path = "model.safetensors", .data = image[0 .. image.len - 1] });
    try t.expectError(error.BadSafetensors, core.Checkpoint.openModel(gpa, io, dir));
    try tmp.dir.writeFile(io, .{ .sub_path = "model.safetensors", .data = image });
    var ck = try core.Checkpoint.openModel(gpa, io, dir);
    defer ck.close();
    const kept = find(&ck, "kept");
    const cut = find(&ck, "cut");
    const device = try gpa.alloc(u8, kept.bytes.len + cut.bytes.len);
    defer gpa.free(device);
    defer Fake.deinit();
    var k: kern.Kernels = undefined;
    k.d = Fake.init(device);
    k.discrete = false;
    var s = try source.Source.init(gpa, .{ .k = &k, .s = cuda.Stream.borrowed(k.d, null) });
    defer s.deinit();
    try s.add(&ck);
    // the shard shrinks after open: its end falls inside "cut", off a 4 KiB boundary, so the mapping still covers it
    var end = cut.offset + 7001;
    if (end % dio.alignment == 0) end += 1;
    try t.expect(end > std.mem.alignBackward(usize, cut.offset, dio.alignment));
    {
        const f = try tmp.dir.openFile(io, "model.safetensors", .{ .mode = .read_write });
        defer f.close(io);
        try f.setLength(io, end);
    }
    const opened = s.files.items[0].file.direct;
    var wrong = false;
    for ([_]bool{ true, false }) |ring| {
        if (ring and s.ring == null) {
            std.debug.print("nemotron checkpoint reads: shortened shard on io_uring skipped, no ring here\n", .{});
            continue;
        }
        try usePath(&s, ring);
        for ([_]bool{ false, true }) |aligned| {
            if (aligned and opened) continue; // an O_DIRECT descriptor already reads by the aligned rule
            s.files.items[0].file.direct = opened or aligned;
            Fake.calls.clearRetainingCapacity();
            @memset(device, 0xee);
            // bytes before the new end still read
            try s.upload(Fake.base, kept.bytes);
            try s.flush();
            try t.expectEqualSlices(u8, image[kept.offset..][0..kept.bytes.len], device[0..kept.bytes.len]);
            Fake.calls.clearRetainingCapacity();
            const got: ?anyerror = if (send(&s, Fake.base + kept.bytes.len, cut.bytes)) |_| null else |e| e;
            try t.expectEqual(@as(usize, 0), s.in_flight);
            try t.expectEqual(@as(usize, 0), Fake.calls.items.len);
            try t.expect(std.mem.allEqual(u8, device[kept.bytes.len..], 0xee));
            if (if (got) |e| e == error.EndOfFile else false) continue;
            wrong = true;
            const how = if (opened) "O_DIRECT" else if (aligned) "the aligned O_DIRECT rule on a page-cache descriptor" else "the page cache";
            const what = if (got) |e| @errorName(e) else "no error";
            std.debug.print("nemotron checkpoint reads: {s} reads through {s} gave {s} past the shard's end, want EndOfFile\n", .{ if (ring) "io_uring" else "thread", how, what });
        }
    }
    if (linux and !direct(&s)) std.debug.print("nemotron checkpoint reads: shortened shard through O_DIRECT skipped, the file system refused it\n", .{});
    try t.expect(!wrong);
}
