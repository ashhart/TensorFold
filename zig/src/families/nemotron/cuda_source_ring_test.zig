//! cuda_source.zig's io_uring reap when a read past a shortened shard's end fails in a batch with another read.
const std = @import("std");
const builtin = @import("builtin");
const cuda = @import("cuda");
const core = @import("core");
const kern = @import("cuda_kernels.zig");
const source = @import("cuda_source.zig");
const t = std.testing;

/// A safetensors shard of U8 tensors `names`, `lens` bytes each, back to back.
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
    for (out, 0..) |*b, i| b.* = @truncate(i *% 131);
    std.mem.writeInt(u64, out[0..8], header.items.len, .little);
    @memcpy(out[8..][0..header.items.len], header.items);
    return out;
}

/// A tensor's mapped bytes and their offset in its file (the mapping itself is not read).
fn find(ck: *const core.Checkpoint, name: []const u8) struct { bytes: []const u8, offset: usize } {
    for (ck.files.items) |*f| if (f.get(name)) |x| return .{ .bytes = x.bytes, .offset = @intFromPtr(x.bytes.ptr) - @intFromPtr(f.map.memory.ptr) };
    unreachable;
}

// a failed read mid-batch must not drop the batch's later completions, or deinit waits for them forever
test "nemotron checkpoint reads: a ring read past a shortened shard's end fails without losing the batch's other read" {
    if (builtin.os.tag != .linux) {
        std.debug.print("nemotron ring batch: skipped, no io_uring on {t}\n", .{builtin.os.tag});
        return error.SkipZigTest;
    }
    const linux = std.os.linux;
    const gpa = t.allocator;
    const io = t.io;
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    const image = try shard(gpa, &.{ "slow", "past" }, &.{ 40 << 20, 20000 });
    defer gpa.free(image);
    try tmp.dir.writeFile(io, .{ .sub_path = "model.safetensors", .data = image });
    var path: [64]u8 = undefined;
    var ck = try core.Checkpoint.openModel(gpa, io, try std.fmt.bufPrint(&path, ".zig-cache/tmp/{s}", .{tmp.sub_path}));
    defer ck.close();
    const slow = find(&ck, "slow");
    const past = find(&ck, "past");
    // pageable slots, and every read here fails before its copy: the driver is never called
    var d: cuda.Driver = undefined;
    var k: kern.Kernels = undefined;
    k.d = &d;
    k.discrete = false;
    var s = try source.Source.init(gpa, .{ .k = &k, .s = cuda.Stream.borrowed(k.d, null) });
    defer s.deinit();
    try s.add(&ck);
    if (s.ring == null) {
        std.debug.print("nemotron ring batch: skipped, io_uring_setup was refused (seccomp?)\n", .{});
        return error.SkipZigTest;
    }
    // the shard shrinks after open: `slow` runs 8 MiB past it into the new end, `past` lies wholly beyond it
    {
        const f = try tmp.dir.openFile(io, "model.safetensors", .{ .mode = .read_write });
        defer f.close(io);
        try f.setLength(io, slow.offset + (8 << 20) + 77);
        try f.sync(io);
        _ = linux.fadvise(f.handle, 0, 0, linux.POSIX_FADV.DONTNEED); // uncached, so `slow` can't finish inside its submit
    }
    try s.upload(1 << 40, slow.bytes);
    // `past` finishes inside its submit; `slow`, still reading, lands after it: one batch of two failed reads
    s.upload(2 << 40, past.bytes) catch |err| {
        std.debug.print("nemotron ring batch: skipped, the first read finished before the second was asked ({t})\n", .{err});
        return error.SkipZigTest;
    };
    try t.expectEqual(@as(usize, 2), s.in_flight);
    const r = &s.ring.?;
    var waited: usize = 0;
    while (r.cq_ready() < 2 and waited < 60_000) : (waited += 1) {
        const ms: std.c.timespec = .{ .sec = 0, .nsec = std.time.ns_per_ms };
        _ = std.c.nanosleep(&ms, null);
    }
    if (r.cq_ready() < 2) {
        std.debug.print("nemotron ring batch: skipped, the 8 MiB read did not finish in 60 s\n", .{});
        return error.SkipZigTest;
    }
    const err: ?anyerror = if (s.flush()) |_| null else |e| e;
    const lost = s.in_flight;
    const left = r.cq_ready();
    // a read still counted with no completion left would keep deinit's reap (cuda_source.zig:83) waiting forever
    for (&s.pending) |*p| p.* = null;
    s.in_flight = 0;
    if (lost != 0) std.debug.print("nemotron ring batch: {s} past the end, then {d} read still in flight with {d} completions left in the ring\n", .{ if (err) |e| @errorName(e) else "no error", lost, left });
    try t.expect(if (err) |e| e == error.EndOfFile or e == error.ReadFailed else false);
    try t.expectEqual(@as(u32, 0), left);
    try t.expectEqual(@as(usize, 0), lost);
}
