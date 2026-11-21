//! Real two-process/two-device NCCL sums and bounded disconnect/timeout behavior over the framed control link.
const std = @import("std");
const builtin = @import("builtin");
const cuda = @import("cuda");
const flash = @import("flashnext");
const check = @import("check.zig");
const wire = flash.rank_wire;

pub fn run(gpu: check.Gpu, rank: usize, port: u16, mode_arg: []const u8, master: []const u8) !void {
    const modes = [_][]const u8{ "sum", "eof", "timeout", "missing", "eof-scatter", "timeout-scatter", "exchange" };
    for (modes) |m| {
        if (std.mem.eql(u8, m, mode_arg)) break;
    } else return error.UnknownRankMode;
    // eof-scatter and timeout-scatter: the peer fails during a reduce-scatter, the first collective
    const scatter = std.mem.endsWith(u8, mode_arg, "-scatter");
    const mode = if (scatter) mode_arg[0 .. mode_arg.len - "-scatter".len] else mode_arg;
    const missing = std.mem.eql(u8, mode, "missing");
    const started = wire.now();
    if (missing) {
        try std.testing.expectError(error.RankTimeout, wire.Link.open(gpu.io, master, port, rank, 200));
        try check.expect(wire.now() - started < 3 * std.time.ns_per_s, "missing peer startup is bounded", .{});
        check.pass("rank {d}: missing peer startup timed out within bound", .{rank});
        return;
    }
    const link = try wire.Link.open(gpu.io, master, port, rank, 30_000);
    defer link.close();
    var no_delay: c_int = 0;
    var option_len: std.posix.socklen_t = @sizeOf(c_int);
    try check.expect(std.c.getsockopt(link.fd, std.posix.IPPROTO.TCP, std.posix.TCP.NODELAY, &no_delay, &option_len) == 0 and no_delay == 1, "both connected rank sockets disable Nagle", .{});
    const until = wire.deadline(30_000);
    const hello: wire.Header = .{ .kind = .hello, .sequence = 1, .object = 77, .arg = 123, .bytes = 7 };
    if (rank == 0) {
        const raw = try hello.encode();
        try link.send(raw[0..3], until);
        try link.send(raw[3..], until);
        try link.send("payload", until);
        const ack = try link.receiveHeader(until);
        try check.expect(ack.kind == .ack and ack.sequence == 1 and ack.bytes == 0, "exact hello acknowledgment", .{});
    } else {
        try std.testing.expectEqualDeep(hello, try link.receiveHeader(until));
        var payload: [7]u8 = undefined;
        try link.receive(&payload, until);
        try check.sameBytes("fragmented control frame payload", &payload, "payload");
        try link.sendHeader(.{ .kind = .ack, .sequence = 1 }, until);
    }
    const before = try gpu.ctx.memInfo();
    const opened = wire.now();
    var collective = try flash.collective.Collective.init(link, rank, 30_000);
    defer collective.deinit();
    check.pass("rank {d}: init {d} ms, {d} MiB of device memory", .{ rank, (wire.now() - opened) / std.time.ns_per_ms, (before.free -| (try gpu.ctx.memInfo()).free) >> 20 });
    // the peer's silence bounds the collective, not NCCL's initialization, which can take longer than this
    if (std.mem.eql(u8, mode, "timeout")) collective.timeout_ms = 200;
    var stream = try cuda.Stream.init(gpu.d, true);
    defer stream.deinit();
    if (std.mem.eql(u8, mode, "exchange")) {
        try @import("flashnext_exchange_tests.zig").run(gpu, &collective, stream, rank);
        return meet(link, rank, 99);
    }
    if (std.mem.eql(u8, mode, "eof") or std.mem.eql(u8, mode, "timeout")) {
        // both communicators are up before the peer fails, so the failure meets the collective, not the init
        try meet(link, rank, 98);
        if (rank == 1) {
            // A deliberate peer omission: the owner either disconnects or stays alive without entering the collective.
            if (std.mem.eql(u8, mode, "eof")) {
                _ = std.c.shutdown(link.fd, std.posix.SHUT.RDWR);
            } else {
                outlive();
            }
            check.pass("rank 1: injected {s}", .{mode_arg});
            return;
        }
        var values = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.asBytes(&[2]f32{ 1, 1 }));
        defer values.free();
        const start = wire.now();
        completeSum(&collective, stream, values.ptr, 1, scatter) catch |err| {
            try check.expect(err == error.RankDisconnected or err == error.RankTimeout or err == error.NcclFailed, "failure is surfaced ({s})", .{@errorName(err)});
            try check.expect(collective.failed and collective.comm == null and collective.parent == null, "failed communicators are aborted and cannot be reused", .{});
            try check.expect(wire.now() - start < 3 * std.time.ns_per_s, "failed collective terminates within bound", .{});
            try std.testing.expectError(error.RankDisconnected, collective.sum(stream, values.ptr, 1));
            try std.testing.expectError(error.RankDisconnected, collective.sumBlock(stream, values.ptr, 1));
            check.pass("rank 0: {s} surfaced as {s}, aborted, and refused reuse within bound", .{ mode_arg, @errorName(err) });
            return;
        };
        return error.ExpectedRankFailure;
    }
    var probe = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.asBytes(&@as(f32, 1)));
    defer probe.free();
    const one_host = try link.oneHost();
    const returned = collective.lib.api.ncclAllReduce(probe.ptr, probe.ptr, 1, .f32, .sum, collective.comm, stream.handle);
    try check.expect(returned == 0 or (!one_host and returned == cuda.nccl.in_progress), "rank {d}: a collective returns enqueued on one host, or in progress across hosts ({d})", .{ rank, returned });
    try collective.finish(stream);
    for ([_]usize{ 0, 1, 17, 4099, 1_000_003 }, 0..) |count, seq| {
        const host = try gpu.gpa.alloc(f32, @max(1, count));
        defer gpu.gpa.free(host);
        for (host, 0..) |*v, i| v.* = @floatFromInt(i % 127 + rank);
        var values = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(host));
        defer values.free();
        try collective.sum(stream, values.ptr, count);
        try collective.sum(stream, values.ptr, count);
        try collective.finish(stream);
        try values.download(0, std.mem.sliceAsBytes(host));
        for (host[0..count], 0..) |actual, i| try check.expect(actual == @as(f32, @floatFromInt(4 * (i % 127) + 2)), "rank {d}: exact sum at {d}", .{ rank, i });
        // Acknowledge completion before either peer can close or start its next reduction.
        const h: wire.Header = .{ .kind = .ready, .sequence = seq + 2, .arg = count };
        if (rank == 0) {
            try link.sendHeader(h, wire.deadline(30_000));
            try std.testing.expectEqualDeep(h, try link.receiveHeader(wire.deadline(30_000)));
        } else {
            try std.testing.expectEqualDeep(h, try link.receiveHeader(wire.deadline(30_000)));
            try link.sendHeader(h, wire.deadline(30_000));
        }
    }
    for ([_]usize{ 0, 1, 17, 124_160 }) |count| {
        const host = try gpu.gpa.alloc(u16, @max(1, count));
        defer gpu.gpa.free(host);
        for (host, 0..) |*v, i| v.* = @truncate(i * 31 + rank * 0x8001);
        var local = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(host));
        defer local.free();
        var output = try cuda.DeviceBuffer.alloc(gpu.d, @max(2, count * 4));
        defer output.free();
        try collective.gather(stream, local.ptr, output.ptr, count);
        try collective.finish(stream);
        const all = try gpu.gpa.alloc(u16, count * 2);
        defer gpu.gpa.free(all);
        if (count > 0) try output.download(0, std.mem.sliceAsBytes(all));
        for (0..2) |owner| for (0..count) |i| {
            const expected: u16 = @truncate(i * 31 + owner * 0x8001);
            try check.expect(all[owner * count + i] == expected, "rank {d}: gathered BF16 bits at {d}/{d}", .{ rank, owner, i });
        };
    }
    try meet(link, rank, 99);
    check.pass("rank {d}: {s}; {d} MiB of device memory after the collectives", .{ rank, if (one_host) "one host: a collective returns enqueued" else "two hosts: collectives on the nonblocking communicator", (before.free -| (try gpu.ctx.memInfo()).free) >> 20 });
    check.pass("rank {d}: BF16 all-gather preserves every bit and rank column order for empty, scalar, tail and vocabulary-half rows", .{rank});
    check.pass("rank {d}: two-process chained NCCL FP32 sums exact for empty, scalar, tail and 1,000,003 values; fragmented framed control preserved", .{rank});
}

/// Both ranks' acknowledgment at `sequence`, so neither goes on (or closes) while the other is still short of it.
fn meet(link: wire.Link, rank: usize, sequence: u64) !void {
    const h: wire.Header = .{ .kind = .ready, .sequence = sequence };
    if (rank == 0) {
        try link.sendHeader(h, wire.deadline(30_000));
        try std.testing.expectEqualDeep(h, try link.receiveHeader(wire.deadline(30_000)));
    } else {
        try std.testing.expectEqualDeep(h, try link.receiveHeader(wire.deadline(30_000)));
        try link.sendHeader(h, wire.deadline(30_000));
    }
}

fn completeSum(c: *flash.collective.Collective, stream: cuda.Stream, values: u64, count: usize, scatter: bool) !void {
    if (scatter) try c.sumBlock(stream, values, count) else try c.sum(stream, values, count);
    try c.finish(stream);
}

/// The injected peer delay outlives rank 0's 3 s bound; this POSIX-only NCCL test never runs on Windows.
fn outlive() void {
    if (builtin.os.tag == .windows) return;
    const delay: std.c.timespec = .{ .sec = 4, .nsec = 0 };
    _ = std.c.nanosleep(&delay, null);
}
