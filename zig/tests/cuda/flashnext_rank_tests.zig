//! Real two-process/two-device NCCL sums and bounded disconnect/timeout behavior over the framed control link.
const std = @import("std");
const cuda = @import("cuda");
const flash = @import("flashnext");
const check = @import("check.zig");
const wire = flash.rank_wire;

pub fn run(gpu: check.Gpu, rank: usize, port: u16, mode: []const u8) !void {
    const missing = std.mem.eql(u8, mode, "missing");
    const started = wire.now();
    if (missing) {
        try std.testing.expectError(error.RankTimeout, wire.Link.open(gpu.io, "127.0.0.1", port, rank, 200));
        try check.expect(wire.now() - started < 3 * std.time.ns_per_s, "missing peer startup is bounded", .{});
        check.pass("rank {d}: missing peer startup timed out within bound", .{rank});
        return;
    }
    const link = try wire.Link.open(gpu.io, "127.0.0.1", port, rank, 30_000);
    defer link.close();
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
    var collective = try flash.collective.Collective.init(link, rank, if (std.mem.eql(u8, mode, "timeout")) 200 else 30_000);
    defer collective.deinit();
    var stream = try cuda.Stream.init(gpu.d, true);
    defer stream.deinit();
    if (std.mem.eql(u8, mode, "eof") or std.mem.eql(u8, mode, "timeout")) {
        if (rank == 1) {
            // A deliberate peer omission: the owner either disconnects or stays alive without entering the collective.
            if (std.mem.eql(u8, mode, "eof")) {
                _ = std.c.shutdown(link.fd, std.posix.SHUT.RDWR);
            } else {
                const delay: std.c.timespec = .{ .sec = 0, .nsec = 600_000_000 };
                _ = std.c.nanosleep(&delay, null);
            }
            check.pass("rank 1: injected {s}", .{mode});
            return;
        }
        var values = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.asBytes(&@as(f32, 1)));
        defer values.free();
        const start = wire.now();
        collective.sum(stream, values.ptr, 1) catch |err| {
            try check.expect(err == error.RankDisconnected or err == error.RankTimeout or err == error.NcclFailed, "failure is surfaced ({s})", .{@errorName(err)});
            try check.expect(collective.failed and collective.comm == null, "failed communicator is aborted and cannot be reused", .{});
            try check.expect(wire.now() - start < 3 * std.time.ns_per_s, "failed collective terminates within bound", .{});
            try std.testing.expectError(error.RankDisconnected, collective.sum(stream, values.ptr, 1));
            check.pass("rank 0: {s} surfaced as {s}, aborted, and refused reuse within bound", .{ mode, @errorName(err) });
            return;
        };
        return error.ExpectedRankFailure;
    }
    for ([_]usize{ 0, 1, 17, 4099, 1_000_003 }, 0..) |count, seq| {
        const host = try gpu.gpa.alloc(f32, @max(1, count));
        defer gpu.gpa.free(host);
        for (host, 0..) |*v, i| v.* = @floatFromInt(i % 127 + rank);
        var values = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(host));
        defer values.free();
        try collective.sum(stream, values.ptr, count);
        try values.download(0, std.mem.sliceAsBytes(host));
        for (host[0..count], 0..) |actual, i| try check.expect(actual == @as(f32, @floatFromInt(2 * (i % 127) + 1)), "rank {d}: exact sum at {d}", .{ rank, i });
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
    check.pass("rank {d}: two-process NCCL FP32 sums exact for empty, scalar, tail and 1,000,003 values; fragmented framed control preserved", .{rank});
}
