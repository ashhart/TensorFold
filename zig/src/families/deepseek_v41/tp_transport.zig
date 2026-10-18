//! Bounded two-rank exchanges over the existing MCDMA registered-window transport.
const std = @import("std");
const fabric = @import("fabric");
const settings = @import("../flashnext/tp_settings.zig");
const moe = @import("tp_moe.zig");

pub const Transport = struct {
    gpa: std.mem.Allocator,
    config_arena: std.heap.ArenaAllocator,
    endpoint: *fabric.mcdma.Endpoint,
    channel: fabric.Channel,
    packet: []u8,
    valid: bool = true,

    pub fn init(gpa: std.mem.Allocator, json: []const u8, max_rows: u32) !Transport {
        if (max_rows == 0 or max_rows > 4096) return error.InvalidPrefillChunk;
        var arena = std.heap.ArenaAllocator.init(gpa);
        errdefer arena.deinit();
        const config = try settings.read(arena.allocator(), json);
        if (config.rank > 1 or config.links.len != 1 or config.links[0].peer != 1 - config.rank)
            return error.BadTpSettings;
        const size = @as(usize, max_rows) * 7 * 5120 + 16;
        const packet = try gpa.alloc(u8, size);
        errdefer gpa.free(packet);
        const library = try arena.allocator().dupeSentinel(u8, config.library, 0);
        const endpoint = try fabric.mcdma.Endpoint.create(gpa, library, .{
            .rank = config.rank,
            .ranks = 2,
            .window_bytes = std.mem.alignForward(usize, fabric.Channel.bytes(2, size), fabric.mcdma.alignment),
            .staging_bytes = std.mem.alignForward(usize, size + 4096, fabric.mcdma.alignment),
            .links = config.links,
        });
        return .{ .gpa = gpa, .config_arena = arena, .endpoint = endpoint, .channel = fabric.Channel.init(endpoint.rdma(), 0, size), .packet = packet };
    }

    pub fn deinit(t: *Transport) void {
        t.endpoint.deinit();
        t.gpa.free(t.packet);
        t.config_arena.deinit();
        t.* = undefined;
    }

    pub fn exchange(t: *Transport) moe.Exchange {
        return .{ .ptr = t, .call = transfer };
    }

    fn transfer(ptr: *anyopaque, tag: u64, send: []const u8, receive: []u8) !void {
        const t: *Transport = @ptrCast(@alignCast(ptr));
        if (!t.valid) return error.InvalidTensorParallelState;
        t.valid = false;
        if (send.len != receive.len or send.len > t.packet.len - 16) return error.InvalidTensorParallelMessage;
        std.mem.writeInt(u64, t.packet[0..8], tag, .little);
        std.mem.writeInt(u64, t.packet[8..16], send.len, .little);
        @memcpy(t.packet[16..][0..send.len], send);
        const payload = t.packet[0 .. 16 + send.len];
        const chunks = [_][]const u8{ payload, payload };
        var incoming: [2][]const u8 = undefined;
        try t.channel.exchange(&chunks, &incoming);
        const peer = incoming[1 - t.channel.myRank()];
        try validate(peer, tag, receive.len);
        @memcpy(receive, peer[16..]);
        t.valid = true;
    }
};

fn validate(packet: []const u8, tag: u64, bytes: usize) !void {
    if (packet.len < 16 or packet.len - 16 != bytes or
        std.mem.readInt(u64, packet[0..8], .little) != tag or
        std.mem.readInt(u64, packet[8..16], .little) != bytes) return error.InvalidTensorParallelMessage;
}

test "TP exchanges reject mismatched phase and malformed payload before copying" {
    var packet: [20]u8 = @splat(0);
    std.mem.writeInt(u64, packet[0..8], 123, .little);
    std.mem.writeInt(u64, packet[8..16], 4, .little);
    try validate(&packet, 123, 4);
    try std.testing.expectError(error.InvalidTensorParallelMessage, validate(&packet, 124, 4));
    try std.testing.expectError(error.InvalidTensorParallelMessage, validate(&packet, 123, 5));
    try std.testing.expectError(error.InvalidTensorParallelMessage, validate(packet[0..8], 123, 4));
    packet[8] = 3;
    try std.testing.expectError(error.InvalidTensorParallelMessage, validate(&packet, 123, 4));
}

fn testTransport(endpoint: fabric.Rdma, packet: []u8) Transport {
    return .{ .gpa = std.testing.allocator, .config_arena = undefined, .endpoint = undefined, .channel = fabric.Channel.init(endpoint, 0, packet.len), .packet = packet };
}

test "TP exchanges carry consecutive peer payloads and reject phase disagreement on both ranks" {
    const gpa = std.testing.allocator;
    const cluster = try fabric.fake.Cluster.init(gpa, 2, fabric.Channel.bytes(2, 64), .immediate, null);
    defer cluster.deinit();
    var packets: [2][64]u8 = undefined;
    var transports = [_]Transport{ testTransport(cluster.endpoint(0), &packets[0]), testTransport(cluster.endpoint(1), &packets[1]) };
    const Worker = struct {
        transport: *Transport,
        failure: ?anyerror = null,
        fn run(w: *@This()) void {
            w.exercise() catch |e| {
                w.failure = e;
            };
        }
        fn exercise(w: *@This()) !void {
            const rank = w.transport.channel.myRank();
            var send: [8]u8 = undefined;
            var receive: [8]u8 = undefined;
            for (0..32) |step| {
                std.mem.writeInt(u64, &send, step * 2 + rank, .little);
                try Transport.transfer(w.transport, step, &send, &receive);
                try std.testing.expectEqual(@as(u64, step * 2 + 1 - rank), std.mem.readInt(u64, &receive, .little));
            }
            try std.testing.expectError(error.InvalidTensorParallelMessage, Transport.transfer(w.transport, 100 + rank, &send, &receive));
            try std.testing.expectError(error.InvalidTensorParallelState, Transport.transfer(w.transport, 102, &send, &receive));
        }
    };
    var workers = [_]Worker{ .{ .transport = &transports[0] }, .{ .transport = &transports[1] } };
    const peer = try std.Thread.spawn(.{}, Worker.run, .{&workers[1]});
    workers[0].run();
    peer.join();
    for (workers) |worker| if (worker.failure) |err| return err;
}

test "TP exchanges time out without a peer and cannot reuse that channel" {
    const cluster = try fabric.fake.Cluster.init(std.testing.allocator, 2, fabric.Channel.bytes(2, 64), .immediate, null);
    defer cluster.deinit();
    var packet: [64]u8 = undefined;
    var transport = testTransport(cluster.endpoint(0), &packet);
    transport.channel.timeout_ns = std.time.ns_per_ms;
    var receive: [1]u8 = undefined;
    try std.testing.expectError(error.PeerTimeout, Transport.transfer(&transport, 0, &.{42}, &receive));
    try std.testing.expectError(error.InvalidTensorParallelState, Transport.transfer(&transport, 0, &.{42}, &receive));
}
