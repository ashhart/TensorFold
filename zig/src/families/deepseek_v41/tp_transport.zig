//! Bounded two-rank exchanges over MCDMA or explicitly selected TCP window writes.
const std = @import("std");
const fabric = @import("fabric");
const settings = @import("../flashnext/tp_settings.zig");
const moe = @import("tp_moe.zig");

const Endpoint = union(enum) {
    mcdma: *fabric.mcdma.Endpoint,
    tcp: *fabric.tcp.Endpoint,

    pub fn rdma(e: Endpoint) fabric.Rdma {
        return switch (e) {
            inline else => |p| p.rdma(),
        };
    }

    fn deinit(e: Endpoint) void {
        switch (e) {
            inline else => |p| p.deinit(),
        }
    }
};

fn tcpSettings(gpa: std.mem.Allocator, json: []const u8) !?fabric.tcp.Config {
    const root = try std.json.parseFromSliceLeaky(std.json.Value, gpa, json, .{ .allocate = .alloc_always });
    if (root != .object) return error.BadTpSettings;
    const transport = root.object.get("transport") orelse return null;
    if (transport != .string) return error.BadTpSettings;
    if (std.mem.eql(u8, transport.string, "mcdma")) return null;
    if (!std.mem.eql(u8, transport.string, "tcp")) return error.BadTpSettings;
    const rank = root.object.get("rank") orelse return error.BadTpSettings;
    const host = root.object.get("host") orelse return error.BadTpSettings;
    const port = root.object.get("port") orelse return error.BadTpSettings;
    const session = root.object.get("session") orelse return error.BadTpSettings;
    if (rank != .integer or rank.integer < 0 or rank.integer > 1 or
        host != .string or host.string.len == 0 or port != .integer or port.integer < 1 or port.integer >= 65535 or
        session != .string or session.string.len == 0 or session.string.len > 256) return error.BadTpSettings;
    var connect_ms: u64 = 30_000;
    if (root.object.get("connect_timeout_ms")) |value| {
        if (value != .integer or value.integer < 1 or value.integer > 900_000) return error.BadTpSettings;
        connect_ms = @intCast(value.integer);
    }
    return .{ .rank = @intCast(rank.integer), .host = host.string, .port = @intCast(port.integer), .session = session.string, .connect_timeout_ns = connect_ms * std.time.ns_per_ms };
}

pub const Transport = struct {
    gpa: std.mem.Allocator,
    config_arena: std.heap.ArenaAllocator,
    endpoint: Endpoint,
    channel: fabric.Channel,
    packet: []u8,
    valid: bool = true,
    control_base: usize = 0,
    trace: bool = false,
    trace_last_ns: u64 = 0,

    pub fn init(gpa: std.mem.Allocator, json: []const u8, max_rows: u32) !Transport {
        return initWithControl(gpa, json, max_rows, 0);
    }

    pub fn initWithControl(gpa: std.mem.Allocator, json: []const u8, max_rows: u32, control_bytes: usize) !Transport {
        return initWithFailure(gpa, json, max_rows, control_bytes, null);
    }

    pub fn initWithFailure(gpa: std.mem.Allocator, json: []const u8, max_rows: u32, control_bytes: usize, failed: ?*std.atomic.Value(bool)) !Transport {
        if (max_rows == 0 or max_rows > 4096) return error.InvalidPrefillChunk;
        var arena = std.heap.ArenaAllocator.init(gpa);
        errdefer arena.deinit();
        const size = @as(usize, max_rows) * 7 * 5120 + 16;
        const packet = try gpa.alloc(u8, size);
        errdefer gpa.free(packet);
        const control_base = std.mem.alignForward(usize, fabric.Channel.bytes(2, size), fabric.mcdma.alignment);
        const window_bytes = try std.math.add(usize, control_base, control_bytes);
        const endpoint: Endpoint = if (try tcpSettings(arena.allocator(), json)) |tcp|
            .{ .tcp = try fabric.tcp.Endpoint.create(gpa, tcp, window_bytes, failed) }
        else blk: {
            const config = try settings.read(arena.allocator(), json);
            if (config.rank > 1 or config.links.len != 1 or config.links[0].peer != 1 - config.rank) return error.BadTpSettings;
            const library = try arena.allocator().dupeSentinel(u8, config.library, 0);
            break :blk .{ .mcdma = try fabric.mcdma.Endpoint.create(gpa, library, .{
                .rank = config.rank,
                .ranks = 2,
                .window_bytes = window_bytes,
                .staging_bytes = std.mem.alignForward(usize, size + 4096, fabric.mcdma.alignment),
                .links = config.links,
            }) };
        };
        const trace = if (std.c.getenv("TF_DSV41_TP_TRACE")) |value| std.mem.eql(u8, std.mem.span(value), "1") else false;
        return .{ .gpa = gpa, .config_arena = arena, .endpoint = endpoint, .channel = fabric.Channel.init(endpoint.rdma(), 0, size), .packet = packet, .control_base = control_base, .trace = trace };
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
        const began = if (t.trace) try fabric.words.checkedNowNs() else 0;
        if (send.len != receive.len or send.len > t.packet.len - 16) return error.InvalidTensorParallelMessage;
        std.mem.writeInt(u64, t.packet[0..8], tag, .little);
        std.mem.writeInt(u64, t.packet[8..16], send.len, .little);
        @memcpy(t.packet[16..][0..send.len], send);
        const payload = t.packet[0 .. 16 + send.len];
        const chunks = [_][]const u8{ payload, payload };
        var incoming: [2][]const u8 = undefined;
        const copied = if (t.trace) try fabric.words.checkedNowNs() else 0;
        const step = try t.channel.post(&chunks);
        const posted = if (t.trace) try fabric.words.checkedNowNs() else 0;
        try t.channel.complete(step, &incoming);
        const completed = if (t.trace) try fabric.words.checkedNowNs() else 0;
        const peer = incoming[1 - t.channel.myRank()];
        try validate(peer, tag, receive.len);
        @memcpy(receive, peer[16..]);
        t.valid = true;
        if (t.trace) {
            const ended = try fabric.words.checkedNowNs();
            std.debug.print("{{\"tp_trace_rank\":{d},\"step\":{d},\"tag\":{d},\"bytes\":{d},\"gap_ns\":{d},\"pack_ns\":{d},\"post_ns\":{d},\"complete_ns\":{d},\"copy_ns\":{d}}}\n", .{ t.channel.myRank(), step, tag, send.len, if (t.trace_last_ns == 0) 0 else began - t.trace_last_ns, copied - began, posted - copied, completed - posted, ended - completed });
            t.trace_last_ns = ended;
        }
    }
};

test "TP TCP settings require an explicit transport and complete pair identity" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const gpa = arena.allocator();
    const config = (try tcpSettings(gpa, "{\"transport\":\"tcp\",\"rank\":1,\"host\":\"fe80::1%en3\",\"port\":19750,\"session\":\"test\"}")).?;
    try std.testing.expectEqual(@as(u32, 1), config.rank);
    try std.testing.expectEqualStrings("fe80::1%en3", config.host);
    const slow = (try tcpSettings(gpa, "{\"transport\":\"tcp\",\"rank\":0,\"host\":\"::1\",\"port\":19750,\"session\":\"test\",\"connect_timeout_ms\":600000}")).?;
    try std.testing.expectEqual(@as(u64, 600 * std.time.ns_per_s), slow.connect_timeout_ns);
    try std.testing.expectEqual(@as(u64, 30 * std.time.ns_per_s), slow.timeout_ns);
    try std.testing.expectEqual(@as(?fabric.tcp.Config, null), try tcpSettings(gpa, "{}"));
    for ([_][]const u8{
        "{\"transport\":\"other\"}",
        "{\"transport\":\"tcp\",\"rank\":2,\"host\":\"127.0.0.1\",\"port\":19750,\"session\":\"test\"}",
        "{\"transport\":\"tcp\",\"rank\":0,\"host\":\"127.0.0.1\",\"port\":65535,\"session\":\"test\"}",
        "{\"transport\":\"tcp\",\"rank\":0,\"host\":\"127.0.0.1\",\"port\":19750,\"session\":\"\"}",
        "{\"transport\":\"tcp\",\"rank\":0,\"host\":\"127.0.0.1\",\"port\":19750,\"session\":\"test\",\"connect_timeout_ms\":900001}",
    }) |bad| try std.testing.expectError(error.BadTpSettings, tcpSettings(gpa, bad));
}

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
