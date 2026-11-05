//! mcdma.zig over a fake libmcdma-fabric: the MTU and kind checks, the kept statuses and each link's exact posts.
const std = @import("std");
const abi = @import("mcdma_abi.zig");
const rma = @import("rdma.zig");
const words = @import("words.zig");
const mcdma = @import("mcdma.zig");
const Endpoint = mcdma.Endpoint;
const Link = mcdma.Link;
const Config = mcdma.Config;
const Error = mcdma.Error;
const alignment = mcdma.alignment;
const roce = mcdma.roce;
const statusName = mcdma.statusName;

/// libmcdma-fabric as the tests below drive it: one peer, every post recorded in order, each status set per test.
const Fake = struct {
    const Call = union(enum) {
        write: [3]u64, // local, offset, length
        signal: [2]u64, // offset, value
        write_signal: [5]u64, // local, offset, length, flag, value
        flush: u64, // timeout
    };
    var link: c_int = 2;
    var open_status: c_int = 0;
    var connect_status: c_int = 0;
    var post_status: c_int = 0;
    var flush_status: c_int = 0;
    var why: []const u8 = "";
    var gid: c_int = 0;
    var mtu: c_int = 0;
    var length: u64 = 0;
    var opens: usize = 0;
    var closes: usize = 0;
    var connects: usize = 0;
    var lists: usize = 0;
    var disconnects: usize = 0;
    var calls: [16]Call = undefined;
    var n: usize = 0;
    var context: u8 = 0;
    var peer: u8 = 0;
    var tb_peer: u8 = 0; // with `mixed`: the peer a via on en4 reaches, over Thunderbolt; every other via is ConnectX
    var mixed: bool = false;

    fn reset(kind: c_int) void {
        link = kind;
        mixed = false;
        open_status = 0;
        connect_status = 0;
        post_status = 0;
        flush_status = 0;
        why = "";
        gid = 0;
        mtu = 0;
        opens = 0;
        closes = 0;
        connects = 0;
        lists = 0;
        disconnects = 0;
        n = 0;
    }

    fn record(call: Call) void {
        if (n < calls.len) calls[n] = call;
        n += 1;
    }

    fn version() callconv(.c) u32 {
        return 1;
    }
    fn open(_: [*:0]const u8, g: c_int, m: c_int, _: *anyopaque, len: usize, _: c_int, _: u64, _: u32, out: *?*abi.Context) callconv(.c) c_int {
        opens += 1;
        gid = g;
        mtu = m;
        length = len;
        if (open_status == 0) out.* = @ptrCast(&context);
        return open_status;
    }
    fn connect(_: *abi.Context, via: [*:0]const u8, _: c_int, _: c_int, _: [*:0]const u8, _: u64, out: *?*abi.Peer) callconv(.c) c_int {
        _ = @atomicRmw(usize, &connects, .Add, 1, .monotonic); // one thread per port
        if (connect_status == 0) out.* = if (mixed and std.mem.startsWith(u8, std.mem.span(via), "en4")) @ptrCast(&tb_peer) else @ptrCast(&peer);
        return connect_status;
    }
    fn connectLinks(c: *abi.Context, via: [*:0]const u8, _: [*]const u16, _: [*]const u16, _: c_uint, name: [*:0]const u8, timeout: u64, out: *?*abi.Peer) callconv(.c) c_int {
        lists += 1;
        return connect(c, via, 0, 0, name, timeout, out);
    }
    fn maxLinks() callconv(.c) c_uint {
        return 8;
    }
    fn kindOf(p: *const abi.Peer) callconv(.c) c_int {
        if (mixed) return if (@intFromPtr(p) == @intFromPtr(&tb_peer)) abi.thunderbolt else roce;
        return link;
    }
    fn peerLength(_: *const abi.Peer) callconv(.c) u64 {
        return length;
    }
    fn write(_: *abi.Peer, local: u64, offset: u64, len: u64) callconv(.c) c_int {
        record(.{ .write = .{ local, offset, len } });
        return post_status;
    }
    fn signal(_: *abi.Peer, offset: u64, value: u64) callconv(.c) c_int {
        record(.{ .signal = .{ offset, value } });
        return post_status;
    }
    fn writeSignal(_: *abi.Peer, local: u64, offset: u64, len: u64, flag: u64, value: u64) callconv(.c) c_int {
        record(.{ .write_signal = .{ local, offset, len, flag, value } });
        return post_status;
    }
    fn flush(_: *abi.Peer, timeout: u64) callconv(.c) c_int {
        record(.{ .flush = timeout });
        return flush_status;
    }
    fn disconnect(p: *?*abi.Peer) callconv(.c) void {
        disconnects += 1;
        p.* = null;
    }
    fn close(c: *?*abi.Context) callconv(.c) void {
        closes += 1;
        c.* = null;
    }
    fn peerStatus(_: *const abi.Peer, buf: ?[*]u8, size: usize) callconv(.c) c_int {
        if (why.len == 0) return 0;
        const k = @min(why.len, size - 1);
        @memcpy(buf.?[0..k], why[0..k]);
        buf.?[k] = 0;
        return 5;
    }

    fn library(fused: bool) abi.Library {
        return .{
            .handle = undefined,
            .api = .{ .abi = version, .open = open, .connect = connect, .link = kindOf, .peer_length = peerLength, .write = write, .signal = signal, .flush = flush, .disconnect = disconnect, .close = close },
            .write_signal = if (fused) writeSignal else null,
            .connect_links = connectLinks,
            .max_links = maxLinks,
        };
    }

    fn endpoint(link_spec: Link) Error!*Endpoint {
        return Endpoint.createWith(std.testing.allocator, library(true), peerStatus, config(&.{link_spec}));
    }

    fn config(links: []const Link) Config {
        return .{ .rank = 0, .ranks = 2, .window_bytes = 4 * alignment, .staging_bytes = 2 * alignment, .links = links, .timeout_ns = std.time.ns_per_s };
    }

    const cx: Link = .{ .peer = 1, .device = "rdma_mcrdma1", .via = "en0/192.0.2.2", .port = 7490, .name = "pair" };
    const staging = 4 * alignment + 2 * alignment; // rank 0's staging ring for peer 1
};

test "a path MTU of 1024, 2048 or 4096 reaches open, 4096 by default; any other is refused before opening" {
    for ([_]c_int{ 1024, 2048, 4096 }) |mtu| {
        Fake.reset(roce);
        var link = Fake.cx;
        link.mtu = mtu;
        const ep = try Fake.endpoint(link);
        ep.deinit();
        try std.testing.expectEqual(mtu, Fake.mtu);
        try std.testing.expectEqual(@as(usize, 1), Fake.closes);
    }
    Fake.reset(abi.thunderbolt);
    const tb = try Fake.endpoint(.{ .peer = 1, .device = "rdma_en4", .via = "en4/192.0.2.2", .port = 7490, .name = "pair" });
    tb.deinit();
    try std.testing.expectEqual(@as(c_int, 4096), Fake.mtu);
    try std.testing.expectEqual(@as(c_int, 1), Fake.gid);
    for ([_]c_int{ 0, 512, 1500, 9000, -1 }) |mtu| {
        Fake.reset(roce);
        var link = Fake.cx;
        link.mtu = mtu;
        try std.testing.expectError(error.InvalidConfig, Fake.endpoint(link));
        try std.testing.expectEqual(@as(usize, 0), Fake.opens);
    }
}

test "after connect a Thunderbolt link is tb5, a ConnectX link cx5, and any other kind is refused with every port closed once" {
    Fake.reset(abi.thunderbolt);
    const tb = try Fake.endpoint(Fake.cx);
    try std.testing.expectEqual(rma.LinkKind.tb5, tb.rdma().link(1));
    try std.testing.expectEqual(rma.LinkKind.local, tb.rdma().link(0));
    tb.deinit();
    Fake.reset(roce);
    const cx = try Fake.endpoint(Fake.cx);
    try std.testing.expectEqual(rma.LinkKind.cx5, cx.rdma().link(1));
    try std.testing.expectEqual(rma.LinkKind.local, cx.rdma().link(0));
    try std.testing.expectEqual(rma.LinkKind.tb5, cx.rdma().link(7)); // no such peer: as before
    cx.deinit();
    try std.testing.expectEqual(@as(usize, 1), Fake.disconnects);
    try std.testing.expectEqual(@as(usize, 1), Fake.closes);
    for ([_]c_int{ 0, 3, -1 }) |other| {
        Fake.reset(other);
        try std.testing.expectError(error.UnsupportedLink, Fake.endpoint(Fake.cx));
        try std.testing.expectEqual(@as(usize, 1), Fake.disconnects);
        try std.testing.expectEqual(@as(usize, 1), Fake.closes);
    }
}

test "ConnectX refuses port lists, scanning GIDs and bonds by name, before opening when the link says cx5" {
    // undeclared: a single device with a lane list connects through connect_links, then is refused as ConnectX
    Fake.reset(roce);
    var lanes = Fake.cx;
    lanes.ports = &.{7490};
    try std.testing.expectError(error.UnsupportedLink, Fake.endpoint(lanes));
    try std.testing.expectEqual(@as(usize, 1), Fake.lists);
    try std.testing.expectEqual(@as(usize, 1), Fake.closes);
    // the same lane list stays a Thunderbolt link's to use
    Fake.reset(abi.thunderbolt);
    (try Fake.endpoint(lanes)).deinit();
    // declared cx5: refused before anything opens
    for ([_]struct { Link, anyerror }{
        .{ .{ .peer = 1, .device = "rdma_mcrdma1", .via = "en0", .port = 7490, .name = "pair", .gid = -1, .kind = .cx5 }, error.NeedGidIndex },
        .{ .{ .peer = 1, .device = "rdma_mcrdma0+rdma_mcrdma1", .via = "en0", .port = 7490, .name = "pair", .kind = .cx5 }, error.UnsupportedLink },
        .{ .{ .peer = 1, .device = "rdma_mcrdma1", .via = "en0", .port = 7490, .name = "pair", .ports = &.{7490}, .kind = .cx5 }, error.UnsupportedLink },
        .{ .{ .peer = 1, .device = "rdma_mcrdma1", .via = "en0", .port = 7490, .name = "pair", .peer_ports = &.{7491}, .kind = .cx5 }, error.UnsupportedLink },
        .{ .{ .peer = 1, .device = "rdma_mcrdma1", .via = "en0", .port = 7490, .name = "pair", .kind = .nccl }, error.InvalidConfig },
        .{ .{ .peer = 1, .device = "rdma_mcrdma1", .via = "en0", .port = 7490, .name = "pair", .kind = .local }, error.InvalidConfig },
    }) |case| {
        Fake.reset(roce);
        try std.testing.expectError(case[1], Fake.endpoint(case[0]));
        try std.testing.expectEqual(@as(usize, 0), Fake.opens);
    }
    // a declared kind refuses the other kind after connect
    var want_cx = Fake.cx;
    want_cx.kind = .cx5;
    Fake.reset(abi.thunderbolt);
    try std.testing.expectError(error.UnsupportedLink, Fake.endpoint(want_cx));
    try std.testing.expectEqual(@as(usize, 1), Fake.closes);
    Fake.reset(roce);
    (try Fake.endpoint(want_cx)).deinit();
    var want_tb = Fake.cx;
    want_tb.kind = .tb5;
    try std.testing.expectError(error.UnsupportedLink, Fake.endpoint(want_tb));
    Fake.reset(abi.thunderbolt);
    (try Fake.endpoint(want_tb)).deinit();
    // a Thunderbolt link may still scan for its GID
    var scan = Fake.cx;
    scan.gid = -1;
    (try Fake.endpoint(scan)).deinit();
    try std.testing.expectEqual(@as(c_int, -1), Fake.gid);
}

test "open and connect keep MCDMA's status: UNSUPPORTED opens are UnsupportedLink, others OpenFailed, connects ConnectFailed" {
    for ([_]struct { c_int, anyerror }{ .{ 2, error.UnsupportedLink }, .{ 4, error.OpenFailed }, .{ 1, error.OpenFailed }, .{ 3, error.OpenFailed } }) |case| {
        Fake.reset(roce);
        Fake.open_status = case[0];
        try std.testing.expectError(case[1], Fake.endpoint(Fake.cx));
        try std.testing.expectEqual(@as(usize, 0), Fake.connects);
        try std.testing.expectEqual(@as(usize, 0), Fake.closes); // nothing opened, nothing to close
    }
    for ([_]c_int{ 5, 6, 4 }) |status| {
        Fake.reset(roce);
        Fake.connect_status = status;
        try std.testing.expectError(error.ConnectFailed, Fake.endpoint(Fake.cx));
        try std.testing.expectEqual(@as(usize, 0), Fake.disconnects);
        try std.testing.expectEqual(@as(usize, 1), Fake.closes);
    }
    try std.testing.expectEqualStrings("PEER", statusName(5));
    try std.testing.expectEqualStrings("TIMEOUT", statusName(6));
    try std.testing.expectEqualStrings("UNSUPPORTED", statusName(2));
    try std.testing.expectEqualStrings("unknown", statusName(99));
}

/// The same posts on either link; returns the calls the library saw.
fn postEach(ep: *Endpoint) ![]const Fake.Call {
    const r = ep.rdma();
    const win = r.window();
    for (win[1024..][0..64], 0..) |*b, i| b.* = @intCast(i);
    Fake.n = 0;
    try r.write(1, 128, &@as([100]u8, @splat(1)));
    try r.write2(1, 4096, &@as([10]u8, @splat(2)), &@as([20]u8, @splat(3)));
    try r.signal(1, 8, 7);
    try r.write2Signal(1, 8192, "head!", "body..", 16, 9);
    try ep.writeSignalFrom(1, 1024, 2048, 64, 24, 3);
    try ep.writeSignalFrom(1, 1024, 2048, 0, 32, 4); // no bytes: the signal alone, on either link
    try r.flush();
    return Fake.calls[0..Fake.n];
}

test "Thunderbolt posts exactly as before: fused write-and-signal after the library's room, staged where it was" {
    Fake.reset(abi.thunderbolt);
    const ep = try Fake.endpoint(Fake.cx);
    defer ep.deinit();
    const calls = try postEach(ep);
    const s = Fake.staging;
    const want = [_]Fake.Call{
        .{ .write = .{ s, 128, 100 } },
        .{ .write = .{ s + 128, 4096, 30 } },
        .{ .signal = .{ 8, 7 } },
        .{ .write_signal = .{ s + 192 + abi.ws_room, 8192, 11, 16, 9 } },
        .{ .write_signal = .{ 1024, 2048, 64, 24, 3 } },
        .{ .signal = .{ 32, 4 } },
    };
    try std.testing.expectEqual(want.len + 1, calls.len);
    for (want, calls[0..want.len]) |w, c| try std.testing.expectEqualDeep(w, c);
    try std.testing.expect(calls[want.len].flush > 0 and calls[want.len].flush <= std.time.ns_per_s);
    try std.testing.expectEqualStrings("head!body..", ep.memory[s + 192 + abi.ws_room ..][0..11]);
}

test "ConnectX never asks for the fused message: write2 then signal, and writeSignalFrom's write then its signal" {
    Fake.reset(roce);
    const ep = try Fake.endpoint(Fake.cx);
    defer ep.deinit();
    const calls = try postEach(ep);
    const s = Fake.staging;
    const want = [_]Fake.Call{
        .{ .write = .{ s, 128, 100 } },
        .{ .write = .{ s + 128, 4096, 30 } },
        .{ .signal = .{ 8, 7 } },
        .{ .write = .{ s + 192, 8192, 11 } },
        .{ .signal = .{ 16, 9 } },
        .{ .write = .{ 1024, 2048, 64 } },
        .{ .signal = .{ 24, 3 } },
        .{ .signal = .{ 32, 4 } },
    };
    try std.testing.expectEqual(want.len + 1, calls.len);
    for (want, calls[0..want.len]) |w, c| try std.testing.expectEqualDeep(w, c);
    try std.testing.expect(calls[want.len] == .flush);
    try std.testing.expectEqualStrings("head!body..", ep.memory[s + 192 ..][0..11]);
    // the fallback keeps the fused path's bounds: no room before `local`, a misaligned flag
    try std.testing.expectError(error.OutOfBounds, ep.writeSignalFrom(1, 32, 2048, 64, 24, 3));
    try std.testing.expectError(error.Unaligned, ep.writeSignalFrom(1, 1024, 2048, 64, 20, 3));
    // a library without the fused symbol leaves write2_signal out of the vtable, as before
    const old = try Endpoint.createWith(std.testing.allocator, Fake.library(false), null, Fake.config(&.{Fake.cx}));
    defer old.deinit();
    try std.testing.expect(old.rdma().vtable.write2_signal == null);
}

test "write2Signal refuses a bad flag before posting anything, over ConnectX as over Thunderbolt" {
    for ([_]c_int{ roce, abi.thunderbolt }) |link| {
        Fake.reset(link);
        const ep = try Fake.endpoint(Fake.cx);
        defer ep.deinit();
        const r = ep.rdma();
        const end = ep.config.window_bytes;
        Fake.n = 0;
        try std.testing.expectError(error.Unaligned, r.write2Signal(1, 8192, "head!", "body..", 20, 9));
        try std.testing.expectError(error.OutOfBounds, r.write2Signal(1, 8192, "head!", "body..", end, 9));
        try std.testing.expectError(error.OutOfBounds, r.write2Signal(1, 8192, "head!", "body..", end - 4, 9));
        try std.testing.expectError(error.OutOfBounds, r.write2Signal(1, end - 4, "head!", "body..", 16, 9));
        // and the same for the fallbacks ConnectX shares: an oversized write, and the local peer
        const big: [3 * alignment]u8 = @splat(5);
        try std.testing.expectError(error.Unaligned, r.write2Signal(1, 0, "head!", &big, 20, 9));
        try r.flush();
        try std.testing.expectEqual(@as(usize, 0), Fake.n); // nothing posted, and nothing pending to flush
        @memset(ep.memory[8192..][0..11], 0);
        try std.testing.expectError(error.Unaligned, r.write2Signal(0, 8192, "head!", "body..", 20, 9));
        try std.testing.expect(std.mem.allEqual(u8, ep.memory[8192..][0..11], 0)); // nothing written to the local window
    }
}

test "one endpoint with a Thunderbolt peer and a ConnectX peer picks each port's path and kind" {
    Fake.reset(abi.thunderbolt);
    Fake.mixed = true;
    defer Fake.mixed = false;
    const links = [_]Link{
        .{ .peer = 1, .device = "rdma_en4", .via = "en4/192.0.2.2", .port = 7490, .name = "tb" },
        .{ .peer = 2, .device = "rdma_mcrdma1", .via = "en0/192.0.2.3", .port = 7492, .name = "cx" },
    };
    {
        const ep = try Endpoint.createWith(std.testing.allocator, Fake.library(true), Fake.peerStatus, .{ .rank = 0, .ranks = 3, .window_bytes = 4 * alignment, .staging_bytes = 2 * alignment, .links = &links, .timeout_ns = std.time.ns_per_s });
        defer ep.deinit();
        const r = ep.rdma();
        try std.testing.expectEqual(rma.LinkKind.tb5, r.link(1));
        try std.testing.expectEqual(rma.LinkKind.cx5, r.link(2));
        try std.testing.expectEqual(rma.LinkKind.local, r.link(0));
        Fake.n = 0;
        try r.write2Signal(1, 8192, "head!", "body..", 16, 9);
        try r.write2Signal(2, 8192, "head!", "body..", 16, 9);
        try ep.writeSignalFrom(1, 1024, 2048, 64, 24, 3);
        try ep.writeSignalFrom(2, 1024, 2048, 64, 24, 3);
        const s1 = 4 * alignment + 1 * 2 * alignment; // rank 0's staging ring for peer 1, then for peer 2
        const s2 = 4 * alignment + 2 * 2 * alignment;
        const want = [_]Fake.Call{
            .{ .write_signal = .{ s1 + abi.ws_room, 8192, 11, 16, 9 } },
            .{ .write = .{ s2, 8192, 11 } },
            .{ .signal = .{ 16, 9 } },
            .{ .write_signal = .{ 1024, 2048, 64, 24, 3 } },
            .{ .write = .{ 1024, 2048, 64 } },
            .{ .signal = .{ 24, 3 } },
        };
        try std.testing.expectEqual(want.len, Fake.n);
        for (want, Fake.calls[0..want.len]) |w, c| try std.testing.expectEqualDeep(w, c);
    }
    try std.testing.expectEqual(@as(usize, 2), Fake.connects);
    try std.testing.expectEqual(@as(usize, 2), Fake.disconnects);
    try std.testing.expectEqual(@as(usize, 2), Fake.closes);
}

test "a peer that went down fails by name within the timeout, and MCDMA's reason is kept" {
    const collective = @import("collective.zig");
    Fake.reset(roce);
    const ep = try Fake.endpoint(Fake.cx);
    defer ep.deinit();
    var buf: [128]u8 = undefined;
    try std.testing.expect(ep.peerStatus(1, &buf) == null); // up
    try std.testing.expect(ep.peerStatus(5, &buf) == null); // no such peer
    const r = ep.rdma();
    // a waiter on a stopped peer: the channel's own timeout, by name
    var ch = collective.Channel.init(r, 0, 4096);
    ch.timeout_ns = 100 * std.time.ns_per_ms;
    var out: [2][]const u8 = undefined;
    const t0 = words.nowNs();
    try std.testing.expectError(error.PeerTimeout, ch.exchange(&.{ "a", "b" }, &out));
    const waited = words.nowNs() - t0;
    try std.testing.expect(waited >= ch.timeout_ns and waited < 2 * std.time.ns_per_s);
    // the library has heard the peer go: every later post is PeerDown at once, the first one logging the reason
    Fake.why = "pair: the other end said goodbye";
    Fake.post_status = 5;
    const t1 = words.nowNs();
    try std.testing.expectError(error.PeerDown, r.signal(1, 8, 1));
    try std.testing.expectError(error.PeerDown, r.write(1, 64, "x"));
    try std.testing.expectError(error.PeerDown, ep.writeSignalFrom(1, 1024, 0, 8, 16, 1));
    try std.testing.expect(words.nowNs() - t1 < std.time.ns_per_s);
    try std.testing.expect(ep.ports[0].down);
    try std.testing.expectEqualStrings("pair: the other end said goodbye", ep.peerStatus(1, &buf).?);
    var tiny: [8]u8 = undefined;
    try std.testing.expectEqualStrings("pair: t", ep.peerStatus(1, &tiny).?);
    // a flush the peer never answers: MCDMA's TIMEOUT, within this endpoint's timeout
    Fake.post_status = 0;
    Fake.flush_status = 6;
    try r.signal(1, 8, 2);
    try std.testing.expectError(error.Timeout, r.flush());
    try std.testing.expect(Fake.calls[Fake.n - 1].flush <= std.time.ns_per_s);
    // without the optional symbol the error is the same, with no reason
    Fake.reset(roce);
    Fake.why = "gone";
    const plain = try Endpoint.createWith(std.testing.allocator, Fake.library(true), null, Fake.config(&.{Fake.cx}));
    defer plain.deinit();
    Fake.post_status = 5;
    try std.testing.expectError(error.PeerDown, plain.rdma().signal(1, 8, 1));
    try std.testing.expect(plain.peerStatus(1, &buf) == null);
}
