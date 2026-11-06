//! mcdma.Endpoint's two ranks in one process over MCDMA's libmcdma-fabric on its stub verbs (TF_MCDMA_FABRIC_STUB).
const std = @import("std");
const fabric = @import("fabric");
const sys = @import("sys");
const mcdma = fabric.mcdma;
const collective = fabric.collective;
const words = fabric.words;
const Rdma = fabric.Rdma;
const LinkKind = fabric.rdma.LinkKind;

const gpa = std.testing.allocator;
const hidden = 1024;
const max_rows = 300;
const slot = max_rows * hidden * 4; // f32 rows
const ch_at = 0;
const plan_at = ch_at + collective.Channel.bytes(2, slot);
const pipe_at = plan_at + collective.Channel.bytes(2, 4096);
const pipe_slot = 64 << 10;
const pipe_bytes = collective.Pipe.bytes(pipe_slot);
const wait_at = pipe_at + 2 * pipe_bytes; // a channel no step ever reaches
const flags_at = wait_at + collective.Channel.bytes(2, 64);
const ws_at = flags_at + 16384; // write2Signal lands here, writeSignalFrom 256 KiB on, its sources 512 KiB on
const src_at = ws_at + (512 << 10) + 4096;
const bulk_at = ws_at + (1 << 20) + 4032; // 9 MiB across the Mac's 4 MiB registration pieces
const bulk_len = 9 << 20;
const window = std.mem.alignForward(usize, bulk_at + bulk_len, mcdma.alignment);
const staging = 6 << 20; // staged sources straddle registration pieces too

fn library() ?[:0]const u8 {
    const path = sys.getenv("TF_MCDMA_FABRIC_STUB") orelse return null;
    return if (std.fs.path.isAbsolute(path)) path else null;
}

var next_port: u16 = 0;
/// A fresh block of UDP ports for one pair: a single Thunderbolt device may take two lanes, port and port + 1.
fn ports() u16 {
    if (next_port == 0) next_port = 30000 + @as(u16, @intCast(@mod(std.c.getpid(), 1000))) * 24;
    next_port += 8;
    return next_port;
}

const Rank = struct {
    links: [1]mcdma.Link,
    config: mcdma.Config,
    ep: ?*mcdma.Endpoint = null,
    err: ?anyerror = null,
    ns: u64 = 0,
    hash: [32]u8 = @splat(0),

    fn init(rank: u32, device: [:0]const u8, port: u16, peer_port: u16, link: mcdma.Link) Rank {
        var l = link;
        l.peer = 1 - rank;
        l.device = device;
        l.port = port;
        l.peer_port = peer_port;
        return .{ .links = .{l}, .config = undefined };
    }

    fn open(r: *Rank, rank: u32, connect_timeout_ns: u64) void {
        r.config = .{ .rank = rank, .ranks = 2, .window_bytes = window, .staging_bytes = staging, .links = &r.links, .timeout_ns = 5 * std.time.ns_per_s, .connect_timeout_ns = connect_timeout_ns };
        const t0 = words.nowNs();
        r.ep = mcdma.Endpoint.create(gpa, library().?, r.config) catch |e| blk: {
            r.err = e;
            break :blk null;
        };
        r.ns = words.nowNs() - t0;
    }
};

const base: mcdma.Link = .{ .peer = 1, .device = "roce0", .via = "lo0", .port = 1, .name = "stubpair" };

/// Both ranks connect at once, as two hosts would.
fn pair(ranks: *[2]Rank, devices: [2][:0]const u8, link: mcdma.Link, connect_timeout_ns: u64) !void {
    const p = ports();
    ranks[0] = .init(0, devices[0], p, p + 4, link);
    ranks[1] = .init(1, devices[1], p + 4, p, link);
    const t = try std.Thread.spawn(.{}, Rank.open, .{ &ranks[1], 1, connect_timeout_ns });
    ranks[0].open(0, connect_timeout_ns);
    t.join();
}

fn closePair(ranks: *[2]Rank) void {
    for (ranks) |*r| if (r.ep) |ep| ep.deinit();
}

fn pattern(rank: u32, k: u64, i: usize) u8 {
    return @truncate(rank *% 131 +% k *% 31 +% i *% 7 +% (i >> 9));
}

/// A row's partial depends only on (rank, row id), so either rank recomputes both.
fn partialRow(rank: u32, id: u64, elem: collective.Elem, out: []u8) void {
    var prng: std.Random.DefaultPrng = .init(id * 131 + rank);
    const r = prng.random();
    for (0..hidden) |i| {
        const v = (r.float(f32) - 0.5) * std.math.pow(f32, 2, @floatFromInt(r.intRangeAtMost(i32, -6, 6)));
        switch (elem) {
            .bf16 => std.mem.writeInt(u16, out[2 * i ..][0..2], collective.toBf16(v), .little),
            .f32 => std.mem.writeInt(u32, out[4 * i ..][0..4], @bitCast(v), .little),
        }
    }
}

fn rowsOf(s: u64, ids: []u64) []u64 {
    const counts = [_]usize{ 1, 7, 64, 300, 2, 129, 1, 33 };
    const k = counts[s % counts.len];
    for (ids[0..k], 0..) |*id, j| id.* = (s * 7919 + j * 104729) % 1000;
    return ids[0..k];
}

/// The host reference: p0 + p1 in fp32, then (for bf16) round to nearest even by hand.
fn expectReference(elem: collective.Elem, parts: [2][]const u8, got: []const u8) !void {
    for (0..hidden) |i| {
        var acc: f32 = 0;
        for (parts, 0..) |p, r| {
            const v: f32 = switch (elem) {
                .bf16 => @bitCast(@as(u32, std.mem.readInt(u16, p[2 * i ..][0..2], .little)) << 16),
                .f32 => @bitCast(std.mem.readInt(u32, p[4 * i ..][0..4], .little)),
            };
            acc = if (r == 0) v else acc + v;
        }
        const bits: u32 = @bitCast(acc);
        switch (elem) {
            .f32 => if (std.mem.readInt(u32, got[4 * i ..][0..4], .little) != bits) return error.ReferenceMismatch,
            .bf16 => {
                const low = bits & 0xFFFF;
                var high = bits >> 16;
                if (low > 0x8000 or (low == 0x8000 and high & 1 == 1)) high += 1;
                if (std.mem.readInt(u16, got[2 * i ..][0..2], .little) != @as(u16, @intCast(high))) return error.ReferenceMismatch;
            },
        }
    }
}

fn waitFor(r: Rdma, at: usize, want: u64) !void {
    const deadline = words.nowNs() + 10 * std.time.ns_per_s;
    while (r.local(at) != want) {
        if (words.nowNs() > deadline) return error.PeerTimeout;
        std.atomic.spinLoopHint();
    }
}

/// The gate's suite on one rank, each result against this rank's own recomputation; all-reduce outputs are hashed.
fn suite(ep: *mcdma.Endpoint, rank: u32, want: LinkKind, hash: *[32]u8) !void {
    const r = ep.rdma();
    const peer: u32 = 1 - rank;
    if (r.link(peer) != want or r.link(rank) != .local) return error.WrongLinkKind;
    var word: [8]u8 = undefined;
    if (r.read(peer, 0, &word)) |_| return error.ReadAllowed else |e| if (e != error.AccessDenied) return e;
    if (r.fetchAdd(peer, flags_at, 1)) |_| return error.AtomicAllowed else |e| if (e != error.AccessDenied) return e;
    var plan_ch = collective.Channel.init(r, plan_at, 4096);
    var ch = collective.Channel.init(r, ch_at, slot);
    var scratch: [2][]const u8 = undefined;
    for (0..3) |_| try collective.barrier(&plan_ch, &scratch);
    for (0..20) |s| {
        const root: u32 = @intCast(s % 2);
        var plan: [24]u8 = undefined;
        for (&plan, 0..) |*b, i| b.* = pattern(root, s, i);
        const got = try collective.broadcast(&plan_ch, root, if (rank == root) &plan else "", &scratch);
        if (!std.mem.eql(u8, got, &plan)) return error.BroadcastMismatch;
    }
    var sha: std.crypto.hash.sha2.Sha256 = .init(.{});
    const partial = try gpa.alloc(u8, slot);
    defer gpa.free(partial);
    const out = try gpa.alloc(u8, slot);
    defer gpa.free(out);
    var ids: [max_rows]u64 = undefined;
    var rows_done: usize = 0;
    for ([_]collective.Elem{ .bf16, .f32 }) |elem| {
        const row = hidden * elem.size();
        for (0..16) |s| {
            const rows = rowsOf(s, &ids);
            for (rows, 0..) |id, j| partialRow(rank, id, elem, partial[j * row ..][0..row]);
            const len = rows.len * row;
            try collective.allReduce(&ch, elem, partial[0..len], out[0..len], &scratch);
            var parts: [2][4 * hidden]u8 = undefined;
            var solo: [4 * hidden]u8 = undefined;
            for (rows, 0..) |id, j| {
                for (&parts, 0..) |*p, pr| partialRow(@intCast(pr), id, elem, p[0..row]);
                const views = [2][]const u8{ parts[0][0..row], parts[1][0..row] };
                collective.reduce(elem, &views, solo[0..row]);
                if (!std.mem.eql(u8, solo[0..row], out[j * row ..][0..row])) return error.ReduceMismatch;
                try expectReference(elem, views, out[j * row ..][0..row]);
            }
            sha.update(out[0..len]);
            rows_done += rows.len;
        }
    }
    sha.final(hash);
    var bufs: [2][64 * 9]u8 = undefined;
    for (0..30) |s| {
        var chunks: [2][]const u8 = undefined;
        for (&chunks, &bufs, 0..) |*c, *b, p| {
            const n = (rank * 7 + p * 3 + s) % 9 * 64;
            for (b[0..n], 0..) |*x, i| x.* = pattern(rank, s * 2 + p, i);
            c.* = b[0..n];
        }
        try collective.allToAll(&ch, &chunks, &scratch);
        for (scratch, 0..) |got, p| {
            const n = (p * 7 + rank * 3 + s) % 9 * 64;
            if (got.len != n) return error.SizeMismatch;
            for (got, 0..) |x, i| if (x != pattern(@intCast(p), s * 2 + rank, i)) return error.AllToAllMismatch;
        }
        const root: u32 = @intCast(s % 2);
        var mine: [40]u8 = undefined;
        for (&mine, 0..) |*b, i| b.* = pattern(rank, s + 500, i);
        try collective.gather(&ch, root, &mine, &scratch);
        if (rank == root) for (scratch, 0..) |got, p| {
            if (got.len != mine.len) return error.GatherMismatch;
            for (got, 0..) |x, i| if (x != pattern(@intCast(p), s + 500, i)) return error.GatherMismatch;
        };
    }
    // pipes both ways at once: send, then take the peer's message
    var tx = collective.Pipe.init(r, pipe_at + peer * pipe_bytes, pipe_slot, peer);
    var rx = collective.Pipe.init(r, pipe_at + rank * pipe_bytes, pipe_slot, peer);
    const msg = try gpa.alloc(u8, pipe_slot);
    defer gpa.free(msg);
    for (0..100) |k| {
        const n = (k * 977 + rank * 13) % pipe_slot;
        for (msg[0..n], 0..) |*b, i| b.* = pattern(rank, k + 2000, i);
        try tx.send(msg[0..n]);
        const m = try rx.recv();
        if (m.len != (k * 977 + peer * 13) % pipe_slot) return error.PipeMismatch;
        for (m, 0..) |b, i| if (b != pattern(peer, k + 2000, i)) return error.PipeMismatch;
        try rx.release();
    }
    // a bulk copy each way, staged through the ring in 2 MiB pieces
    const big = try gpa.alloc(u8, bulk_len);
    defer gpa.free(big);
    for (big, 0..) |*b, i| b.* = pattern(rank, 77, i);
    try collective.bulk(r, peer, bulk_at, big, 2 << 20, flags_at + 8, 77);
    try waitFor(r, flags_at + 8, 77);
    for (r.window()[bulk_at..][0..bulk_len], 0..) |b, i| if (b != pattern(peer, 77, i)) return error.BulkMismatch;
    // more signals than any staging word, the last one stays
    for (1..5001) |i| try r.signal(peer, flags_at + 16, i);
    try r.flush();
    try waitFor(r, flags_at + 16, 5000);
    try collective.barrier(&plan_ch, &scratch);
    // the write-and-signal paths: fused on Thunderbolt, write then signal on ConnectX
    const win = r.window();
    for (1..41) |k| {
        const head_len = 1 + k * 13 % 200;
        const body_len = k * 977 % 50000;
        for (msg[0 .. head_len + body_len], 0..) |*b, i| b.* = pattern(rank, k + 3000, i);
        try r.write2Signal(peer, ws_at, msg[0..head_len], msg[head_len..][0..body_len], flags_at + 24, k);
        try waitFor(r, flags_at + 24, k);
        for (win[ws_at..][0 .. head_len + body_len], 0..) |b, i| if (b != pattern(peer, k + 3000, i)) return error.Write2SignalMismatch;
        const len = 1 + k * 2311 % 100000;
        for (win[src_at..][0..len], 0..) |*b, i| b.* = pattern(rank, k + 4000, i);
        try ep.writeSignalFrom(peer, src_at, ws_at + (256 << 10), len, flags_at + 32, k);
        try waitFor(r, flags_at + 32, k);
        for (win[ws_at + (256 << 10) ..][0..len], 0..) |b, i| if (b != pattern(peer, k + 4000, i)) return error.WriteSignalFromMismatch;
        try collective.barrier(&plan_ch, &scratch); // the peer has both: sources and landing places may change
    }
    try r.flush();
    if (rows_done == 0) return error.NothingReduced;
}

const Run = struct {
    ranks: *[2]Rank,
    want: LinkKind,
    errs: [2]?anyerror = .{ null, null },

    fn one(run: *Run, rank: u32) void {
        const r = &run.ranks[rank];
        suite(r.ep.?, rank, run.want, &r.hash) catch |e| {
            run.errs[rank] = e;
        };
    }
};

fn runSuite(devices: [2][:0]const u8, link: mcdma.Link, want: LinkKind) !void {
    if (library() == null) return error.SkipZigTest;
    var ranks: [2]Rank = undefined;
    try pair(&ranks, devices, link, 20 * std.time.ns_per_s);
    defer closePair(&ranks);
    for (ranks) |r| if (r.err) |e| {
        std.debug.print("{s}: create failed: {s}\n", .{ devices[0], @errorName(e) });
        return e;
    };
    var run: Run = .{ .ranks = &ranks, .want = want };
    const t0 = words.nowNs();
    const t = try std.Thread.spawn(.{}, Run.one, .{ &run, 1 });
    run.one(0);
    t.join();
    for (run.errs, 0..) |e, rank| if (e) |err| {
        std.debug.print("{s}/{s} rank {d}: {s}\n", .{ devices[0], devices[1], rank, @errorName(err) });
        return err;
    };
    try std.testing.expectEqualSlices(u8, &ranks[0].hash, &ranks[1].hash);
    std.debug.print("{s}/{s}: kind {s}, path MTU {d}, suite in {d} ms, all-reduce sha256 {x} on both ranks\n", .{ devices[0], devices[1], @tagName(want), link.mtu, (words.nowNs() - t0) / std.time.ns_per_ms, ranks[0].hash[0..8] });
}

test "ConnectX (stub roce0, roce1) at path MTU 1024 and 4096: every collective equals the host reference on both ranks, kind cx5" {
    var link = base;
    link.mtu = 1024;
    link.kind = .cx5;
    try runSuite(.{ "roce0", "roce1" }, link, .cx5);
    link.mtu = 4096;
    link.kind = null;
    try runSuite(.{ "roce0", "roce1" }, link, .cx5);
}

test "Thunderbolt (stub tb0, tb1) runs the same suite as before: kind tb5, the fused write-and-signal" {
    try runSuite(.{ "tb0", "tb1" }, base, .tb5);
}

/// A waiter on a stopped peer gets PeerTimeout; once the peer closes (BYE), the next post is PeerDown with the reason.
fn stoppedPeer(devices: [2][:0]const u8, want: LinkKind) !void {
    if (library() == null) return error.SkipZigTest;
    var ranks: [2]Rank = undefined;
    try pair(&ranks, devices, base, 20 * std.time.ns_per_s);
    defer closePair(&ranks);
    for (ranks) |r| if (r.err) |e| return e;
    const ep = ranks[0].ep.?;
    const r = ep.rdma();
    try std.testing.expectEqual(want, r.link(1));
    var ch = collective.Channel.init(r, wait_at, 64);
    ch.timeout_ns = 300 * std.time.ns_per_ms;
    var out: [2][]const u8 = undefined;
    var t0 = words.nowNs();
    try std.testing.expectError(error.PeerTimeout, ch.exchange(&.{ "", "x" }, &out));
    const waited = words.nowNs() - t0;
    try std.testing.expect(waited >= ch.timeout_ns and waited < 2 * std.time.ns_per_s);
    var buf: [160]u8 = undefined;
    try std.testing.expect(ep.peerStatus(1, &buf) == null);
    ranks[1].ep.?.deinit(); // a clean exit: BYE over the exchange
    ranks[1].ep = null;
    // the stub aborts on a RoCE WRITE to a registration that is gone: nothing is posted until MCDMA heard the BYE
    t0 = words.nowNs();
    const why = while (words.nowNs() - t0 < 2 * std.time.ns_per_s) {
        if (ep.peerStatus(1, &buf)) |reason| break reason;
        words.pause(std.time.ns_per_ms);
    } else return error.StillUp;
    const heard = words.nowNs() - t0;
    try std.testing.expect(heard < std.time.ns_per_s);
    try std.testing.expect(std.mem.indexOf(u8, why, "goodbye") != null);
    // every post after that is PeerDown at once, the first logging the link and the reason
    t0 = words.nowNs();
    try std.testing.expectError(error.PeerDown, r.signal(1, flags_at, 1));
    try std.testing.expectError(error.PeerDown, r.write(1, flags_at, "x"));
    try std.testing.expectError(error.PeerDown, ep.writeSignalFrom(1, src_at, ws_at, 64, flags_at, 2));
    try std.testing.expect(words.nowNs() - t0 < std.time.ns_per_s);
    std.debug.print("{s}: waiter PeerTimeout after {d} ms; peer down heard after {d} ms; posts PeerDown; mcdma: {s}\n", .{ devices[0], waited / std.time.ns_per_ms, heard / std.time.ns_per_ms, why });
}

test "a stopped peer fails by name within a bounded time over ConnectX, with MCDMA's reason" {
    try stoppedPeer(.{ "roce0", "roce1" }, .cx5);
}

test "a stopped peer fails by name within a bounded time over Thunderbolt, as before" {
    try stoppedPeer(.{ "tb0", "tb1" }, .tb5);
}

fn alone(device: [:0]const u8, link: mcdma.Link, connect_timeout_ns: u64) !Rank {
    var r: Rank = .init(0, device, ports(), 1, link);
    r.links[0].peer_port = r.links[0].port + 4;
    r.open(0, connect_timeout_ns);
    if (r.ep) |ep| ep.deinit();
    return r;
}

test "startup failures name their cause: a port down, a scanning GID or a bond on ConnectX, no peer, mismatched kinds" {
    if (library() == null) return error.SkipZigTest;
    const second = std.time.ns_per_s;
    _ = sys.env.setenv("STUB_PORT_DOWN", "roce0", 1);
    var r = try alone("roce0", base, second);
    _ = sys.env.unsetenv("STUB_PORT_DOWN");
    try std.testing.expectEqual(@as(?anyerror, error.OpenFailed), r.err);
    try std.testing.expect(r.ns < second);
    var scan = base;
    scan.gid = -1;
    r = try alone("roce0", scan, second);
    try std.testing.expectEqual(@as(?anyerror, error.OpenFailed), r.err); // MCDMA's DEVICE, logged with the GID hint
    scan.kind = .cx5;
    r = try alone("roce0", scan, second);
    try std.testing.expectEqual(@as(?anyerror, error.NeedGidIndex), r.err);
    r = try alone("roce0+roce1", base, second);
    try std.testing.expectEqual(@as(?anyerror, error.UnsupportedLink), r.err); // MCDMA's UNSUPPORTED at open
    r = try alone("roce0", base, 300 * std.time.ns_per_ms);
    try std.testing.expectEqual(@as(?anyerror, error.ConnectFailed), r.err); // MCDMA's TIMEOUT, logged
    try std.testing.expect(r.ns < 2 * second);
    var ranks: [2]Rank = undefined;
    try pair(&ranks, .{ "roce0", "tb1" }, base, 3 * second);
    defer closePair(&ranks);
    for (ranks) |x| {
        try std.testing.expectEqual(@as(?anyerror, error.ConnectFailed), x.err); // the kinds never meet: TIMEOUT or PEER, logged
        try std.testing.expect(x.ns < 5 * second);
    }
}
