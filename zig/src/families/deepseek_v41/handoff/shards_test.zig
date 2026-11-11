//! Two KV_SPLIT producers on two fake links, pulled in parallel by `prefill_source.Remote` into one landing.
const std = @import("std");
const fabric = @import("fabric");
const plan = @import("plan.zig");
const ex = @import("export.zig");
const im = @import("import.zig");
const metal = @import("metal.zig");
const p2 = @import("p2.zig");
const landing = @import("landing.zig");
const s = @import("synth.zig");

const producer = fabric.producer;
const ps = fabric.prefill_source;
const half = 4 << 20;
const model = "dsv41-test";

fn target(arena: std.mem.Allocator, capacity: u32) !metal.Packed {
    const probe: metal.Packed = .{ .live = &.{}, .storage = &.{}, .layers = 43, .capacity = capacity };
    return .{ .live = try arena.alloc(u8, probe.liveBytes()), .storage = try arena.alloc(u8, metal.Packed.storageBytes(capacity)), .layers = 43, .capacity = capacity };
}

/// One shard's producer: its own fake link, table and serving thread; `publish` records the export a trigger asked for.
const Peer = struct {
    a: std.mem.Allocator,
    link: fabric.link_fake.Link = undefined,
    table: producer.Table = undefined,
    serving: producer.Serving = undefined,
    exp: producer.Export = undefined,
    published: bool = false,
    stop: std.atomic.Value(bool) = .init(false),
    daemons: std.Thread = undefined,
    service: std.Thread = undefined,
    mac: fabric.mailbox.Mailbox = undefined,
    peer: fabric.mailbox.Mailbox = undefined,
    client: fabric.mailbox.Client = undefined,

    fn start(p: *Peer, shard: u32, shards: u32) !void {
        p.link = try fabric.link_fake.Link.init(p.a, half, half, true, .immediate);
        p.link.up();
        p.mac = p.link.box(0);
        p.peer = p.link.box(1);
        p.table = .{ .gpa = p.a, .ttl_ns = 60 * std.time.ns_per_s, .clock = fabric.words.nowNs };
        p.serving.init(p.a, &p.peer, fabric.words.native, &p.table, model);
        p.serving.responder.tp_rank = shard;
        p.serving.responder.tp_size = shards;
        p.daemons = try std.Thread.spawn(.{}, fabric.link_fake.Link.run, .{ &p.link, &p.stop });
        p.service = try std.Thread.spawn(.{}, producer.Serving.run, .{&p.serving});
        p.client = try fabric.mailbox.Client.init(&p.mac, fabric.words.native, 30 * std.time.ns_per_s);
    }

    fn publish(p: *Peer, arena: std.mem.Allocator, w: s.World, shape: plan.Shape, id: [16]u8, tokens: []const u32, fail: bool) !void {
        if (fail) return p.table.add(id, "test", .{ .failed = "the shard's export failed" });
        const built = try p2.layers(arena, try ex.build(arena, shape, w.cuda));
        p.exp = .{ .handoff = id, .tokens = tokens, .first_token = 0, .tokens_per_row = 1, .layers = built.layers, .sources = built.sources };
        try p.table.add(id, "test", .{ .ready = &p.exp });
        p.published = true;
    }

    fn finish(p: *Peer) void {
        p.stop.store(true, .release);
        p.serving.stop.store(true, .release);
        p.daemons.join();
        p.service.join();
        if (p.published and p.exp.frames.len > 0) p.a.free(p.exp.frames);
        p.serving.deinit();
        p.table.deinit();
        p.link.deinit();
    }
};

/// The trigger of the test: the pull's id arrives here, and both producers record their shard under it.
const Both = struct {
    arena: std.mem.Allocator,
    w: s.World,
    upto: u64,
    peers: *[2]Peer,
    fail: ?u32,
    asked: u32 = 0,

    fn start(ptr: *anyopaque, id: [16]u8, tokens: []const u32) anyerror!void {
        const b: *Both = @ptrCast(@alignCast(ptr));
        b.asked += 1;
        for (b.peers, 0..) |*p, k| {
            const shape: plan.Shape = .{ .upto = b.upto, .shard = @intCast(k), .shards = 2, .split = true };
            try p.publish(b.arena, b.w, shape, id, tokens[0..b.upto], b.fail == @as(u32, @intCast(k)));
        }
    }
};

fn pullBoth(upto: u64, fail: ?u32) !void {
    const a = std.testing.allocator;
    var arena_state: std.heap.ArenaAllocator = .init(a);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const w = try s.make(arena, .{ .upto = upto, .world = 2, .split = true, .seed = upto });
    const prompt = try arena.alloc(u32, upto + 1);
    for (prompt, 0..) |*t, i| t.* = @intCast(i * 7 + 1);
    var peers: [2]Peer = .{ .{ .a = a }, .{ .a = a } };
    var started: usize = 0;
    defer for (peers[0..started]) |*p| p.finish();
    for (&peers, 0..) |*p, k| {
        try p.start(@intCast(k), 2);
        started += 1;
    }
    var both: Both = .{ .arena = arena, .w = w, .upto = upto, .peers = &peers, .fail = fail };
    const links = [1]*fabric.mailbox.Client{&peers[1].client};
    var remote: ps.Remote = .{ .gpa = a, .policy = .{ .enabled = true, .min_prefix = 0, .tail = 1 }, .client = &peers[0].client, .peers = &links, .trigger = .{ .ptr = &both, .start_fn = Both.start }, .model = model, .protocol = 2 };
    const offer = remote.source().offer(prompt, 0).?;
    try std.testing.expectEqual(upto, offer.upto);

    const via = try target(arena, 2048);
    var land: landing.Landing = undefined;
    try land.initShards(a, upto, 2, .{ .@"packed" = via });
    defer land.deinit();
    if (fail) |_| {
        try std.testing.expectError(error.Refused, remote.source().fill(prompt, offer, land.state()));
        try std.testing.expectEqualStrings("the shard's export failed", remote.reason.text());
        return;
    }
    try remote.source().fill(prompt, offer, land.state());
    try std.testing.expectEqual(@as(u32, 1), both.asked);
    _ = try land.complete();

    // the same shards' parts landed directly give the same bytes
    const direct = try target(arena, 2048);
    var imp = try im.Import.init(arena, .{ .upto = upto, .shards = 2, .split = true }, .{ .@"packed" = direct }, .{ .drafter = false });
    var buf: [40]u8 = undefined;
    for (0..2) |k| for (try ex.build(arena, .{ .upto = upto, .shard = @intCast(k), .shards = 2, .split = true }, w.cuda)) |p| {
        try imp.put(@intCast(k), p.spec.layer, p.spec.name(&buf), p.spec.dtype, p.spec.rows, p.spec.cols, p.bytes);
    };
    _ = try imp.complete();
    try std.testing.expectEqualSlices(u8, direct.live, via.live);
    try std.testing.expectEqualSlices(u8, direct.storage, via.storage);
}

test "a Remote pulls a KV_SPLIT export from two links at once, landing the direct import's bytes" {
    try pullBoth(300, null);
    try pullBoth(1301, null);
}

test "a shard that failed to export fails the whole pull with its producer's reason" {
    try pullBoth(300, 1);
}

test "a sink that takes one producer only is refused when the export is split, before any link is used" {
    const a = std.testing.allocator;
    const Nop = struct {
        fn start(_: *anyopaque, _: [16]u8, _: []const u32) anyerror!void {}
        fn put(_: *anyopaque, _: *const fabric.manifest.Layer, _: u64, _: u64, _: []const u8) anyerror!void {}
        fn commit(_: *anyopaque, _: u64, _: ps.Provenance) void {}
        fn abort(_: *anyopaque) void {}
    };
    var none: u8 = 0;
    var client: fabric.mailbox.Client = undefined;
    const links = [1]*fabric.mailbox.Client{&client};
    var remote: ps.Remote = .{ .gpa = a, .policy = .{ .enabled = true }, .client = &client, .peers = &links, .trigger = .{ .ptr = &none, .start_fn = Nop.start }, .model = model, .protocol = 2 };
    const sink: ps.StateSink = .{ .ptr = &none, .put_fn = Nop.put, .commit_fn = Nop.commit, .abort_fn = Nop.abort };
    const prompt = [_]u32{ 1, 2, 3 };
    try std.testing.expectError(error.SinkNotSharded, remote.source().fill(&prompt, .{ .upto = 2, .provenance = .remote }, sink));
}
