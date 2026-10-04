//! A KV handoff end to end over the fake link: producer and service on the listen end, decoder and client on the connect end.
const std = @import("std");
const fabric = @import("fabric");
const layout = fabric.layout;
const mailbox = fabric.mailbox;
const manifest = fabric.manifest;
const producer = fabric.producer;
const decoder = fabric.decoder;
const frames = fabric.frames;

const gpa = std.testing.allocator;
const half = 4 << 20;

/// Two cache tensors: a packed-attention layer split over two frames and an MLA layer in one.
const Model = struct {
    attn: []u8,
    mla: []u8,
    attn_rows: [100]u64,
    mla_rows: [3]u64 = .{ 7, 3, 9 },
    layers: [2]manifest.Layer = undefined,
    sources: [2]frames.Source = undefined,

    fn init() !*Model {
        const m = try gpa.create(Model);
        m.attn = try gpa.alloc(u8, 200 * 8 * 16 * 256 * 2);
        m.mla = try gpa.alloc(u8, 50 * 16 * 576 * 2);
        m.mla_rows = .{ 7, 3, 9 };
        for (m.attn, 0..) |*b, i| b.* = @truncate(i *% 2654435761 >> 13);
        for (m.mla, 0..) |*b, i| b.* = @truncate(i *% 40503 >> 7);
        for (&m.attn_rows, 0..) |*r, i| r.* = 199 - 2 * i % 200;
        m.layers = .{
            .{ .index = 3, .kind = .attention, .shape = &.{ 100, 8, 16, 256 }, .dims = &.{ .block, .head, .token, .kv_head_dim }, .dtype = .bfloat16, .heads = 8, .total_heads = 8, .head_size = 128 },
            .{ .index = 11, .kind = .mla, .shape = &.{ 3, 16, 576 }, .dims = &.{ .block, .token, .latent }, .dtype = .bfloat16, .latent_size = 512, .rope_size = 64 },
        };
        m.sources = .{
            .{ .bytes = m.attn, .shape = &.{ 200, 8, 16, 256 }, .block_dim = 0, .elem = 2, .rows = &m.attn_rows },
            .{ .bytes = m.mla, .shape = &.{ 50, 16, 576 }, .block_dim = 0, .elem = 2, .rows = &m.mla_rows },
        };
        return m;
    }

    fn deinit(m: *Model) void {
        gpa.free(m.attn);
        gpa.free(m.mla);
        gpa.destroy(m);
    }
};

/// The decoder's destination: each layer's rows C-contiguous, as a backend's cache import would place them.
const Sink = struct {
    got: [2][]u8,
    model: *Model,

    fn put(ptr: *anyopaque, layer: *const manifest.Layer, row_start: u64, rows: u64, bytes: []const u8) anyerror!void {
        const s: *Sink = @ptrCast(@alignCast(ptr));
        const i: usize = if (layer.index == 3) 0 else 1;
        const at: usize = @intCast(row_start * layer.rowBytes());
        @memcpy(s.got[i][at..][0..bytes.len], bytes);
        _ = rows;
    }
};

const Producer = struct {
    service: mailbox.Service,
    responder: producer.Responder,
    stop: std.atomic.Value(bool) = .init(false),

    fn publish(ptr: *anyopaque, seq: u32, len: usize) anyerror!void {
        const p: *Producer = @ptrCast(@alignCast(ptr));
        try p.service.publish(seq, len);
    }

    fn run(p: *Producer) void {
        while (!p.stop.load(.acquire)) {
            const req = (p.service.next(std.time.ns_per_ms) catch return) orelse continue;
            p.responder.handle(req.seq, req.payload) catch |err| p.responder.failed(req.seq, err) catch return;
        }
    }
};

fn clock() u64 {
    return fabric.words.nowNs();
}

fn handoff(direct: bool) !void {
    var link = try fabric.link_fake.Link.init(gpa, half, half, direct, .immediate);
    defer link.deinit();
    link.up();
    const mac = link.box(0);
    const peer = link.box(1);
    const model = try Model.init();
    defer model.deinit();
    var table: producer.Table = .{ .gpa = gpa, .ttl_ns = 60 * std.time.ns_per_s, .clock = clock };
    defer table.deinit();
    var p: Producer = .{ .service = mailbox.Service.init(&peer, fabric.words.native), .responder = undefined };
    p.responder = .{ .gpa = gpa, .table = &table, .out = .{ .area = p.service.replyArea(), .ptr = &p, .publish_fn = Producer.publish }, .model = "org/model" };
    defer p.responder.deinit();
    const tokens = try gpa.alloc(u32, 1600);
    defer gpa.free(tokens);
    for (tokens, 0..) |*t, i| t.* = @intCast(i * 7 % 50000);
    const id: [16]u8 = .{ 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 1, 2, 3, 4, 5, 6 };
    var exp: producer.Export = .{ .handoff = id, .tokens = tokens, .first_token = 0, .tokens_per_row = 16, .layers = &model.layers, .sources = &model.sources };
    defer gpa.free(exp.frames);
    var stop: std.atomic.Value(bool) = .init(false);
    const daemons = try std.Thread.spawn(.{}, fabric.link_fake.Link.run, .{ &link, &stop });
    const service = try std.Thread.spawn(.{}, Producer.run, .{&p});
    defer {
        stop.store(true, .release);
        p.stop.store(true, .release);
        daemons.join();
        service.join();
    }
    var client = try mailbox.Client.init(&mac, fabric.words.native, 5 * std.time.ns_per_s);
    var sink: Sink = .{ .got = .{ try gpa.alloc(u8, 100 * 65536), try gpa.alloc(u8, 3 * 18432) }, .model = model };
    defer for (sink.got) |g| gpa.free(g);
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    var reason: decoder.Reason = .{};
    const expect: decoder.Expect = .{ .model = "org/model", .tokens = tokens, .open_timeout_ns = std.time.ns_per_s };
    const adder = try std.Thread.spawn(.{}, struct {
        fn later(t: *producer.Table, e: *producer.Export) void {
            fabric.words.pause(20 * std.time.ns_per_ms);
            t.add(e.handoff, "req-1", .{ .ready = e }) catch {};
        }
    }.later, .{ &table, &exp });
    const got = try decoder.pull(arena.allocator(), &client, id, expect, .{ .ptr = &sink, .put_fn = Sink.put }, &reason);
    adder.join();
    try std.testing.expectEqual(@as(u64, 3), got.frames);
    try std.testing.expectEqual(@as(u64, 100 * 65536 + 3 * 18432), got.bytes);
    var want = try gpa.alloc(u8, 100 * 65536);
    defer gpa.free(want);
    _ = try model.sources[0].gather(0, 100, want);
    try std.testing.expectEqualSlices(u8, want, sink.got[0]);
    _ = try model.sources[1].gather(0, 3, want);
    try std.testing.expectEqualSlices(u8, want[0 .. 3 * 18432], sink.got[1]);
    var finished: std.ArrayList([]const u8) = .empty;
    defer finished.deinit(gpa);
    try table.takeFinished(&finished);
    try std.testing.expectEqual(@as(usize, 1), finished.items.len);
    try std.testing.expectEqualStrings("req-1", finished.items[0]);
    try std.testing.expectEqual(@as(?producer.Entry, null), table.get(id));
}

test "a handoff waits for its export, then every frame lands byte for byte (direct replies)" {
    try handoff(true);
}

test "the same handoff with the connect end reading replies (pull mode)" {
    try handoff(false);
}

test "a refused export reaches the decoder with the producer's reason" {
    var link = try fabric.link_fake.Link.init(gpa, half, half, true, .immediate);
    defer link.deinit();
    link.up();
    const mac = link.box(0);
    const peer = link.box(1);
    var table: producer.Table = .{ .gpa = gpa, .ttl_ns = std.time.ns_per_s, .clock = clock };
    defer table.deinit();
    const id: [16]u8 = @splat(4);
    try table.add(id, "req-2", .{ .failed = "float8_e4m3fn KV caches cannot be exported" });
    var p: Producer = .{ .service = mailbox.Service.init(&peer, fabric.words.native), .responder = undefined };
    p.responder = .{ .gpa = gpa, .table = &table, .out = .{ .area = p.service.replyArea(), .ptr = &p, .publish_fn = Producer.publish }, .model = "org/model" };
    defer p.responder.deinit();
    var stop: std.atomic.Value(bool) = .init(false);
    const daemons = try std.Thread.spawn(.{}, fabric.link_fake.Link.run, .{ &link, &stop });
    const service = try std.Thread.spawn(.{}, Producer.run, .{&p});
    defer {
        stop.store(true, .release);
        p.stop.store(true, .release);
        daemons.join();
        service.join();
    }
    var client = try mailbox.Client.init(&mac, fabric.words.native, 5 * std.time.ns_per_s);
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    var reason: decoder.Reason = .{};
    var dummy: u8 = 0;
    const sink: decoder.Sink = .{ .ptr = &dummy, .put_fn = Sink.put };
    try std.testing.expectError(error.Refused, decoder.pull(arena.allocator(), &client, id, .{ .model = "org/model", .tokens = &.{ 1, 2, 3 } }, sink, &reason));
    try std.testing.expectEqualStrings("float8_e4m3fn KV caches cannot be exported", reason.text());
    try std.testing.expect(!client.poisoned);
}

test "a link that drops mid-call poisons the client and a new generation is needed" {
    var link = try fabric.link_fake.Link.init(gpa, half, half, true, .immediate);
    defer link.deinit();
    link.up();
    const mac = link.box(0);
    var client = try mailbox.Client.init(&mac, fabric.words.native, std.time.ns_per_s);
    const dropper = try std.Thread.spawn(.{}, struct {
        fn later(l: *fabric.link_fake.Link) void {
            fabric.words.pause(10 * std.time.ns_per_ms);
            l.down();
        }
    }.later, .{&link});
    try std.testing.expectError(error.LinkLost, client.call("nobody answers"));
    dropper.join();
    try std.testing.expect(client.poisoned);
    try std.testing.expectError(error.Poisoned, client.call("again"));
    link.up();
    const fresh = try mailbox.Client.init(&mac, fabric.words.native, std.time.ns_per_s);
    try std.testing.expectEqual(@as(u64, 2), fresh.generation);
}

test "the remote prefill source fills whole chunks and commits them as remote, never cacheable" {
    const ps = fabric.prefill_source;
    const State = struct {
        committed: ?u64 = null,
        provenance: ?ps.Provenance = null,
        aborted: bool = false,
        fn put(_: *anyopaque, _: *const manifest.Layer, _: u64, _: u64, _: []const u8) anyerror!void {}
        fn commit(ptr: *anyopaque, upto: u64, p: ps.Provenance) void {
            const s: *@This() = @ptrCast(@alignCast(ptr));
            s.committed = upto;
            s.provenance = p;
        }
        fn abort(ptr: *anyopaque) void {
            const s: *@This() = @ptrCast(@alignCast(ptr));
            s.aborted = true;
        }
    };
    const Trig = struct {
        fn start(_: *anyopaque, _: [16]u8, _: []const u32) anyerror!void {
            return error.ProducerUnreachable;
        }
    };
    var link = try fabric.link_fake.Link.init(gpa, half, half, true, .immediate);
    defer link.deinit();
    link.up();
    const mac = link.box(0);
    var client = try mailbox.Client.init(&mac, fabric.words.native, std.time.ns_per_s);
    var t: u8 = 0;
    var remote: ps.Remote = .{ .gpa = gpa, .policy = .{ .enabled = true, .min_prefix = 64, .chunk = 32 }, .client = &client, .trigger = .{ .ptr = &t, .start_fn = Trig.start }, .model = "org/model" };
    const src = remote.source();
    var prompt: [100]u32 = undefined;
    for (&prompt, 0..) |*x, i| x.* = @intCast(i);
    const offer = src.offer(&prompt, 0).?;
    try std.testing.expectEqual(@as(u64, 96), offer.upto);
    try std.testing.expectEqual(ps.Provenance.remote, offer.provenance);
    var st: State = .{};
    const sink: ps.StateSink = .{ .ptr = &st, .put_fn = State.put, .commit_fn = State.commit, .abort_fn = State.abort };
    try std.testing.expectError(error.ProducerUnreachable, src.fill(&prompt, offer, sink));
    try std.testing.expect(st.aborted and st.committed == null);
    try std.testing.expectEqual(@as(?ps.Offer, null), src.offer(prompt[0..40], 0));
}
