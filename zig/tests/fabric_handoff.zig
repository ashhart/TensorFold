//! KV handoff end-to-end over the fake link: producer/service on the listen end, decoder/client on the connect end.
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

/// Protocol 2: a recurrent layer's two parts (a recurrence and a convolution tail) beside an attention layer.
const Recurrent = struct {
    rec: [4 * 8 * 8 * 4]u8 = undefined,
    conv: [3 * 40 * 2]u8 = undefined,
    attn: [4 * 4 * 16 * 2]u8 = undefined,
    attn_rows: [4]u64 = .{ 0, 1, 2, 3 },
    part_bytes: [2][]const u8 = undefined,
    parts: [2]manifest.Part = .{
        .{ .name = "recurrence", .shape = &.{ 4, 8, 8 }, .dims = &.{ .head, .value, .key }, .dtype = .float32 },
        .{ .name = "conv", .shape = &.{ 3, 40 }, .dims = &.{ .tap, .channel }, .dtype = .bfloat16 },
    },
    layers: [2]manifest.Layer = undefined,
    sources: [2]frames.Source = undefined,
    got: [3][]u8 = undefined,

    fn fill(r: *Recurrent) void {
        for (&r.rec, 0..) |*b, i| b.* = @truncate(i *% 31 +% 7);
        for (&r.conv, 0..) |*b, i| b.* = @truncate(i *% 17 +% 3);
        for (&r.attn, 0..) |*b, i| b.* = @truncate(i *% 13 +% 1);
        r.part_bytes = .{ &r.rec, &r.conv };
        r.layers = .{
            .{ .index = 0, .kind = .recurrent, .shape = &.{}, .dims = &.{}, .dtype = .float32, .parts = &r.parts },
            .{ .index = 3, .kind = .attention, .shape = &.{ 4, 4, 1, 16 }, .dims = &.{ .block, .head, .token, .head_dim }, .dtype = .bfloat16, .heads = 4, .total_heads = 4, .head_size = 16 },
        };
        r.sources = .{
            .{ .bytes = &.{}, .shape = &.{}, .block_dim = 0, .elem = 1, .rows = &.{}, .parts = &r.part_bytes },
            .{ .bytes = &r.attn, .shape = &.{ 4, 4, 1, 16 }, .block_dim = 0, .elem = 2, .rows = &r.attn_rows },
        };
    }

    fn put(ptr: *anyopaque, layer: *const manifest.Layer, row_start: u64, rows: u64, bytes: []const u8) anyerror!void {
        const r: *Recurrent = @ptrCast(@alignCast(ptr));
        if (layer.kind == .recurrent) {
            r.got[@intCast(row_start)] = try gpa.dupe(u8, bytes);
            return;
        }
        if (row_start != 0 or rows != 4) return error.UnexpectedRows;
        r.got[2] = try gpa.dupe(u8, bytes);
    }
};

/// A protocol 2 export over the fake link, pulled by a decoder reading up to `protocol`.
fn recurrentHandoff(protocol: u16) !void {
    var link = try fabric.link_fake.Link.init(gpa, half, half, true, .immediate);
    defer link.deinit();
    link.up();
    const mac = link.box(0);
    const peer = link.box(1);
    var r: Recurrent = .{};
    r.fill();
    var table: producer.Table = .{ .gpa = gpa, .ttl_ns = 60 * std.time.ns_per_s, .clock = clock };
    defer table.deinit();
    var p: Producer = .{ .service = mailbox.Service.init(&peer, fabric.words.native), .responder = undefined };
    p.responder = .{ .gpa = gpa, .table = &table, .out = .{ .area = p.service.replyArea(), .ptr = &p, .publish_fn = Producer.publish }, .model = "org/model" };
    defer p.responder.deinit();
    const tokens = [_]u32{ 11, 12, 13, 14 };
    const id: [16]u8 = @splat(0x22);
    var exp: producer.Export = .{ .handoff = id, .tokens = &tokens, .first_token = 0, .tokens_per_row = 1, .layers = &r.layers, .sources = &r.sources };
    defer gpa.free(exp.frames);
    try std.testing.expectEqual(@as(u16, 2), exp.protocol());
    try table.add(id, "req-r", .{ .ready = &exp });
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
    r.got = .{ &.{}, &.{}, &.{} };
    defer for (r.got) |g| if (g.len > 0) gpa.free(g);
    const expect: decoder.Expect = .{ .model = "org/model", .tokens = &tokens, .protocol = protocol };
    const sink: decoder.Sink = .{ .ptr = &r, .put_fn = Recurrent.put };
    if (protocol < fabric.wire.recurrent_version) {
        try std.testing.expectError(error.Protocol, decoder.pull(arena.allocator(), &client, id, expect, sink, &reason));
        return;
    }
    const got = try decoder.pull(arena.allocator(), &client, id, expect, sink, &reason);
    try std.testing.expectEqual(@as(u64, 3), got.frames); // two parts, then the attention rows in one frame
    try std.testing.expectEqual(@as(u16, 2), got.manifest.protocol);
    try std.testing.expectEqualSlices(u8, &r.rec, r.got[0]);
    try std.testing.expectEqualSlices(u8, &r.conv, r.got[1]);
    try std.testing.expectEqualSlices(u8, &r.attn, r.got[2]);
    try std.testing.expectEqual(@as(?producer.Entry, null), table.get(id));
}

test "protocol 2: a recurrent layer's parts land whole, beside an attention layer's rows" {
    try recurrentHandoff(fabric.wire.recurrent_version);
}

test "protocol 2: a protocol 1 decoder refuses the export by its header" {
    try recurrentHandoff(fabric.wire.version);
}

test "protocol 2: a hybrid state crosses the link a part or a page row a frame, and lands byte for byte" {
    const hx = fabric.hybrid_export;
    const plan: hx.Plan = .{ // six linear and two attention layers; 2048 tokens in page rows of 16
        .g = .{ .linear = 6, .attention = 2, .rec_bytes = 8 * 16 * 16 * 4, .conv_bytes = 3 * 96 * 2, .kv_heads = 4, .row_bytes = 256 * 2 },
        .s = .{ .rec = .{ 8, 16, 16 }, .conv = .{ 3, 96 }, .head_dim = 256, .interval = 4 },
        .at = 2048,
        .block = 16,
    };
    // a page row is 64 KiB, so a 4 MiB reply half takes 63 of an attention layer's 128: 3 frames a layer
    var link = try fabric.link_fake.Link.init(gpa, half, half, true, .immediate);
    defer link.deinit();
    link.up();
    const mac = link.box(0);
    const peer = link.box(1);
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const image = try a.alloc(u8, @intCast(plan.g.bytes(plan.at)));
    for (image, 0..) |*b, i| b.* = @truncate(i *% 2654435761 >> 11);
    const built = try hx.build(a, plan, image);
    var table: producer.Table = .{ .gpa = gpa, .ttl_ns = 60 * std.time.ns_per_s, .clock = clock };
    defer table.deinit();
    var p: Producer = .{ .service = mailbox.Service.init(&peer, fabric.words.native), .responder = undefined };
    p.responder = .{ .gpa = gpa, .table = &table, .out = .{ .area = p.service.replyArea(), .ptr = &p, .publish_fn = Producer.publish }, .model = "org/hybrid" };
    defer p.responder.deinit();
    var tokens: [2048]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(1000 + i);
    const id: [16]u8 = @splat(0x2b);
    var exp: producer.Export = .{ .handoff = id, .tokens = &tokens, .first_token = 0, .tokens_per_row = plan.block, .layers = built.layers, .sources = built.sources };
    defer gpa.free(exp.frames);
    try table.add(id, "req-h", .{ .ready = &exp });
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
    var reason: decoder.Reason = .{};
    const landed = try a.alloc(u8, image.len);
    @memset(landed, 0);
    var landing: hx.Landing = .{ .plan = plan, .image = landed };
    const expect: decoder.Expect = .{ .model = "org/hybrid", .tokens = &tokens, .protocol = fabric.wire.recurrent_version };
    const got = try decoder.pull(a, &client, id, expect, .{ .ptr = &landing, .put_fn = hx.Landing.putAny }, &reason);
    try std.testing.expectEqual(@as(u64, 6 * 2 + 2 * 3), got.frames);
    try std.testing.expect(landing.whole());
    try std.testing.expectEqualSlices(u8, image, landed);
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
        fn start(ptr: *anyopaque, _: [16]u8, tokens: []const u32) anyerror!void {
            @as(*usize, @ptrCast(@alignCast(ptr))).* = tokens.len; // the prefix and one token past it
            return error.ProducerUnreachable;
        }
    };
    var link = try fabric.link_fake.Link.init(gpa, half, half, true, .immediate);
    defer link.deinit();
    link.up();
    const mac = link.box(0);
    var client = try mailbox.Client.init(&mac, fabric.words.native, std.time.ns_per_s);
    var t: usize = 0;
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
    try std.testing.expectEqual(@as(usize, 97), t);
    try std.testing.expectEqual(@as(?ps.Offer, null), src.offer(prompt[0..40], 0));
}
