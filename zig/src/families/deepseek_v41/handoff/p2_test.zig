//! The hand-off over protocol 2 on the fake link: served parts pulled through `ShardSink` equal a direct import.
const std = @import("std");
const fabric = @import("fabric");
const g = @import("geometry.zig");
const plan = @import("plan.zig");
const ex = @import("export.zig");
const metal = @import("metal.zig");
const im = @import("import.zig");
const p2 = @import("p2.zig");
const s = @import("synth.zig");

const geo = g.release;
const producer = fabric.producer;
const half = 4 << 20;
const model = "dsv41-test";
const chunk: u64 = 70 << 10;

fn shapeFor(upto: u64, shard: u32, shards: u32, ds_rows: u32) plan.Shape {
    return .{ .upto = upto, .shard = shard, .shards = shards, .split = shards > 1, .ds_rows = ds_rows, .chunk = chunk };
}

fn target(arena: std.mem.Allocator, capacity: u32) !metal.Packed {
    const probe: metal.Packed = .{ .live = &.{}, .storage = &.{}, .layers = 43, .capacity = capacity };
    return .{ .live = try arena.alloc(u8, probe.liveBytes()), .storage = try arena.alloc(u8, metal.Packed.storageBytes(capacity)), .layers = 43, .capacity = capacity };
}

/// One shard's export served over its own fake link and pulled into `imp`.
fn pullShard(a: std.mem.Allocator, arena: std.mem.Allocator, w: s.World, shape: plan.Shape, tokens: []const u32, expect_tokens: []const u32, protocol: u16, imp: *im.Import) !fabric.decoder.Pulled {
    var link = try fabric.link_fake.Link.init(a, half, half, true, .immediate);
    defer link.deinit();
    link.up();
    const mac = link.box(0);
    const peer = link.box(1);
    var table: producer.Table = .{ .gpa = a, .ttl_ns = 60 * std.time.ns_per_s, .clock = fabric.words.nowNs };
    defer table.deinit();
    var serving: producer.Serving = undefined;
    serving.init(a, &peer, fabric.words.native, &table, model);
    defer serving.deinit();
    serving.responder.tp_rank = shape.shard;
    serving.responder.tp_size = shape.shards;
    const built = try p2.layers(arena, try ex.build(arena, shape, w.cuda));
    const id: [16]u8 = @splat(@intCast(shape.shard + 1));
    var exp: producer.Export = .{ .handoff = id, .tokens = tokens, .first_token = 0, .tokens_per_row = 1, .layers = built.layers, .sources = built.sources };
    defer a.free(exp.frames); // the responder plans the frames into the export; its owner frees them
    try table.add(id, "test", .{ .ready = &exp });
    var stop: std.atomic.Value(bool) = .init(false);
    const daemons = try std.Thread.spawn(.{}, fabric.link_fake.Link.run, .{ &link, &stop });
    const service = try std.Thread.spawn(.{}, producer.Serving.run, .{&serving});
    defer {
        stop.store(true, .release);
        serving.stop.store(true, .release);
        daemons.join();
        service.join();
    }
    var client = try fabric.mailbox.Client.init(&mac, fabric.words.native, 30 * std.time.ns_per_s);
    var sink: p2.ShardSink = .{ .imp = imp, .shard = shape.shard };
    var reason: fabric.decoder.Reason = .{};
    return fabric.decoder.pull(arena, &client, id, .{ .model = model, .tokens = expect_tokens, .tp_rank = shape.shard, .tp_size = shape.shards, .protocol = protocol }, sink.sink(), &reason);
}

fn endToEnd(upto: u64, world: u32, ds_rows: u32) !void {
    const a = std.testing.allocator;
    var arena_state: std.heap.ArenaAllocator = .init(a);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const w = try s.make(arena, .{ .upto = upto, .world = world, .split = world > 1, .ds_rows = ds_rows, .seed = upto });
    const tokens = try arena.alloc(u32, upto);
    for (tokens, 0..) |*t, i| t.* = @intCast(i * 7 + 1);

    const via = try target(arena, 1024);
    var imp = try im.Import.init(arena, shapeFor(upto, 0, world, ds_rows), .{ .@"packed" = via }, .{});
    for (0..world) |sh| {
        const shape = shapeFor(upto, @intCast(sh), world, ds_rows);
        const got = try pullShard(a, arena, w, shape, tokens, tokens, 2, &imp);
        try p2.check(arena, shape, got.manifest); // the manifest announced exactly the plan
        const again = try p2.shapeOf(geo, got.manifest);
        try std.testing.expectEqual(shape.upto, again.upto);
        try std.testing.expectEqual(shape.shards, again.shards);
        try std.testing.expectEqual(shape.split, again.split);
        if (sh == 0) try std.testing.expectEqual(ds_rows, again.ds_rows);
    }
    _ = try imp.complete();

    // the same parts landed directly give the same bytes
    const direct = try target(arena, 1024);
    var imp2 = try im.Import.init(arena, shapeFor(upto, 0, world, ds_rows), .{ .@"packed" = direct }, .{});
    var buf: [40]u8 = undefined;
    for (0..world) |sh| for (try ex.build(arena, shapeFor(upto, @intCast(sh), world, ds_rows), w.cuda)) |p| {
        try imp2.put(@intCast(sh), p.spec.layer, p.spec.name(&buf), p.spec.dtype, p.spec.rows, p.spec.cols, p.bytes);
    };
    _ = try imp2.complete();
    try std.testing.expectEqualSlices(u8, direct.live, via.live);
    try std.testing.expectEqualSlices(u8, direct.storage, via.storage);
}

test "protocol 2 over the fake link: one replicated producer, drafter rings riding" {
    try endToEnd(300, 1, 128);
}

test "protocol 2 over the fake link: two KV_SPLIT producers, a part per chunk window, one shared import" {
    try endToEnd(517, 2, 0);
}

test "a protocol 1 decoder refuses the export, and a pull for the wrong prompt or rank is refused" {
    const a = std.testing.allocator;
    var arena_state: std.heap.ArenaAllocator = .init(a);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const w = try s.make(arena, .{ .upto = 20 });
    const tokens = try arena.alloc(u32, 20);
    for (tokens, 0..) |*t, i| t.* = @intCast(i);
    const cap = try target(arena, 64);
    var imp = try im.Import.init(arena, shapeFor(20, 0, 1, 0), .{ .@"packed" = cap }, .{});
    try std.testing.expectError(error.Protocol, pullShard(a, arena, w, shapeFor(20, 0, 1, 0), tokens, tokens, 1, &imp));
    var other = try arena.dupe(u32, tokens);
    other[3] += 1;
    try std.testing.expectError(error.Mismatch, pullShard(a, arena, w, shapeFor(20, 0, 1, 0), tokens, other, 2, &imp));
}

test "the manifest is one recurrent layer per model layer, and a changed one is refused by check" {
    var arena_state: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const shape = shapeFor(300, 0, 1, 40);
    const specs = try plan.specs(arena, shape);
    const layers = try p2.describe(arena, specs);
    var m: fabric.manifest.Manifest = .{ .protocol = 2, .handoff = @splat(1), .model = model, .prompt_tokens = 300, .first_token = 0, .token_sha256 = fabric.wire.tokenSha256(&.{1}), .block_size = 1, .tp_rank = 0, .tp_size = 1, .layers = layers, .frames = 1 };
    try p2.check(arena, shape, m);
    for (layers) |l| try std.testing.expect(l.kind == .recurrent and l.parts.len > 0);
    try std.testing.expectEqual(@as(u32, 0), layers[0].index);
    try std.testing.expectEqualStrings("swa", layers[0].parts[0].name);
    try std.testing.expectEqual(fabric.manifest.Dtype.bfloat16, layers[0].parts[0].dtype);
    try std.testing.expectEqual(fabric.manifest.Dtype.float32, layers[2].parts[1].dtype); // the carry
    m.prompt_tokens = 299;
    try std.testing.expectError(error.BadManifest, p2.check(arena, shape, m));
    m.prompt_tokens = 300;
    m.protocol = 1;
    try std.testing.expectError(error.BadManifest, p2.check(arena, shape, m));
    m.protocol = 2;
    try std.testing.expectError(error.BadManifest, p2.check(arena, shape, .{ .protocol = 2, .handoff = m.handoff, .model = model, .prompt_tokens = 300, .first_token = 0, .token_sha256 = m.token_sha256, .block_size = 1, .tp_rank = 0, .tp_size = 1, .layers = layers[1..], .frames = 1 }));
    const parts = try arena.dupe(fabric.manifest.Part, layers[0].parts);
    parts[0].shape = &.{ 127, 512 };
    const changed = try arena.dupe(fabric.manifest.Layer, layers);
    changed[0].parts = parts;
    m.layers = changed;
    try std.testing.expectError(error.BadManifest, p2.check(arena, shape, m));
}

test "a protocol 1 reader of the manifest text refuses it" {
    var arena_state: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const specs = try plan.specs(arena, shapeFor(10, 0, 1, 0));
    const layers = try p2.describe(arena, specs);
    const m: fabric.manifest.Manifest = .{ .protocol = 2, .handoff = @splat(1), .model = model, .prompt_tokens = 10, .first_token = 0, .token_sha256 = fabric.wire.tokenSha256(&.{1}), .block_size = 1, .tp_rank = 0, .tp_size = 1, .layers = layers, .frames = 1 };
    var out: std.Io.Writer.Allocating = .init(arena);
    try fabric.manifest.write(m, &out.writer);
    try std.testing.expectError(error.BadManifest, fabric.manifest.parse(arena, out.written()));
    const back = try fabric.manifest.parseUpTo(arena, out.written(), 2);
    try std.testing.expectEqual(layers.len, back.layers.len);
}
