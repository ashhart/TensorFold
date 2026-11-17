//! A streamed export over the fake link: frames pulled while another thread readies them land as the whole.
const std = @import("std");
const fabric = @import("fabric");
const plan = @import("plan.zig");
const ex = @import("export.zig");
const metal = @import("metal.zig");
const im = @import("import.zig");
const p2 = @import("p2.zig");
const s = @import("synth.zig");

const producer = fabric.producer;
const half = 4 << 20;
const model = "dsv41-test";
const upto: u64 = 600;
const shape: plan.Shape = .{ .upto = upto, .ds_rows = 128, .chunk = 70 << 10 };

const Feed = struct {
    progress: *producer.Progress,
    order: []const usize,
    fail: ?[]const u8,
    stop: *const std.atomic.Value(bool),

    /// Marks frames ready in `order`, a nap apart, then the export whole (or failed).
    fn run(f: *Feed) void {
        for (f.order) |i| {
            if (f.stop.load(.acquire)) return;
            std.Thread.yield() catch {};
            fabric.words.pause(200 * std.time.ns_per_us);
            f.progress.mark(i);
        }
        if (f.fail) |why| return f.progress.fail(why);
        f.progress.whole.store(true, .release);
    }
};

/// Pulls a streamed export of the synthetic world whose frames become ready in the reverse of the manifest's order.
fn pull(a: std.mem.Allocator, arena: std.mem.Allocator, stream: bool, fail: ?[]const u8, imp: *im.Import) !fabric.decoder.Pulled {
    const w = try s.make(arena, .{ .upto = upto, .ds_rows = 128, .seed = 9 });
    const tokens = try arena.alloc(u32, upto);
    for (tokens, 0..) |*t, i| t.* = @intCast(i * 7 + 1);
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
    const built = try p2.layers(arena, try ex.build(arena, shape, w.cuda));
    const id: [16]u8 = @splat(0x33);
    const ready = try arena.alloc(std.atomic.Value(u8), built.layers.len * 64);
    var progress: producer.Progress = .{ .ready = ready };
    for (ready) |*r| r.* = .init(0);
    var exp: producer.Export = .{ .handoff = id, .tokens = tokens, .first_token = 0, .tokens_per_row = 1, .layers = built.layers, .sources = built.sources, .progress = &progress };
    defer a.free(exp.frames);
    try table.add(id, "test", .{ .ready = &exp });
    var stop: std.atomic.Value(bool) = .init(false);
    const daemons = try std.Thread.spawn(.{}, fabric.link_fake.Link.run, .{ &link, &stop });
    const service = try std.Thread.spawn(.{}, producer.Serving.run, .{&serving});
    var order: std.ArrayList(usize) = .empty;
    // the frames' count is known once planned; the feed marks every index the plan can hold, last first
    var at: usize = ready.len;
    while (at > 0) : (at -= 1) try order.append(arena, at - 1);
    var feed: Feed = .{ .progress = &progress, .order = order.items, .fail = fail, .stop = &stop };
    const feeder = try std.Thread.spawn(.{}, Feed.run, .{&feed});
    defer {
        stop.store(true, .release);
        serving.stop.store(true, .release);
        feeder.join();
        daemons.join();
        service.join();
    }
    var client = try fabric.mailbox.Client.init(&mac, fabric.words.native, 30 * std.time.ns_per_s);
    var sink: p2.ShardSink = .{ .imp = imp, .shard = 0 };
    var reason: fabric.decoder.Reason = .{};
    return fabric.decoder.pull(arena, &client, id, .{ .model = model, .tokens = tokens, .protocol = 2, .stream = stream }, sink.sink(), &reason);
}

fn target(arena: std.mem.Allocator) !metal.Packed {
    const probe: metal.Packed = .{ .live = &.{}, .storage = &.{}, .layers = 43, .capacity = 1024 };
    return .{ .live = try arena.alloc(u8, probe.liveBytes()), .storage = try arena.alloc(u8, metal.Packed.storageBytes(1024)), .layers = 43, .capacity = 1024 };
}

test "a streamed export pulled while its frames turn ready, last first, lands as the whole export" {
    const a = std.testing.allocator;
    var arena_state: std.heap.ArenaAllocator = .init(a);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const w = try s.make(arena, .{ .upto = upto, .ds_rows = 128, .seed = 9 });
    const via = try target(arena);
    var imp = try im.Import.init(arena, shape, .{ .@"packed" = via }, .{});
    const got = try pull(a, arena, true, null, &imp);
    try p2.check(arena, shape, got.manifest);
    _ = try imp.complete();
    const direct = try target(arena);
    var imp2 = try im.Import.init(arena, shape, .{ .@"packed" = direct }, .{});
    var buf: [40]u8 = undefined;
    for (try ex.build(arena, shape, w.cuda)) |p| try imp2.put(0, p.spec.layer, p.spec.name(&buf), p.spec.dtype, p.spec.rows, p.spec.cols, p.bytes);
    _ = try imp2.complete();
    try std.testing.expectEqualSlices(u8, direct.live, via.live);
    try std.testing.expectEqualSlices(u8, direct.storage, via.storage);
}

test "a decoder that did not ask to stream waits for the whole export; a failed one is refused with its reason" {
    const a = std.testing.allocator;
    var arena_state: std.heap.ArenaAllocator = .init(a);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const via = try target(arena);
    var imp = try im.Import.init(arena, shape, .{ .@"packed" = via }, .{});
    _ = try pull(a, arena, false, null, &imp);
    var imp2 = try im.Import.init(arena, shape, .{ .@"packed" = try target(arena) }, .{});
    try std.testing.expectError(error.Refused, pull(a, arena, true, "device lost", &imp2));
}
