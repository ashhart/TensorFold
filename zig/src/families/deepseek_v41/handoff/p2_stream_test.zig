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
const manifest = fabric.manifest;
const half = 4 << 20;
const model = "dsv41-test";
const upto: u64 = 600;
const shape: plan.Shape = .{ .upto = upto, .ds_rows = 128, .chunk = 70 << 10 };

/// Lands each pulled part in the import, noting their order and whether the export was whole at the first.
const Tap = struct {
    inner: p2.ShardSink,
    arena: std.mem.Allocator,
    progress: *const producer.Progress,
    landed: std.atomic.Value(usize) = .init(0),
    order: std.ArrayList([2]u64) = .empty, // (layer, part) in landing order
    whole_at_first: ?bool = null,

    fn sink(t: *Tap) fabric.decoder.Sink {
        return .{ .ptr = t, .put_fn = put };
    }

    fn put(ptr: *anyopaque, layer: *const manifest.Layer, row_start: u64, rows: u64, bytes: []const u8) anyerror!void {
        const t: *Tap = @ptrCast(@alignCast(ptr));
        if (t.whole_at_first == null) t.whole_at_first = t.progress.whole.load(.acquire);
        try t.order.append(t.arena, .{ layer.index, row_start });
        const inner = t.inner.sink();
        try inner.put_fn(inner.ptr, layer, row_start, rows, bytes);
        _ = t.landed.fetchAdd(1, .release);
    }
};

/// The converter's stand-in: copies each part into the export and marks it ready, the first part last (or never).
const Feed = struct {
    progress: *producer.Progress,
    from: []const ex.Part,
    into: []const ex.Part,
    landed: *const std.atomic.Value(usize),
    fail: ?[]const u8,
    hold: bool, // the first part waits for the others to land (a streaming decoder takes them early)
    delay_ns: u64, // and then this long (a slow converter)
    stop: *const std.atomic.Value(bool),

    fn ready(f: *Feed, i: usize) void {
        @memcpy(f.into[i].bytes, f.from[i].bytes);
        f.progress.mark(i);
    }

    /// Parts last to first; the first, held, waits until every other has landed at the decoder, so its WAIT is certain.
    fn run(f: *Feed) void {
        var i = f.from.len;
        while (i > 1) {
            i -= 1;
            if (f.stop.load(.acquire)) return;
            std.Thread.yield() catch {};
            fabric.words.pause(200 * std.time.ns_per_us);
            f.ready(i);
        }
        if (f.fail) |why| return f.progress.fail(why);
        var held: u64 = 0; // bounded, so a producer that never lets the decoder in ends the test rather than hangs it
        while (f.hold and held < 5 * std.time.ns_per_s and f.landed.load(.acquire) < f.from.len - 1) : (held += 200 * std.time.ns_per_us) {
            if (f.stop.load(.acquire)) return;
            fabric.words.pause(200 * std.time.ns_per_us);
        }
        var waited: u64 = 0;
        while (waited < f.delay_ns and !f.stop.load(.acquire)) : (waited += std.time.ns_per_ms) fabric.words.pause(std.time.ns_per_ms);
        f.ready(0);
        f.progress.whole.store(true, .release);
    }
};

/// How a pull goes: what the decoder asks and knows, and how the converter behaves.
const Mode = struct {
    stream: bool,
    fail: ?[]const u8 = null,
    delay_ns: u64 = 0,
    timeout_ns: u64 = 30 * std.time.ns_per_s,
    working: ?*const std.atomic.Value(bool) = null,
    failed: ?*const std.atomic.Value(bool) = null,
};

/// Pulls a streamed export of the synthetic world whose parts are converted in the reverse of the manifest's order.
fn pull(a: std.mem.Allocator, arena: std.mem.Allocator, mode: Mode, tap: *Tap) !fabric.decoder.Pulled {
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
    // the export's parts start as zeros: a frame served before its part is converted lands wrong bytes
    const full = try ex.build(arena, shape, w.cuda);
    const blank = try arena.alloc(ex.Part, full.len);
    for (full, blank) |p, *b| {
        b.* = .{ .spec = p.spec, .bytes = try arena.alloc(u8, p.bytes.len) };
        @memset(b.bytes, 0);
    }
    const built = try p2.layers(arena, blank);
    const id: [16]u8 = @splat(0x33);
    const ready = try arena.alloc(std.atomic.Value(u8), blank.len);
    var progress: producer.Progress = .{ .ready = ready };
    for (ready) |*r| r.* = .init(0);
    var exp: producer.Export = .{ .handoff = id, .tokens = tokens, .first_token = 0, .tokens_per_row = 1, .layers = built.layers, .sources = built.sources, .progress = &progress };
    defer a.free(exp.frames);
    tap.progress = &progress;
    try table.add(id, "test", .{ .ready = &exp });
    var stop: std.atomic.Value(bool) = .init(false);
    const daemons = try std.Thread.spawn(.{}, fabric.link_fake.Link.run, .{ &link, &stop });
    const service = try std.Thread.spawn(.{}, producer.Serving.run, .{&serving});
    var feed: Feed = .{ .progress = &progress, .from = full, .into = blank, .landed = &tap.landed, .fail = mode.fail, .hold = mode.stream, .delay_ns = mode.delay_ns, .stop = &stop };
    const feeder = try std.Thread.spawn(.{}, Feed.run, .{&feed});
    defer {
        stop.store(true, .release);
        serving.stop.store(true, .release);
        feeder.join();
        daemons.join();
        service.join();
    }
    var client = try fabric.mailbox.Client.init(&mac, fabric.words.native, 30 * std.time.ns_per_s);
    var reason: fabric.decoder.Reason = .{};
    const got = try fabric.decoder.pull(arena, &client, id, .{ .model = model, .tokens = tokens, .protocol = 2, .stream = mode.stream, .open_timeout_ns = mode.timeout_ns, .working = mode.working, .failed = mode.failed }, tap.sink(), &reason);
    // a frame is a part: the converter's marks are the frames' readiness
    try std.testing.expectEqual(blank.len, exp.frames.len);
    return got;
}

fn target(arena: std.mem.Allocator) !metal.Packed {
    const probe: metal.Packed = .{ .live = &.{}, .storage = &.{}, .layers = 43, .capacity = 1024 };
    return .{ .live = try arena.alloc(u8, probe.liveBytes()), .storage = try arena.alloc(u8, metal.Packed.storageBytes(1024)), .layers = 43, .capacity = 1024 };
}

/// The import of the whole export built directly, the reference a pulled one must equal.
fn direct(arena: std.mem.Allocator, w: s.World) !metal.Packed {
    const out = try target(arena);
    var imp = try im.Import.init(arena, shape, .{ .@"packed" = out }, .{});
    var buf: [40]u8 = undefined;
    for (try ex.build(arena, shape, w.cuda)) |p| try imp.put(0, p.spec.layer, p.spec.name(&buf), p.spec.dtype, p.spec.rows, p.spec.cols, p.bytes);
    _ = try imp.complete();
    return out;
}

test "a streamed export pulled while its parts turn ready, last first, lands as the whole export" {
    const a = std.testing.allocator;
    var arena_state: std.heap.ArenaAllocator = .init(a);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const w = try s.make(arena, .{ .upto = upto, .ds_rows = 128, .seed = 9 });
    const via = try target(arena);
    var imp = try im.Import.init(arena, shape, .{ .@"packed" = via }, .{});
    var tap: Tap = .{ .inner = .{ .imp = &imp, .shard = 0 }, .arena = arena, .progress = undefined };
    const got = try pull(a, arena, .{ .stream = true }, &tap);
    try p2.check(arena, shape, got.manifest);
    _ = try imp.complete();
    const want = try direct(arena, w);
    try std.testing.expectEqualSlices(u8, want.live, via.live);
    try std.testing.expectEqualSlices(u8, want.storage, via.storage);
    // the first part was held back and landed last, and the export was not whole when the first part landed
    const first: [2]u64 = .{ got.manifest.layers[0].index, 0 };
    try std.testing.expectEqual(got.frames, tap.order.items.len);
    try std.testing.expectEqualSlices(u64, &first, &tap.order.items[tap.order.items.len - 1]);
    try std.testing.expectEqual(@as(?bool, false), tap.whole_at_first);
}

test "a decoder that did not ask to stream gets its first part only once the export is whole" {
    const a = std.testing.allocator;
    var arena_state: std.heap.ArenaAllocator = .init(a);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const w = try s.make(arena, .{ .upto = upto, .ds_rows = 128, .seed = 9 });
    const via = try target(arena);
    var imp = try im.Import.init(arena, shape, .{ .@"packed" = via }, .{});
    var tap: Tap = .{ .inner = .{ .imp = &imp, .shard = 0 }, .arena = arena, .progress = undefined };
    const got = try pull(a, arena, .{ .stream = false }, &tap);
    try std.testing.expectEqual(@as(?bool, true), tap.whole_at_first);
    _ = got;
    _ = try imp.complete();
    const want = try direct(arena, w);
    try std.testing.expectEqualSlices(u8, want.live, via.live);
    try std.testing.expectEqualSlices(u8, want.storage, via.storage);
}

test "a streamed export that fails is refused with its reason" {
    const a = std.testing.allocator;
    var arena_state: std.heap.ArenaAllocator = .init(a);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    var imp = try im.Import.init(arena, shape, .{ .@"packed" = try target(arena) }, .{});
    var tap: Tap = .{ .inner = .{ .imp = &imp, .shard = 0 }, .arena = arena, .progress = undefined };
    try std.testing.expectError(error.Refused, pull(a, arena, .{ .stream = true, .fail = "device lost" }, &tap));
}

/// A pull with a converter whose first part comes 150 ms late, and a decoder that waits 20 ms for anything new.
fn slow(a: std.mem.Allocator, working: ?*const std.atomic.Value(bool)) !void {
    var arena_state: std.heap.ArenaAllocator = .init(a);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    var imp = try im.Import.init(arena, shape, .{ .@"packed" = try target(arena) }, .{});
    var tap: Tap = .{ .inner = .{ .imp = &imp, .shard = 0 }, .arena = arena, .progress = undefined };
    _ = try pull(a, arena, .{ .stream = true, .delay_ns = 150 * std.time.ns_per_ms, .timeout_ns = 20 * std.time.ns_per_ms, .working = working }, &tap);
}

test "a decoder gives up on a silent producer, unless the producer is known to be working" {
    const a = std.testing.allocator;
    try std.testing.expectError(error.Expired, slow(a, null));
    var idle: std.atomic.Value(bool) = .init(false);
    try std.testing.expectError(error.Expired, slow(a, &idle));
    var working: std.atomic.Value(bool) = .init(true);
    try slow(a, &working);
}

test "a decoder told the producer failed stops, while the export is announced and while it is awaited" {
    const a = std.testing.allocator;
    var arena_state: std.heap.ArenaAllocator = .init(a);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    var imp = try im.Import.init(arena, shape, .{ .@"packed" = try target(arena) }, .{});
    var tap: Tap = .{ .inner = .{ .imp = &imp, .shard = 0 }, .arena = arena, .progress = undefined };
    const failed: std.atomic.Value(bool) = .init(true);
    const slow_ns = 5 * std.time.ns_per_s;
    try std.testing.expectError(error.Aborted, pull(a, arena, .{ .stream = true, .delay_ns = slow_ns, .failed = &failed }, &tap));
    try std.testing.expectEqual(@as(usize, 0), tap.order.items.len);
    try std.testing.expectError(error.Aborted, pull(a, arena, .{ .stream = false, .delay_ns = slow_ns, .failed = &failed }, &tap));
    try std.testing.expectEqual(@as(usize, 0), tap.order.items.len);
}
