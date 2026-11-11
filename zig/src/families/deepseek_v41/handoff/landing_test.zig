//! A hand-off file fed through the landing: everything lands as a direct import does; rings if the slot holds them.
const std = @import("std");
const g = @import("geometry.zig");
const plan = @import("plan.zig");
const ex = @import("export.zig");
const metal = @import("metal.zig");
const im = @import("import.zig");
const blob = @import("blob.zig");
const landing = @import("landing.zig");
const s = @import("synth.zig");
const fixture = @import("../handoff_file_test.zig");
const fabric = @import("fabric");

const capacity: u32 = 1024;

/// A blank Packed slot of `layers` rings: 40 (the backbone's) or 43 (with the drafter's).
fn slot(arena: std.mem.Allocator, layers: u32) !metal.Packed {
    const probe: metal.Packed = .{ .live = &.{}, .storage = &.{}, .layers = layers, .capacity = capacity };
    const pk: metal.Packed = .{ .live = try arena.alloc(u8, probe.liveBytes()), .storage = try arena.alloc(u8, metal.Packed.storageBytes(capacity)), .layers = layers, .capacity = capacity };
    @memset(pk.storage, 0);
    return pk;
}

fn target(arena: std.mem.Allocator) !metal.Packed {
    return slot(arena, 40);
}

fn tokensOf(arena: std.mem.Allocator, n: usize) ![]u32 {
    const t = try arena.alloc(u32, n);
    for (t, 0..) |*v, i| v.* = @intCast(i * 5 + 3);
    return t;
}

test "a hand-off file lands like a direct import, and its drafter rings are read and dropped" {
    for ([_]struct { upto: u64, ds_rows: u32 }{ .{ .upto = 300, .ds_rows = 128 }, .{ .upto = 131, .ds_rows = 0 }, .{ .upto = 512, .ds_rows = 40 } }) |cs| {
        var arena_state: std.heap.ArenaAllocator = .init(std.testing.allocator);
        defer arena_state.deinit();
        const arena = arena_state.allocator();
        const w = try s.make(arena, .{ .upto = cs.upto, .ds_rows = cs.ds_rows, .seed = cs.upto });
        const shape: plan.Shape = .{ .upto = cs.upto, .ds_rows = cs.ds_rows };
        const parts = try ex.build(arena, shape, w.cuda);
        const tokens = try tokensOf(arena, @intCast(cs.upto));
        const file = try blob.write(arena, "dsv41-test", tokens, parts);

        const parsed = try blob.Blob.parse(arena, file);
        try std.testing.expectEqualSlices(u32, tokens, parsed.tokens);
        try std.testing.expectEqual(cs.upto, parsed.manifest.prompt_tokens);

        const via = try target(arena);
        var l: landing.Landing = undefined;
        try l.init(arena, cs.upto, .{ .@"packed" = via });
        defer l.deinit();
        try parsed.feed(l.decoderSink());
        _ = try l.complete();
        try std.testing.expectEqual(@as(u32, if (cs.ds_rows > 0) 3 else 0), l.dropped);

        const direct = try target(arena);
        var imp = try im.Import.init(arena, .{ .upto = cs.upto }, .{ .@"packed" = direct }, .{});
        var buf: [40]u8 = undefined;
        for (try ex.build(arena, .{ .upto = cs.upto }, w.cuda)) |p| try imp.put(0, p.spec.layer, p.spec.name(&buf), p.spec.dtype, p.spec.rows, p.spec.cols, p.bytes);
        _ = try imp.complete();
        try std.testing.expectEqualSlices(u8, direct.live, via.live);
        try std.testing.expectEqualSlices(u8, direct.storage, via.storage);
    }
}

test "a hand-off file refuses a corrupt header, a short body and tokens that do not hash to the manifest" {
    var arena_state: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const w = try s.make(arena, .{ .upto = 20 });
    const tokens = try tokensOf(arena, 20);
    const file = try blob.write(arena, "dsv41-test", tokens, try ex.build(arena, .{ .upto = 20 }, w.cuda));
    _ = try blob.Blob.parse(arena, file);
    try std.testing.expectError(error.BadBlob, blob.Blob.parse(arena, file[0 .. file.len - 1]));
    try std.testing.expectError(error.BadBlob, blob.Blob.parse(arena, file[0..10]));
    const bad = try arena.dupe(u8, file);
    bad[0] = 'X';
    try std.testing.expectError(error.BadBlob, blob.Blob.parse(arena, bad));
    const other = try arena.dupe(u8, file);
    other[20] ^= 1; // the first token
    try std.testing.expectError(error.BadBlob, blob.Blob.parse(arena, other));
}

test "a producer's stream becomes a Blob that lands as the stream's own parts do, and a stream for another prompt is refused" {
    var fx = try fixture.Fixture.make(300, 128);
    defer fx.deinit();
    var arena_state: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    var reason: fabric.decoder.Reason = .{};
    const b = try blob.fromStream(arena, fx.bytes, .{ .model = "dsv41-test", .tokens = fx.tokens, .protocol = 2 }, &reason);
    try std.testing.expectEqual(@as(u64, 300), b.manifest.prompt_tokens);

    const via = try target(arena);
    var l: landing.Landing = undefined;
    try l.init(arena, 300, .{ .@"packed" = via });
    defer l.deinit();
    try b.feed(l.decoderSink());
    _ = try l.complete();
    try std.testing.expectEqual(@as(u32, 3), l.dropped);

    const direct = try target(arena);
    var imp = try im.Import.init(arena, .{ .upto = 300 }, .{ .@"packed" = direct }, .{});
    for (fx.built.exp.layers, fx.built.exp.sources) |layer, src| if (layer.index < g.release.layers) for (layer.parts, src.parts) |part, bytes| {
        const dtype: plan.Dtype = if (part.dtype == .float32) .f32 else .bf16;
        try imp.put(0, layer.index, part.name, dtype, part.shape[0], part.shape[1], bytes);
    };
    _ = try imp.complete();
    try std.testing.expectEqualSlices(u8, direct.live, via.live);
    try std.testing.expectEqualSlices(u8, direct.storage, via.storage);

    var other = try arena.dupe(u32, fx.tokens);
    other[3] += 1;
    try std.testing.expectError(error.Mismatch, blob.fromStream(arena, fx.bytes, .{ .model = "dsv41-test", .tokens = other, .protocol = 2 }, &reason));
}

test "a hand-off file's drafter rings land in a slot that has them, as a direct import lands them, and all or none" {
    for ([_]struct { upto: u64, ds_rows: u32 }{ .{ .upto = 300, .ds_rows = 128 }, .{ .upto = 131, .ds_rows = 0 }, .{ .upto = 512, .ds_rows = 40 } }) |cs| {
        var arena_state: std.heap.ArenaAllocator = .init(std.testing.allocator);
        defer arena_state.deinit();
        const arena = arena_state.allocator();
        const w = try s.make(arena, .{ .upto = cs.upto, .ds_rows = cs.ds_rows, .seed = cs.upto });
        const shape: plan.Shape = .{ .upto = cs.upto, .ds_rows = cs.ds_rows };
        const parts = try ex.build(arena, shape, w.cuda);
        const tokens = try tokensOf(arena, @intCast(cs.upto));
        const parsed = try blob.Blob.parse(arena, try blob.write(arena, "dsv41-test", tokens, parts));

        const via = try slot(arena, 43);
        var l: landing.Landing = undefined;
        try l.init(arena, cs.upto, .{ .@"packed" = via });
        defer l.deinit();
        try parsed.feed(l.decoderSink());
        const got = try l.complete();
        try std.testing.expectEqual(@as(u32, 0), l.dropped);
        try std.testing.expectEqual(@as(u8, if (cs.ds_rows > 0) 7 else 0), l.rings);
        try std.testing.expectEqual(cs.upto - cs.ds_rows, got.ds_first);
        try std.testing.expectEqual(cs.upto, got.ds_end);
        try std.testing.expect(!got.drafter_dropped);

        const direct = try slot(arena, 43);
        var imp = try im.Import.init(arena, shape, .{ .@"packed" = direct }, .{});
        var buf: [40]u8 = undefined;
        for (parts) |p| try imp.put(0, p.spec.layer, p.spec.name(&buf), p.spec.dtype, p.spec.rows, p.spec.cols, p.bytes);
        const want = try imp.complete();
        try std.testing.expectEqual(want.ds_first, got.ds_first);
        try std.testing.expectEqualSlices(u8, direct.live, via.live);
        try std.testing.expectEqualSlices(u8, direct.storage, via.storage);
        const rings = via.live[40 * metal.Packed.window * metal.Packed.ring_stride .. 43 * metal.Packed.window * metal.Packed.ring_stride];
        try std.testing.expectEqual(cs.ds_rows == 0, std.mem.allEqual(u8, rings, 0)); // the drafter's bytes are in the slot
        if (cs.ds_rows > 0) { // a slot with no drafter rings reads and drops them, and says so
            const small = try slot(arena, 40);
            var l3: landing.Landing = undefined;
            try l3.init(arena, cs.upto, .{ .@"packed" = small });
            defer l3.deinit();
            try parsed.feed(l3.decoderSink());
            try std.testing.expectEqual(@as(u32, 3), l3.dropped);
            try std.testing.expect((try l3.complete()).drafter_dropped);
        }
    }
}

test "a drafter ring part that is not whole, repeated, or for a stage that does not exist is refused" {
    var arena_state: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const w = try s.make(arena, .{ .upto = 300, .ds_rows = 128 });
    const parts = try ex.build(arena, .{ .upto = 300, .ds_rows = 128 }, w.cuda);
    const parsed = try blob.Blob.parse(arena, try blob.write(arena, "dsv41-test", try tokensOf(arena, 300), parts));
    const via = try slot(arena, 43);
    var l: landing.Landing = undefined;
    try l.init(arena, 300, .{ .@"packed" = via });
    defer l.deinit();
    const first = for (parsed.manifest.layers) |*layer| {
        if (layer.index == g.release.layers) break layer;
    } else unreachable;
    const bytes = try arena.alloc(u8, 128 * 1024);
    @memset(bytes, 0);
    const sink = l.decoderSink();
    try std.testing.expectError(error.BadPart, sink.put_fn(sink.ptr, first, 0, 1, bytes[0 .. 127 * 1024])); // a short part
    try sink.put_fn(sink.ptr, first, 0, 1, bytes);
    try std.testing.expectError(error.BadPart, sink.put_fn(sink.ptr, first, 0, 1, bytes)); // the same stage again
    try std.testing.expectError(error.Incomplete, l.complete()); // one stage of three: no ring is ever half there
}
