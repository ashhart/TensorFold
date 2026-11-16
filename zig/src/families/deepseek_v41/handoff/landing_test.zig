//! A hand-off file fed through the landing: ring parts are dropped, everything else lands as a direct import does.
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

fn target(arena: std.mem.Allocator) !metal.Packed {
    const probe: metal.Packed = .{ .live = &.{}, .storage = &.{}, .layers = 40, .capacity = capacity };
    const pk: metal.Packed = .{ .live = try arena.alloc(u8, probe.liveBytes()), .storage = try arena.alloc(u8, metal.Packed.storageBytes(capacity)), .layers = 40, .capacity = capacity };
    @memset(pk.storage, 0);
    return pk;
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
