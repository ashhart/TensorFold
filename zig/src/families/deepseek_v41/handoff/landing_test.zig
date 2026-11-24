//! A hand-off file fed through the landing: ring parts are dropped, everything else lands as a direct import does.
const std = @import("std");
const g = @import("geometry.zig");
const c = @import("codec.zig");
const plan = @import("plan.zig");
const ex = @import("export.zig");
const metal = @import("metal.zig");
const im = @import("import.zig");
const blob = @import("blob.zig");
const landing = @import("landing.zig");
const s = @import("synth.zig");

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

test "a pooled row the served lane cannot hold exactly is refused, while the bf16 release lane takes it" {
    var arena_state: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const upto: u64 = 20;
    const w = try s.make(arena, .{ .upto = upto, .seed = 3 });
    const parts = try ex.build(arena, .{ .upto = upto }, w.cuda);
    const pk = try target(arena);
    var imp = try im.Import.init(arena, .{ .upto = upto }, .{ .@"packed" = pk }, .{});
    var buf: [40]u8 = undefined;
    var offered: usize = 0;
    for (parts) |p| {
        var bytes = p.bytes;
        if (p.spec.kind == .comp) { // a whole first block of 1.25: off the E2M1 grid at its own scale
            bytes = try arena.dupe(u8, p.bytes);
            for (0..16) |i| std.mem.writeInt(u16, bytes[2 * i ..][0..2], c.f32ToBf16(1.25), .little);
            offered += 1;
            try std.testing.expectError(error.NotExact, imp.put(0, p.spec.layer, p.spec.name(&buf), p.spec.dtype, p.spec.rows, p.spec.cols, bytes));
            continue;
        }
        try imp.put(0, p.spec.layer, p.spec.name(&buf), p.spec.dtype, p.spec.rows, p.spec.cols, bytes);
    }
    try std.testing.expect(offered > 0);
    try std.testing.expectError(error.Incomplete, imp.complete()); // the refused part never landed

    // the release lane stores the same row as bf16: the exact consumer is not broken by the refusal
    const layers = try arena.alloc(metal.Release.Layer, g.release.layers);
    for (layers, 0..) |*l, i| {
        l.* = .{ .window = try arena.alloc(u8, 2 * 144 * 1024) };
        if (g.release.sourceIndex(@intCast(i))) |si| {
            const rows: usize = @intCast(g.release.rowsOf(si, upto) + 1);
            l.* = .{ .window = l.window, .pool = try arena.alloc(u8, rows * 1024), .keys = try arena.alloc(u8, rows * 68), .carry = try arena.alloc(u8, 4 * 2048) };
        }
    }
    var rel = try im.Import.init(arena, .{ .upto = upto }, .{ .release = .{ .layers = layers } }, .{});
    for (parts) |p| if (p.spec.kind == .comp) {
        var bytes = try arena.dupe(u8, p.bytes);
        for (0..16) |i| std.mem.writeInt(u16, bytes[2 * i ..][0..2], c.f32ToBf16(1.25), .little);
        try rel.put(0, p.spec.layer, p.spec.name(&buf), p.spec.dtype, p.spec.rows, p.spec.cols, bytes);
    };
}
