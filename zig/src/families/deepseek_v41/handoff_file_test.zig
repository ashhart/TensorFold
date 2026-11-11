//! The hand-off file round-trips an export into the importer byte for byte; bad files are refused as the link would.
const std = @import("std");
const fabric = @import("fabric");
const plan = @import("handoff/plan.zig");
const metal = @import("handoff/metal.zig");
const im = @import("handoff/import.zig");
const p2 = @import("handoff/p2.zig");
const s = @import("handoff/synth.zig");
const cx = @import("cuda_export.zig");
const cxt = @import("cuda_export_test.zig");
const file = @import("handoff_file.zig");

const gpa = std.testing.allocator;
const model = "dsv41-test";

pub const Fixture = struct {
    arena_state: std.heap.ArenaAllocator,
    tokens: []u32,
    built: *cx.Built,
    bytes: []u8,
    shape: plan.Shape,

    pub fn make(upto: u64, ds_rows: u32) !Fixture {
        var f: Fixture = undefined;
        f.arena_state = .init(gpa);
        errdefer f.arena_state.deinit();
        const arena = f.arena_state.allocator();
        const w = try s.make(arena, .{ .upto = upto, .ds_rows = ds_rows, .seed = upto });
        const snap = try cxt.snapshotOf(arena, w);
        f.tokens = try arena.alloc(u32, upto);
        for (f.tokens, 0..) |*t, i| t.* = @intCast(i * 3 + 7);
        f.built = try cx.build(gpa, @splat(0xab), f.tokens, .{ .pos = upto, .ds_rows = ds_rows, .pieces = snap.pieces, .pages = snap.pages });
        errdefer f.built.deinit(gpa);
        var out: std.Io.Writer.Allocating = .init(gpa);
        errdefer out.deinit();
        try file.write(gpa, &out.writer, &f.built.exp, model, 0, 1);
        f.bytes = try out.toOwnedSlice();
        f.shape = .{ .upto = upto, .ds_rows = ds_rows };
        return f;
    }

    pub fn deinit(f: *Fixture) void {
        gpa.free(f.bytes);
        f.built.deinit(gpa);
        f.arena_state.deinit();
    }
};

pub fn target(arena: std.mem.Allocator) !metal.Packed {
    const probe: metal.Packed = .{ .live = &.{}, .storage = &.{}, .layers = 43, .capacity = 1024 };
    return .{ .live = try arena.alloc(u8, probe.liveBytes()), .storage = try arena.alloc(u8, metal.Packed.storageBytes(1024)), .layers = 43, .capacity = 1024 };
}

test "a hand-off file replays into the importer as a direct import of the same parts does" {
    var f = try Fixture.make(300, 128);
    defer f.deinit();
    const arena = f.arena_state.allocator();

    const via = try target(arena);
    var imp = try im.Import.init(arena, f.shape, .{ .@"packed" = via }, .{});
    var sink: p2.ShardSink = .{ .imp = &imp, .shard = 0 };
    var reason: fabric.decoder.Reason = .{};
    const got = try file.pull(arena, f.bytes, .{ .model = model, .tokens = f.tokens, .protocol = 2 }, sink.sink(), &reason);
    try p2.check(arena, f.shape, got.manifest);
    _ = try imp.complete();

    const direct = try target(arena);
    var imp2 = try im.Import.init(arena, f.shape, .{ .@"packed" = direct }, .{});
    for (f.built.exp.layers, f.built.exp.sources) |layer, src| for (layer.parts, src.parts) |part, bytes| {
        const dtype: plan.Dtype = if (part.dtype == .float32) .f32 else .bf16;
        try imp2.put(0, layer.index, part.name, dtype, part.shape[0], part.shape[1], bytes);
    };
    _ = try imp2.complete();
    try std.testing.expectEqualSlices(u8, direct.live, via.live);
    try std.testing.expectEqualSlices(u8, direct.storage, via.storage);

    // the stream is the producer's own: a protocol 2 MANIFEST first, then a DATA message a part
    try std.testing.expectEqualStrings("MKVH", f.bytes[0..4]);
    const head = fabric.wire.unpackUpTo(f.bytes, 2).ok;
    try std.testing.expectEqual(fabric.wire.Kind.manifest, head.kind);
    try std.testing.expectEqual(@as(u16, 2), head.version);
    try std.testing.expectEqual(@as(u32, @intCast(got.frames)), head.frames);
    var parts: u64 = 0;
    for (f.built.exp.layers) |l| parts += l.parts.len;
    try std.testing.expectEqual(parts, got.frames);
    try std.testing.expectEqual(f.built.bytes, got.bytes);
}

test "a file for another prompt, checkpoint or protocol is refused as the link's decoder refuses it" {
    var f = try Fixture.make(40, 0);
    defer f.deinit();
    const arena = f.arena_state.allocator();
    const via = try target(arena);
    var imp = try im.Import.init(arena, f.shape, .{ .@"packed" = via }, .{});
    var sink: p2.ShardSink = .{ .imp = &imp, .shard = 0 };
    var reason: fabric.decoder.Reason = .{};
    const other = try arena.dupe(u32, f.tokens);
    other[3] += 1;
    try std.testing.expectError(error.Mismatch, file.pull(arena, f.bytes, .{ .model = model, .tokens = other, .protocol = 2 }, sink.sink(), &reason));
    try std.testing.expectError(error.Mismatch, file.pull(arena, f.bytes, .{ .model = "other", .tokens = f.tokens, .protocol = 2 }, sink.sink(), &reason));
    try std.testing.expectError(error.Mismatch, file.pull(arena, f.bytes, .{ .model = model, .tokens = f.tokens, .tp_rank = 1, .tp_size = 2, .protocol = 2 }, sink.sink(), &reason));
    try std.testing.expectError(error.Protocol, file.pull(arena, f.bytes, .{ .model = model, .tokens = f.tokens, .protocol = 1 }, sink.sink(), &reason));
}

test "a damaged payload, a cut file and trailing bytes are refused" {
    var f = try Fixture.make(40, 0);
    defer f.deinit();
    const arena = f.arena_state.allocator();
    const via = try target(arena);
    var imp = try im.Import.init(arena, f.shape, .{ .@"packed" = via }, .{});
    var sink: p2.ShardSink = .{ .imp = &imp, .shard = 0 };
    var reason: fabric.decoder.Reason = .{};
    const expect: fabric.decoder.Expect = .{ .model = model, .tokens = f.tokens, .protocol = 2 };

    const flipped = try arena.dupe(u8, f.bytes);
    flipped[flipped.len - 1] ^= 0x40; // the last payload byte of the last frame
    try std.testing.expectError(error.Corrupt, file.pull(arena, flipped, expect, sink.sink(), &reason));
    try std.testing.expectError(error.Protocol, file.pull(arena, f.bytes[0 .. f.bytes.len - 1], expect, sink.sink(), &reason));
    try std.testing.expectError(error.Protocol, file.pull(arena, f.bytes[0..fabric.wire.header_bytes], expect, sink.sink(), &reason));
    try std.testing.expectError(error.Protocol, file.pull(arena, &.{}, expect, sink.sink(), &reason));
    const longer = try arena.alloc(u8, f.bytes.len + 1);
    @memcpy(longer[0..f.bytes.len], f.bytes);
    longer[f.bytes.len] = 0;
    try std.testing.expectError(error.Protocol, file.pull(arena, longer, expect, sink.sink(), &reason));
}

test "a producer's refusal in the file is reported as the decoder reports it" {
    var buf: [fabric.wire.header_bytes + 9]u8 = undefined;
    buf[0..fabric.wire.header_bytes].* = fabric.wire.pack(.{ .version = 2, .kind = .@"error", .handoff = @splat(1), .nbytes = 9 });
    @memcpy(buf[fabric.wire.header_bytes..], "no export");
    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    var imp: im.Import = undefined;
    var sink: p2.ShardSink = .{ .imp = &imp, .shard = 0 };
    var reason: fabric.decoder.Reason = .{};
    try std.testing.expectError(error.Refused, file.pull(arena, &buf, .{ .model = model, .tokens = &.{1}, .protocol = 2 }, sink.sink(), &reason));
    try std.testing.expectEqualStrings("no export", reason.text());
}

test "the file's name is the hand-off id" {
    var buf: [32 + file.extension.len]u8 = undefined;
    const id: [16]u8 = .{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 255 };
    try std.testing.expectEqualStrings("000102030405060708090a0b0c0d0eff.mkvh2", file.fileName(&buf, id));
}
