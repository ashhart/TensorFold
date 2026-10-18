//! Full-checkpoint experiment: two-rank prompt, snapshot replay and concurrent decode agree byte for byte.
const std = @import("std");
const cuda = @import("cuda");
const flash = @import("flashnext");
const check = @import("check.zig");
const prompt = flash.prompt;
const ranks = flash.rank_engine;

pub fn run(gpu: check.Gpu, dir: []const u8, rank: usize, port: u16, ids_path: []const u8) !void {
    const context = 256;
    const chunk = 32;
    const source = try std.Io.Dir.cwd().readFileAlloc(gpu.io, ids_path, gpu.gpa, .limited(1 << 20));
    defer gpu.gpa.free(source);
    var ids: std.ArrayList(u32) = .empty;
    defer ids.deinit(gpu.gpa);
    var pieces = std.mem.tokenizeAny(u8, source, " \r\n\t");
    while (pieces.next()) |word| try ids.append(gpu.gpa, try std.fmt.parseInt(u32, word, 10));
    if (ids.items.len == 0 or ids.items.len > context / 2) return error.BadPrompt;
    var stream = try cuda.Stream.init(gpu.d, true);
    defer stream.deinit();
    var glue = try flash.Glue.load(gpu.d);
    defer glue.deinit();
    const tri: flash.Tri = .{ .s = stream, .own = &glue };
    const identity = try ranks.identity(gpu.gpa, gpu.io, dir, context, chunk);
    const session = try ranks.Session.init(gpu.gpa, gpu.io, "127.0.0.1", port, rank, identity);
    const engine = prompt.Engine.initRank(gpu.gpa, gpu.io, @constCast(gpu.d), &stream, dir, 2, rank) catch |err| {
        session.poison();
        session.deinit();
        return err;
    };
    session.engine = engine;
    defer {
        session.deinit();
        engine.deinit();
    }
    engine.collective = &session.collective;
    if (engine.chunk != chunk) return error.RankIdentity;
    const room = try session.room((try gpu.ctx.memInfo()).free);
    try check.expect(room > prompt.seqBytes(context) * 3, "both ranks have room for experimental caches", .{});
    var logits = try cuda.DeviceBuffer.alloc(gpu.d, 2 * prompt.vocabulary * 2);
    defer logits.free();
    if (rank == 1) {
        try session.follow(flash.Tri, tri, logits.ptr);
        check.pass("rank 1 replay completed", .{});
        return;
    }
    var first = try session.create(context);
    defer session.drop(&first);
    try session.forward(flash.Tri, tri, &first, ids.items, .{});
    check.pass("two-rank prompt completed ({d} tokens)", .{ids.items.len});
    for (0..prompt.vocabulary) |i| {
        const value: f32 = @bitCast(@as(u32, std.mem.readInt(u16, first.logits[i * 2 ..][0..2], .little)) << 16);
        try check.expect(std.math.isFinite(value), "finite initial logits at {d}", .{i});
    }
    const kept = try session.save(&first);
    defer session.dropSaved(kept);
    const token = prompt.argmax(first.logits);
    try session.forward(flash.Tri, tri, &first, &.{token}, .{});
    const expected = try gpu.gpa.dupe(u8, first.logits);
    defer gpu.gpa.free(expected);
    try session.restore(&first, kept);
    try session.forward(flash.Tri, tri, &first, &.{token}, .{});
    try check.sameBytes("two-rank snapshot restore and replay", first.logits, expected);
    try session.restore(&first, kept);
    var second = try session.create(context);
    defer session.drop(&second);
    try session.forward(flash.Tri, tri, &second, ids.items, .{});
    try session.round(flash.Tri, tri, &.{ &first, &second }, &.{ token, token }, logits.ptr);
    try check.sameBytes("two-rank concurrent first stream", first.logits, expected);
    try check.sameBytes("two-rank concurrent second stream", second.logits, expected);
    for (0..32) |_| {
        const next = prompt.argmax(first.logits);
        std.debug.print("TOKEN {d}\n", .{next});
        try session.round(flash.Tri, tri, &.{&first}, &.{next}, logits.ptr);
    }
    check.pass("two-rank full-model prompt, snapshot replay, concurrent byte equality and 32 decode rows", .{});
}
