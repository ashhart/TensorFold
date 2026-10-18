//! A two-rank verification window agrees with serial rows and rolls back at partial and full acceptance seams.
const std = @import("std");
const cuda = @import("cuda");
const flash = @import("flashnext");
const check = @import("check.zig");
const prompt = flash.prompt;
const mtp = flash.mtp;

pub fn run(gpu: check.Gpu, dir: []const u8, rank: usize, port: u16, ids_path: []const u8) !void {
    const context = 256;
    const source = try std.Io.Dir.cwd().readFileAlloc(gpu.io, ids_path, gpu.gpa, .limited(1 << 20));
    defer gpu.gpa.free(source);
    var ids: std.ArrayList(u32) = .empty;
    defer ids.deinit(gpu.gpa);
    var words = std.mem.tokenizeAny(u8, source, " \r\n\t");
    while (words.next()) |word| try ids.append(gpu.gpa, try std.fmt.parseInt(u32, word, 10));
    if (ids.items.len == 0 or ids.items.len > 128) return error.BadPrompt;
    var stream = try cuda.Stream.init(gpu.d, true);
    defer stream.deinit();
    var glue = try flash.Glue.load(gpu.d);
    defer glue.deinit();
    const tri: flash.Tri = .{ .s = stream, .own = &glue };
    var identity = try flash.rank_engine.identity(gpu.gpa, gpu.io, dir, context, 32);
    identity.mtp = 1;
    const session = try flash.rank_engine.Session.init(gpu.gpa, gpu.io, "127.0.0.1", port, rank, identity);
    const e = prompt.Engine.initRank(gpu.gpa, gpu.io, @constCast(gpu.d), &stream, dir, 2, rank) catch |err| {
        session.poison();
        session.deinit();
        return err;
    };
    session.engine = e;
    e.collective = &session.collective;
    defer {
        session.deinit();
        e.deinit();
    }
    if (e.chunk != 32) return error.RankIdentity;
    var logits = try cuda.DeviceBuffer.alloc(gpu.d, 2 * prompt.vocabulary * 2);
    defer logits.free();
    if (rank == 1) {
        try session.follow(flash.Tri, tri, logits.ptr);
        check.pass("rank 1 verification windows and prefix commits completed", .{});
        return;
    }
    var serial = try session.create(context);
    defer session.drop(&serial);
    try session.forward(flash.Tri, tri, &serial, ids.items, .{});
    const base = try session.save(&serial);
    defer session.dropSaved(base);
    const forced = [_]u32{ 37, 44, 102, 1, 320, 17, 19, 71, 99, 4, 80, 122, 7, 25, 61, 94 };
    const row_bytes = prompt.vocabulary * 2;
    const expected = try gpu.gpa.alloc(u8, forced.len * row_bytes);
    defer gpu.gpa.free(expected);
    for (forced, 0..) |token, i| {
        try session.forward(flash.Tri, tri, &serial, &.{token}, .{});
        @memcpy(expected[i * row_bytes ..][0..row_bytes], serial.logits);
    }
    var window = try session.create(context);
    defer session.drop(&window);
    var draft = try mtp.Draft.init(gpu.gpa, e.driver, stream, context, 1, 32);
    defer draft.deinit();
    const rows = try gpu.gpa.alloc(u8, expected.len);
    defer gpu.gpa.free(rows);
    for ([_]usize{ 1, 3, 7, 15, 16 }) |keep| {
        try session.restore(&window, base);
        try session.verify(flash.Tri, tri, &window, &draft, &forced);
        try stream.synchronize();
        try e.driver.check(e.driver.api.cuMemcpyDtoH_v2(rows.ptr, draft.logits.ptr, rows.len), "cuMemcpyDtoH");
        try check.sameBytes("each two-rank window row equals its serial teacher-forced row", rows, expected);
        try session.keep(flash.Tri, tri, &window, &draft, keep);
        try check.expect(window.fed == ids.items.len + keep, "kept prefix position is exact", .{});
        try session.restore(&serial, base);
        for (forced[0..keep]) |token| try session.forward(flash.Tri, tri, &serial, &.{token}, .{});
        try check.expect(std.mem.eql(i64, &window.history, &serial.history), "kept host n-gram history matches serial", .{});
        try session.forward(flash.Tri, tri, &serial, &.{13}, .{});
        try session.forward(flash.Tri, tri, &window, &.{13}, .{});
        try check.sameBytes("rejected suffix cannot change the next row after prefix commit", window.logits, serial.logits);
        check.pass("two-rank forced window: all rows and next row byte-exact at keep={d}/{d}", .{ keep, forced.len });
    }
}
