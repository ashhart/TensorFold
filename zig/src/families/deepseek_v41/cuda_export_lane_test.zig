//! Rank 0 of a split pool: a refused HTTP export allocates nothing; an export the followers missed is refused whole.
const std = @import("std");
const api = @import("engine_api");
const lane = @import("cuda_export_lane.zig");
const serve = @import("cuda_export_serve.zig");
const cx = @import("cuda_export.zig");
const cxt = @import("cuda_export_test.zig");
const s = @import("handoff/synth.zig");
const Forward = @import("forward.zig").Forward;

const gpa = std.testing.allocator;
const io = std.testing.io;

/// A synthetic world's state handed out as the engine's read-back; counts what comes back.
const Source = struct {
    snap: cxt.Snap,
    taken: u32 = 0,
    released: u32 = 0,

    fn take(ptr: *anyopaque, _: std.mem.Allocator, at: u64) anyerror!cx.Captured {
        const src: *Source = @ptrCast(@alignCast(ptr));
        src.taken += 1;
        return .{ .ctx = src, .state = .{ .pos = at, .ds_rows = @min(at, 128), .pieces = src.snap.pieces, .pages = src.snap.pages } };
    }

    fn release(ctx: *anyopaque) void {
        const src: *Source = @ptrCast(@alignCast(ctx));
        src.released += 1;
    }
};

/// Followers the leader cannot reach: the dispatch fails, and the failures sent after it are counted.
const Unreached = struct {
    var fails: u32 = 0;

    fn send(_: *Forward, _: [16]u8, _: u64) anyerror!void {
        return error.BrokenPipe;
    }

    fn fail(_: *Forward, _: [16]u8) anyerror!void {
        fails += 1;
    }
};

test "an HTTP producer over a split pool is refused before its exporter is allocated" {
    var problem: []const u8 = "";
    const at: lane.Export.Where = .{ .link = api.LaneHost.Remote.http_link, .dump = "", .cap = 1, .checkpoint = "dsv41-test" };
    try std.testing.expectEqual(@as(?*lane.Export, null), try lane.Export.openOver(gpa, io, undefined, undefined, 2, at, &problem));
    try std.testing.expect(std.mem.indexOf(u8, problem, "TF_DSV41_KV_SPLIT") != null);
}

test "a split export whose followers were not asked is refused on rank 0 as well, and its state is released" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    const a = arena_state.allocator();
    const dump = try tmp.dir.realPathFileAlloc(io, ".", a);
    const w = try s.make(a, .{ .upto = 130, .ds_rows = 128, .seed = 130 });
    var src: Source = .{ .snap = try cxt.snapshotOf(a, w) };
    var problem: []const u8 = "";
    const p = try serve.Produce.open(gpa, io, .{ .ptr = &src, .take = Source.take, .release = Source.release }, .{ .dump = dump, .model = "dsv41-test", .shards = 2 }, &problem);
    defer p.stop();
    var x: lane.Export = .{ .gpa = gpa, .source = undefined, .produce = p, .f = undefined, .split = true, .followers = .{ .send = Unreached.send, .fail = Unreached.fail } };
    const e = x.exporter();
    const tokens = try a.alloc(u32, 130);
    for (tokens, 0..) |*t, i| t.* = @intCast(i * 11 + 2);
    const id: [16]u8 = @splat(0x5a);
    Unreached.fails = 0;
    e.exported(e.ptr, .{ .id = id }, tokens, try e.snapshots.vtable.save(e.snapshots.ptr, null, 130));

    const cwd = std.Io.Dir.cwd();
    try std.testing.expectError(error.FileNotFound, cwd.access(io, try std.fmt.allocPrint(a, "{s}/{x}.mkvh2", .{ dump, &id }), .{}));
    try std.testing.expectError(error.FileNotFound, cwd.access(io, try std.fmt.allocPrint(a, "{s}/{x}.tokens.json", .{ dump, &id }), .{}));
    const why = try cwd.readFileAlloc(io, try std.fmt.allocPrint(a, "{s}/{x}.error", .{ dump, &id }), a, .limited(4096));
    try std.testing.expect(std.mem.indexOf(u8, why, "followers") != null);
    try std.testing.expectEqual(@as(u32, 1), Unreached.fails); // the followers that did hear refuse their shards too
    try std.testing.expectEqual(src.taken, src.released);
}
