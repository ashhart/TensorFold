//! Actual family admission and splice dispatch with virtual device allocations, separate from model arithmetic.
const std = @import("std");
const cuda = @import("cuda");
const lanes = @import("lanes");
const prompt = @import("cuda_prompt.zig");
const mtp = @import("cuda_mtp.zig");
const images = @import("cuda_images.zig");
const fixtures = @import("cuda_lanes_restore_test.zig");
const shared = @import("qwen_vision");
const Fake = struct {
    var calls: usize = 0;
    var launches: usize = 0;
    var first_at: i32 = 0;
    fn upload(_: u64, _: ?*const anyopaque, _: usize) callconv(.c) c_int {
        return 0;
    }
    fn encode(_: *anyopaque, pixels: []const f32, grids: []const shared.Grid, _: u64) anyerror!void {
        try std.testing.expectEqual(@as(usize, grids[0].h) * grids[0].w * shared.patch_width, pixels.len);
        try std.testing.expectEqual(@as(f32, 0.25), pixels[0]);
        try std.testing.expect(grids[0].h == 2 or grids[0].h == 4);
        calls += 1;
    }
    fn launch(_: cuda.abi.Function, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: cuda.abi.Stream, args: ?[*]?*anyopaque, _: ?[*]?*anyopaque) callconv(.c) c_int {
        first_at = @as(*const i32, @ptrCast(@alignCast(args.?[3].?))).*;
        launches += 1;
        return 0;
    }
};
fn run(comptime Cuda: type, comptime open: anytype, comptime drop: anytype, drafting: bool, cached: ?usize) !void {
    const a = std.testing.allocator;
    Fake.calls = 0;
    Fake.launches = 0;
    var driver = fixtures.fakeDriver();
    driver.api.cuMemcpyHtoD_v2 = Fake.upload;
    driver.api.cuLaunchKernel = Fake.launch;
    var stream: cuda.Stream = .{ .d = &driver, .handle = null };
    const f: cuda.Function = .{ .d = &driver, .handle = null };
    var engine: prompt.Engine = undefined;
    engine.gpa = a;
    engine.driver = &driver;
    engine.stream = &stream;
    engine.chunk = 16;
    engine.eos = 7;
    engine.round_graph = .{};
    var ids = [_]u32{1};
    var head: mtp.Head = undefined;
    head.ids = &ids;
    var v: images.Vision = undefined;
    v.a = a;
    v.limits = .{};
    v.prep = f;
    v.pool = f;
    v.encoder = .{ .ctx = &v, .driver = &driver, .stream = stream, .splice = f, .encode = Fake.encode, .config = .{ .hidden = 1024, .heads = 16, .depth = 24, .intermediate = 4096, .out_hidden = 2560, .image_token = 50, .video_token = 51, .vision_start = 52, .vision_end = 53 } };
    var self: Cuda = undefined;
    self.gpa = a;
    self.context = 64;
    self.engine = &engine;
    self.stream = stream;
    self.ranks = null;
    self.head = if (drafting) &head else null;
    self.vision = &v;
    self.lanes = .empty;
    defer {
        var it = self.lanes.valueIterator();
        while (it.next()) |l| drop(&self, l);
        self.lanes.deinit(a);
    }
    const side: u32 = if (cached != null) 4 else 2;
    const pixels = try a.alloc(f32, @as(usize, side) * side * shared.patch_width);
    defer a.free(pixels);
    @memset(pixels, 0.25);
    const Snap = @import("cuda_snap.zig").Snap;
    var kept: Snap = .{ .buf = .{ .d = &driver, .ptr = 0x100, .len = 1, .borrowed = true }, .at = cached orelse 1, .history = .{ 8, 9 }, .head = drafting };
    const tokens: []const u32 = if (cached != null) &.{ 7, 52, 50, 50, 50, 50, 53, 9 } else &.{ 7, 52, 50, 53, 9 };
    var s = try lanes.Stream.init(a, .{ .id = "image", .prompt = tokens, .max_new = 4, .drafts = drafting, .reuse = if (cached) |at| .{ .saved = &kept, .at = @intCast(at) } else .{}, .images = &.{.{ .pixels = pixels, .gh = side, .gw = side, .tokens = side * side / 4, .at = 2 }} });
    defer s.deinit(a);
    const l = try open(&self, &s);
    try std.testing.expectEqual(@as(usize, 1), Fake.calls);
    const view = l.seq.view(0).media.?;
    try std.testing.expectEqual(@as(usize, side) * side / 4, l.seq.media.?.image.plan.rows);
    if (cached) |at| {
        try std.testing.expectEqual(at, l.seq.fed);
        try std.testing.expectEqual(@as(u32, @intCast(at)), s.cached);
        try std.testing.expect(!s.reuse_failed);
        try std.testing.expectEqual([2]i64{ 8, 9 }, l.seq.history);
        if (l.draft) |d| try std.testing.expectEqual(at, d.len);
    }
    try std.testing.expectEqual(@as(i32, 0), l.seq.media.?.image.plan.src[2]);
    try view.image.inject(stream, 0x123400, 1, 3, 4);
    try std.testing.expectEqual(@as(usize, 1), Fake.launches);
    try std.testing.expectEqual(@as(i32, 1), Fake.first_at);
    try view.image.inject(stream, 0x123400, tokens.len, 1, 4);
    try std.testing.expectEqual(@as(usize, 1), Fake.launches);
    const media = @import("cuda_media.zig");
    const Round = struct { seq: struct { media: ?media.View }, pos: usize };
    const rows = [_]Round{ .{ .seq = .{ .media = view }, .pos = tokens.len }, .{ .seq = .{ .media = view }, .pos = tokens.len + 1 } };
    try media.inject(view, &rows, stream, 0x123400, 0, 2);
    try std.testing.expectEqual(@as(usize, 1), Fake.launches);
    if (drafting) {
        try std.testing.expect(l.draft != null);
        try std.testing.expectEqual(view.image.rope, l.draft.?.media.?.image.rope);
        try std.testing.expectEqual(view.image.features, l.draft.?.media.?.image.features);
    }
}
// A lane release whose stream wait fails once while its rounds are still pending; frees during that window count.
const Wait = struct {
    var pending: bool = false;
    var fail_waits: usize = 0;
    var bad_free: usize = 0;
    fn free(_: u64) callconv(.c) c_int {
        if (pending) bad_free += 1;
        return 0;
    }
    fn sync(_: cuda.abi.Stream) callconv(.c) c_int {
        if (pending and fail_waits > 0) {
            fail_waits -= 1;
            return cuda.abi.error_not_ready;
        }
        pending = false;
        return 0;
    }
    fn context() callconv(.c) c_int {
        pending = false;
        return 0;
    }
};
fn release(comptime Cuda: type, comptime open: anytype) !void {
    const a = std.testing.allocator;
    var driver = fixtures.fakeDriver();
    driver.api.cuMemFree_v2 = Wait.free;
    driver.api.cuStreamSynchronize = Wait.sync;
    driver.api.cuCtxSynchronize = Wait.context;
    var stream: cuda.Stream = .{ .d = &driver, .handle = null };
    var engine: prompt.Engine = undefined;
    engine.gpa = a;
    engine.driver = &driver;
    engine.stream = &stream;
    engine.chunk = 16;
    engine.eos = 7;
    engine.round_graph = .{};
    var self: Cuda = undefined;
    self.gpa = a;
    self.context = 64;
    self.engine = &engine;
    self.stream = stream;
    self.ranks = null;
    self.head = null;
    self.vision = null;
    self.lanes = .empty;
    defer self.lanes.deinit(a);
    var s = try lanes.Stream.init(a, .{ .id = "release", .prompt = &.{ 1, 2, 3 }, .max_new = 1 });
    defer s.deinit(a);
    _ = try open(&self, &s);
    Wait.pending = true;
    Wait.fail_waits = 1;
    Wait.bad_free = 0;
    self.backend().vtable.release(&self, &s);
    try std.testing.expect(!Wait.pending);
    try std.testing.expectEqual(@as(usize, 0), Wait.bad_free);
}
pub fn Cases(comptime Cuda: type, comptime open: anytype, comptime drop: anytype) type {
    return struct {
        test "an image lane restores kept prefixes that end before, inside and after its image, with and without MTP" {
            for ([_]bool{ false, true }) |drafting| for ([_]usize{ 1, 3, 6 }) |at| try run(Cuda, open, drop, drafting, at);
        }
        test "image rows reach a plain family lane and the CUDA splice" {
            try run(Cuda, open, drop, false, null);
        }
        test "image rows and their M-RoPE positions reach a drafted family lane" {
            try run(Cuda, open, drop, true, null);
        }
        test "a lane release whose stream wait fails drains before the sequence is freed" {
            try release(Cuda, open);
        }
    };
}
