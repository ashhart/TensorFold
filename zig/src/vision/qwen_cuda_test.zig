//! The shared tower's failures: a failed upload, plan, launch or wait drains the stream before any owner is freed.
const std = @import("std");
const cuda = @import("cuda");
const v = @import("qwen_cuda.zig");
const M = struct {
    const encoder: cuda.abi.Stream = @ptrFromInt(0xe0); // the tower's stream, apart from the default stream's null
    var pending_default: bool = false; // a copy on the default stream that no wait has proven complete
    var pending_encoder: bool = false;
    var bad_free: usize = 0;
    var copies: usize = 0;
    var fail_copy: usize = 0;
    var waits: usize = 0;
    var fail_waits: usize = 0;
    var mode: usize = 0;
    var live: usize = 0;
    var htod_pends: bool = false; // a synchronous upload accepted, its completion still owed to a wait
    var default_waits: usize = 0;
    fn busy() bool {
        return pending_default or pending_encoder;
    }
    fn on(s: cuda.abi.Stream) *bool {
        return if (s == encoder) &pending_encoder else &pending_default;
    }
    fn alloc(_: *anyopaque, n: usize, al: std.mem.Alignment, ra: usize) ?[*]u8 {
        return std.testing.allocator.rawAlloc(n, al, ra);
    }
    fn resize(_: *anyopaque, _: []u8, _: std.mem.Alignment, _: usize, _: usize) bool {
        return false;
    }
    fn remap(_: *anyopaque, _: []u8, _: std.mem.Alignment, _: usize, _: usize) ?[*]u8 {
        return null;
    }
    fn free(_: *anyopaque, b: []u8, al: std.mem.Alignment, ra: usize) void {
        if (busy()) bad_free += 1;
        std.testing.allocator.rawFree(b, al, ra);
    }
    fn copy(_: u64, _: ?*const anyopaque, _: usize, s: cuda.abi.Stream) callconv(.c) c_int {
        copies += 1;
        if (copies == fail_copy) return cuda.abi.error_not_ready;
        on(s).* = true;
        return 0;
    }
    fn sync(s: cuda.abi.Stream) callconv(.c) c_int {
        waits += 1;
        if (s != encoder) default_waits += 1;
        if (on(s).* and fail_waits > 0) {
            fail_waits -= 1;
            return cuda.abi.error_not_ready;
        }
        on(s).* = false; // a stream's wait proves only its own copies
        return 0;
    }
    fn context() callconv(.c) c_int {
        pending_default = false;
        pending_encoder = false;
        return 0;
    }
    fn launch(_: cuda.abi.Function, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: cuda.abi.Stream, _: ?[*]?*anyopaque, _: ?[*]?*anyopaque) callconv(.c) c_int {
        return if (mode == 2) cuda.abi.error_not_ready else 0;
    }
    const MM = struct {
        lib: *const cuda.cublaslt.Library,
        pub fn run(_: *@This(), _: usize, _: usize, _: usize, _: u64, _: u64, _: u64, _: u64, _: usize, _: cuda.Stream) !void {
            if (mode == 1) return error.InjectedPlanFailure;
        }
        pub fn deinit(_: *@This()) void {
            if (busy()) bad_free += 1;
        }
    };
    fn dalloc(out: *u64, _: usize) callconv(.c) c_int {
        live += 1;
        out.* = live * 0x10000;
        return 0;
    }
    fn dfree(_: u64) callconv(.c) c_int {
        if (busy()) bad_free += 1;
        live -= 1;
        return 0;
    }
    fn upload(_: u64, _: ?*const anyopaque, _: usize) callconv(.c) c_int {
        if (htod_pends) pending_default = true;
        return 0;
    }
    // the copy-source owner's pinned sources and events
    fn hostAlloc(out: *?*anyopaque, n: usize, _: c_uint) callconv(.c) c_int {
        const bytes = std.testing.allocator.alignedAlloc(u8, .@"16", n + 16) catch return cuda.abi.error_not_ready;
        std.mem.writeInt(usize, bytes[0..@sizeOf(usize)], n + 16, .little);
        out.* = bytes.ptr + 16;
        return 0;
    }
    fn hostFree(ptr: ?*anyopaque) callconv(.c) c_int {
        if (busy()) bad_free += 1;
        const base: [*]align(16) u8 = @alignCast(@as([*]u8, @ptrCast(ptr.?)) - 16);
        const n = std.mem.readInt(usize, base[0..@sizeOf(usize)], .little);
        std.testing.allocator.free(base[0..n]);
        return 0;
    }
    fn eventCreate(out: *cuda.abi.Event, _: c_uint) callconv(.c) c_int {
        out.* = @ptrFromInt(1);
        return 0;
    }
    fn eventRecord(_: cuda.abi.Event, _: cuda.abi.Stream) callconv(.c) c_int {
        return 0;
    }
    fn eventQuery(_: cuda.abi.Event) callconv(.c) c_int {
        return cuda.abi.error_not_ready;
    }
    fn eventDestroy(_: cuda.abi.Event) callconv(.c) c_int {
        return 0;
    }
    fn setup(d: *cuda.Driver, t: *v.Tower) void {
        pending_default = false;
        pending_encoder = false;
        default_waits = 0;
        bad_free = 0;
        copies = 0;
        fail_copy = 0;
        waits = 0;
        fail_waits = 0;
        mode = 0;
        live = 0;
        htod_pends = false;
        d.* = undefined;
        d.api.cuMemcpyHtoDAsync_v2 = copy;
        d.api.cuStreamSynchronize = sync;
        d.api.cuCtxSynchronize = context;
        d.api.cuLaunchKernel = launch;
        d.api.cuMemAlloc_v2 = dalloc;
        d.api.cuMemFree_v2 = dfree;
        d.api.cuMemcpyHtoD_v2 = upload;
        d.api.cuMemHostAlloc = hostAlloc;
        d.api.cuMemFreeHost = hostFree;
        d.api.cuEventCreate = eventCreate;
        d.api.cuEventRecord = eventRecord;
        d.api.cuEventQuery = eventQuery;
        d.api.cuEventDestroy_v2 = eventDestroy;
        t.* = undefined;
        t.gpa = .{ .ptr = t, .vtable = &.{ .alloc = alloc, .resize = resize, .remap = remap, .free = free } };
        t.d = d;
        t.s = .{ .d = d, .handle = encoder };
        t.c = .{ .hidden = 64, .heads = 1, .depth = 1, .intermediate = 64, .out_hidden = 2560, .image_token = 50, .video_token = 51, .vision_start = 52, .vision_end = 53 };
        t.scratch = 0x100000;
        t.blocks = std.mem.zeroes(@TypeOf(t.blocks));
        inline for (@typeInfo(@TypeOf(t.f)).@"struct".field_names) |name| @field(t.f, name) = .{ .d = d, .handle = null };
    }
    fn encode(p: *anyopaque, px: []const f32, grids: []const v.Grid, out: u64) anyerror!void {
        const t: *v.Tower = @ptrCast(@alignCast(p));
        return t.encodeWith(px, grids, out, MM);
    }
};
fn failure(mode: usize, copy: usize, waits: usize) !void {
    var d: cuda.Driver = undefined;
    var t: v.Tower = undefined;
    M.setup(&d, &t);
    M.mode = mode;
    M.fail_copy = copy;
    M.fail_waits = waits;
    const px = try std.testing.allocator.alloc(f32, 4 * v.patch_width);
    defer std.testing.allocator.free(px);
    @memset(px, 0.25);
    const result = v.media.Media.init(t.gpa, .{ .ctx = &t, .config = t.c, .driver = &d, .stream = t.s, .splice = .{ .d = &d, .handle = null }, .encode = M.encode }, &.{ 7, 52, 50, 53, 9 }, &.{.{ .pixels = px, .gh = 2, .gw = 2, .tokens = 1, .at = 2 }}, .{});
    if (result) |value| {
        var m = value;
        m.deinit();
        return error.ExpectedFailure;
    } else |_| {}
    try std.testing.expectEqual(@as(usize, if (copy == 2) 2 else 4), M.copies);
    try std.testing.expectError(if (mode == 1) error.InjectedPlanFailure else error.NotReady, result);
    const bad = M.bad_free;
    M.pending_default = false;
    M.pending_encoder = false;
    try std.testing.expectEqual(@as(usize, 0), bad);
    try std.testing.expectEqual(@as(usize, 0), M.live);
    // A second request may reuse scratch only after the previous error drained.
    M.mode = 0;
    M.fail_copy = 0;
    M.fail_waits = 0;
    var again = try v.media.Media.init(t.gpa, .{ .ctx = &t, .config = t.c, .driver = &d, .stream = t.s, .splice = .{ .d = &d, .handle = null }, .encode = M.encode }, &.{ 7, 52, 50, 53, 9 }, &.{.{ .pixels = px, .gh = 2, .gw = 2, .tokens = 1, .at = 2 }}, .{});
    again.deinit();
    try std.testing.expectEqual(@as(usize, 0), M.live);
}
test "a failed second upload drains the stream before the host rows and features are freed" {
    try failure(0, 2, 0);
}
test "a failed matmul plan drains the stream before any owner is freed" {
    try failure(1, 0, 0);
}
test "a failed kernel launch drains the stream before any owner is freed" {
    try failure(2, 0, 0);
}
test "a failed final wait is retried before anything is freed" {
    try failure(0, 0, 1);
}
fn uploadWait(follower: bool) !void {
    var d: cuda.Driver = undefined;
    var t: v.Tower = undefined;
    M.setup(&d, &t);
    M.pending_default = true; // the fixture's own rule first: the encoder's wait leaves a default-stream copy owed
    try t.s.synchronize();
    try std.testing.expect(M.pending_default);
    M.pending_default = false;
    M.default_waits = 0;
    M.htod_pends = true;
    M.fail_waits = 2; // the upload's own wait fails, then the drain's stream wait; the context's completes
    const enc: v.media.Encoder = .{ .ctx = &t, .config = t.c, .driver = &d, .stream = t.s, .splice = .{ .d = &d, .handle = null }, .encode = M.encode };
    if (follower) {
        const p = try v.media.Plan.init(t.gpa, 5, &.{.{ .at = 2, .gh = 2, .gw = 2 }}, .{});
        try std.testing.expectError(error.NotReady, v.media.Media.allocate(p, enc));
    } else {
        const px = try std.testing.allocator.alloc(f32, 4 * v.patch_width);
        defer std.testing.allocator.free(px);
        const images = [_]@import("lanes").stream.Image{.{ .pixels = px, .gh = 2, .gw = 2, .tokens = 1, .at = 2 }};
        try std.testing.expectError(error.NotReady, v.media.Media.init(t.gpa, enc, &.{ 7, 52, 50, 53, 9 }, &images, .{}));
    }
    const bad = M.bad_free;
    const owed = M.busy();
    M.pending_default = false;
    M.pending_encoder = false;
    try std.testing.expectEqual(@as(usize, 0), bad);
    try std.testing.expect(!owed);
    // the upload's wait and the drain's, both on the default stream: the encoder's wait proves none of its copies
    try std.testing.expectEqual(@as(usize, 2), M.default_waits);
    try std.testing.expectEqual(@as(usize, 0), M.live);
}
test "an image plan whose upload fails its wait drains before the plan and buffers are freed" {
    try uploadWait(false);
}
test "a rank's received plan whose upload fails its wait drains before anything is freed" {
    try uploadWait(true);
}
test "a tower load upload that fails its wait drains before load can free its rows or the checkpoint" {
    var d: cuda.Driver = undefined;
    var t: v.Tower = undefined;
    M.setup(&d, &t);
    t.arena = .{ .d = &d, .ptr = 0x200000, .len = 4096 };
    t.at = 0;
    M.htod_pends = true;
    M.fail_waits = 2;
    const rows: [64]u8 = @splat(1);
    try std.testing.expectError(error.NotReady, t.put(&rows));
    try std.testing.expect(!M.busy());
}
const Child = struct {
    extern "c" fn fork() c_int;
    extern "c" fn kill(c_int, c_int) c_int;
};
test "a stream and context that never complete end the process before an owner is freed" {
    if (@import("builtin").os.tag == .windows) return error.SkipZigTest; // the child process comes from fork
    const pid = Child.fork();
    if (pid < 0) return error.ForkFailed;
    if (pid == 0) {
        var d: cuda.Driver = undefined;
        var t: v.Tower = undefined;
        M.setup(&d, &t);
        d.api.cuCtxSynchronize = struct {
            fn fail() callconv(.c) c_int {
                return cuda.abi.error_not_ready;
            }
        }.fail;
        M.fail_waits = 100;
        M.mode = 1;
        var px: [4 * v.patch_width]f32 = @splat(0);
        t.encodeWith(&px, &.{.{ .h = 2, .w = 2 }}, 0x10000, M.MM) catch {};
        std.c._exit(9);
    }
    var reaped = false;
    defer if (!reaped) {
        _ = Child.kill(pid, 9);
        var status: c_int = 0;
        _ = std.c.waitpid(pid, &status, 0);
    };
    for (0..2000) |_| {
        var status: c_int = 0;
        if (std.c.waitpid(pid, &status, 1) == pid) {
            reaped = true;
            try std.testing.expectEqual(@as(c_int, 0), status & 0x7f);
            try std.testing.expectEqual(@as(c_int, 1), (status >> 8) & 0xff);
            return;
        }
        std.Io.sleep(std.testing.io, .fromMilliseconds(1), .awake) catch {};
    }
    return error.ChildTimeout;
}
