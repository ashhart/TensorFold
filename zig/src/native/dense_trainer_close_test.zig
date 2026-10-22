//! The actual native close, slide adapter, generic learner and dense 27B trainer under pending-work sentinels.
const std = @import("std");
const cuda = @import("cuda");
const flashnext = @import("qwen35");
const lanes = @import("lanes");
const Adapter = @import("slide_adapter.zig").Adapter(flashnext.native.slide);
const T = flashnext.train.Trainer;
const a = std.testing.allocator;
const F = struct {
    var pending = false;
    var fail_wait = false;
    var fail_context = false;
    var fatal_sentinel = false;
    var early: usize = 0;
    var host_freed: usize = 0;
    var device_freed: usize = 0;
    var unloaded: usize = 0;
    var parents: usize = 0;
    var memory: [8][16]u8 align(16) = undefined;
    var allocated: usize = 0;
    var source: usize = 0;
    fn reset() void {
        pending = false;
        early = 0;
        host_freed = 0;
        device_freed = 0;
        unloaded = 0;
        parents = 0;
        allocated = 0;
        source = 0;
    }
    fn check() void {
        if (pending) {
            early += 1;
            if (fatal_sentinel) std.c._exit(71);
        }
    }
    fn alloc(out: *u64, _: usize) callconv(.c) cuda.abi.Result {
        out.* = @intFromPtr(&memory[allocated]);
        allocated += 1;
        return 0;
    }
    fn hostAlloc(out: *?*anyopaque, n: usize, _: c_uint) callconv(.c) cuda.abi.Result {
        var ptr: u64 = 0;
        const code = alloc(&ptr, n);
        out.* = @ptrFromInt(ptr);
        return code;
    }
    fn mapped(out: *u64, ptr: ?*anyopaque, _: c_uint) callconv(.c) cuda.abi.Result {
        out.* = @intFromPtr(ptr.?);
        return 0;
    }
    fn hostFree(_: ?*anyopaque) callconv(.c) cuda.abi.Result {
        check();
        host_freed += 1;
        return 0;
    }
    fn devFree(ptr: u64) callconv(.c) cuda.abi.Result {
        if (ptr != 0) {
            check();
            device_freed += 1;
        }
        return 0;
    }
    fn unload(_: cuda.abi.Module) callconv(.c) cuda.abi.Result {
        check();
        unloaded += 1;
        return 0;
    }
    fn wait(_: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
        if (fail_wait and pending) return cuda.abi.error_not_ready;
        pending = false;
        return 0;
    }
    fn context() callconv(.c) cuda.abi.Result {
        if (fail_context) return cuda.abi.error_not_ready;
        pending = false;
        return 0;
    }
    fn launch(_: cuda.abi.Function, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: cuda.abi.Stream, _: ?[*]?*anyopaque, _: ?[*]?*anyopaque) callconv(.c) cuda.abi.Result {
        pending = true;
        return 0;
    }
    fn module(d: *cuda.Driver) cuda.Module {
        return .{ .d = d, .handle = @ptrFromInt(8) };
    }
    fn allocator() std.mem.Allocator {
        return .{ .ptr = undefined, .vtable = &.{ .alloc = cpuAlloc, .resize = cpuResize, .remap = cpuRemap, .free = cpuFree } };
    }
    fn cpuAlloc(_: *anyopaque, n: usize, al: std.mem.Alignment, ra: usize) ?[*]u8 {
        return a.rawAlloc(n, al, ra);
    }
    fn cpuResize(_: *anyopaque, bytes: []u8, al: std.mem.Alignment, n: usize, ra: usize) bool {
        return a.rawResize(bytes, al, n, ra);
    }
    fn cpuRemap(_: *anyopaque, bytes: []u8, al: std.mem.Alignment, n: usize, ra: usize) ?[*]u8 {
        return a.rawRemap(bytes, al, n, ra);
    }
    fn cpuFree(_: *anyopaque, bytes: []u8, al: std.mem.Alignment, ra: usize) void {
        if (@intFromPtr(bytes.ptr) == source) check();
        a.rawFree(bytes, al, ra);
    }
    fn family(p: *anyopaque) void {
        check();
        parents += 1;
        const e: *flashnext.engine.Engine = @ptrCast(@alignCast(p));
        a.destroy(e);
    }
};
pub fn close(comptime Host: type, comptime serve: anytype, comptime closeHost: anytype) !void {
    F.reset();
    var d: cuda.Driver = .{ .lib = undefined, .api = undefined };
    d.api.cuMemAlloc_v2 = F.alloc;
    d.api.cuMemFree_v2 = F.devFree;
    d.api.cuMemHostAlloc = F.hostAlloc;
    d.api.cuMemHostGetDevicePointer_v2 = F.mapped;
    d.api.cuMemFreeHost = F.hostFree;
    d.api.cuModuleUnload = F.unload;
    d.api.cuStreamSynchronize = F.wait;
    d.api.cuCtxSynchronize = F.context;
    d.api.cuLaunchKernel = F.launch;
    const stream: cuda.Stream = .{ .d = &d, .handle = null };
    const e = try a.create(flashnext.engine.Engine);
    e.* = undefined;
    e.d = &d;
    e.stream = stream;
    const t = try a.create(T);
    t.* = .{ .gpa = a, .e = e, .g = undefined, .k = undefined, .gk = undefined };
    inline for (.{ "host", "ys", "delta", "push", "tails" }) |field| {
        const host = try cuda.HostBuffer.allocMapped(&d, 16);
        @field(t, field) = .{ .host = host, .dev = try host.device() };
    }
    try t.owned.append(a, try cuda.DeviceBuffer.alloc(&d, 16));
    for (&t.states) |*cache| cache.* = .{ .state = .{ .d = &d, .ptr = 0, .len = 0 }, .kv = .{ .d = &d, .ptr = 0, .len = 0 }, .log = .{ .d = &d, .ptr = 0, .len = 0 }, .logits = &.{}, .cap = 0, .len = 0, .written = 0 };
    t.states[0].state = try cuda.DeviceBuffer.alloc(&d, 16);
    t.k = .{ .own = F.module(&d), .attn = F.module(&d), .train = F.module(&d), .fo = undefined, .fa = undefined, .ft = undefined };
    t.gk = .{ .module = F.module(&d), .train = F.module(&d), .fns = undefined, .train_fns = undefined };
    const adapter = try a.create(Adapter);
    adapter.* = Adapter.init(a, std.testing.io, flashnext.native.slide.Model{ .e = e, .dir = "", .lay = @FieldType(flashnext.native.slide.Model, "lay").fromExperts(64, 128, 8, .empty) });
    adapter.learner.trainer = t;
    adapter.learner.pocketed = true;
    adapter.learner.kept_rounds = 1;
    adapter.learner.pockets.gpa = F.allocator();
    try adapter.learner.pockets.rows.appendNTimes(F.allocator(), 0, 8);
    F.source = @intFromPtr(adapter.learner.pockets.rows.items.ptr);
    var backend: lanes.fake.Fake = .{ .gpa = a };
    defer backend.deinit();
    const h = try a.create(Host);
    h.* = .{ .gpa = a, .gpu = null, .family = e, .release = F.family, .inner = backend.backend(), .vtable = undefined, .cfg = undefined, .clock = undefined, .core = undefined, .host = undefined };
    try serve(h, std.testing.io, .{ .exact_width = 8 }, 8, .{ .lanes = 1 }, null);
    h.served = .{ .backend = backend.backend(), .facts = .{ .exact_width = 8 }, .stream_bytes = 0 };
    adapter.learner.pockets.save_pending = true;
    const hook = adapter.hook();
    try std.testing.expect(hook.step(hook.ctx).stale);
    adapter.learner.pockets.save_pending = false;
    h.slide = adapter;
    h.slide_free = struct {
        fn free(gpa: std.mem.Allocator, p: *anyopaque) void {
            const s: *Adapter = @ptrCast(@alignCast(p));
            s.deinit();
            gpa.destroy(s);
        }
    }.free;
    var args: cuda.Args = .{};
    args.add(t.ys.dev);
    args.add(t.tails.dev);
    args.add(t.delta.dev);
    args.add(t.push.dev);
    try cuda.launch.launch(.{ .d = &d, .handle = @ptrFromInt(9) }, .{ .grid = .{ .x = 1 }, .block = .{ .x = 1 } }, stream, &args);
    closeHost(h);
    try std.testing.expectEqual(@as(usize, 0), F.early);
    try std.testing.expectEqual(@as(usize, 5), F.host_freed);
    try std.testing.expectEqual(@as(usize, 2), F.device_freed);
    try std.testing.expectEqual(@as(usize, 5), F.unloaded);
    try std.testing.expectEqual(@as(usize, 1), F.parents);
    try std.testing.expect(!F.pending);
}
const Child = struct {
    extern "c" fn fork() c_int;
};
pub fn run(comptime Host: type, comptime serve: anytype, comptime closeHost: anytype) !void {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const dir = try tmp.dir.realPathFileAlloc(std.testing.io, ".", a);
    defer a.free(dir);
    try @import("engine_api").save_guard.begin(std.testing.io, dir);
    var problem: []const u8 = "";
    try std.testing.expect(@import("save_admission.zig").refused(std.testing.io, dir, &problem));
    try std.testing.expect(std.mem.indexOf(u8, problem, "unresolved Living Weights save") != null);
    try @import("engine_api").save_guard.finish(std.testing.io, dir);
    try std.testing.expect(!@import("save_admission.zig").refused(std.testing.io, dir, &problem));
    F.fail_wait = true;
    F.fail_context = false;
    F.fatal_sentinel = false;
    try close(Host, serve, closeHost);
    F.fail_wait = false;
    try close(Host, serve, closeHost);
    if (@import("builtin").os.tag == .windows) return error.SkipZigTest; // the fatal case's child comes from fork
    const pid = Child.fork();
    if (pid < 0) return error.ForkFailed;
    if (pid == 0) {
        F.fail_wait = true;
        F.fail_context = true;
        F.fatal_sentinel = true;
        close(Host, serve, closeHost) catch std.c._exit(72);
        std.c._exit(9);
    }
    var status: c_int = 0;
    _ = std.c.waitpid(pid, &status, 0);
    try std.testing.expectEqual(@as(c_int, 0), status & 0x7f);
    try std.testing.expectEqual(@as(c_int, 1), (status >> 8) & 0xff);
}
