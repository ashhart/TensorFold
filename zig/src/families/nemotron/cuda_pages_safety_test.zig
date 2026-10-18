//! Actual paged constructors under accepted clears, CPU allocation failure and completion/free sentinels.
const std = @import("std");
const cuda = @import("cuda");
const state = @import("cuda_state.zig");
const Engine = @import("cuda_engine.zig").Engine;
const Pages = @import("cuda_pages.zig").Pages;
const a = std.testing.allocator;
const Case = enum { clear, slots, cpu, wait, close, success };
const F = struct {
    var pending = false;
    var fail_stream = false;
    var fail_context = false;
    var fail_clear = false;
    var fail_cpu = false;
    var fatal = false;
    var allocated: usize = 0;
    var freed: usize = 0;
    var clears: usize = 0;
    var early: usize = 0;
    var cpu_allocs: usize = 0;
    var cpu_frees: usize = 0;
    var written: [64]bool = @splat(false);
    var live: [64]bool = @splat(false);
    fn alloc(out: *u64, _: usize) callconv(.c) cuda.abi.Result {
        allocated += 1;
        live[allocated] = true;
        out.* = allocated * 4096;
        return 0;
    }
    fn free(ptr: u64) callconv(.c) cuda.abi.Result {
        const id = ptr / 4096;
        std.debug.assert(live[id]);
        if (pending and written[id]) {
            early += 1;
            if (fatal) std.c._exit(71);
        }
        live[id] = false;
        freed += 1;
        return 0;
    }
    fn clear(ptr: u64, _: u8, _: usize, _: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
        written[ptr / 4096] = true;
        pending = true;
        clears += 1;
        return if (fail_clear) cuda.abi.error_not_ready else 0;
    }
    fn wait(_: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
        if (fail_stream) return cuda.abi.error_not_ready;
        pending = false;
        return 0;
    }
    fn context() callconv(.c) cuda.abi.Result {
        if (fail_context) return cuda.abi.error_not_ready;
        pending = false;
        return 0;
    }
    fn cpuAlloc(_: *anyopaque, n: usize, al: std.mem.Alignment, ra: usize) ?[*]u8 {
        if (fail_cpu and clears > 0) return null;
        const p = a.rawAlloc(n, al, ra) orelse return null;
        cpu_allocs += 1;
        return p;
    }
    fn resize(_: *anyopaque, _: []u8, _: std.mem.Alignment, _: usize, _: usize) bool {
        return false;
    }
    fn remap(_: *anyopaque, _: []u8, _: std.mem.Alignment, _: usize, _: usize) ?[*]u8 {
        return null;
    }
    fn cpuFree(_: *anyopaque, bytes: []u8, al: std.mem.Alignment, ra: usize) void {
        cpu_frees += 1;
        a.rawFree(bytes, al, ra);
    }
    fn allocator() std.mem.Allocator {
        return .{ .ptr = undefined, .vtable = &.{ .alloc = cpuAlloc, .resize = resize, .remap = remap, .free = cpuFree } };
    }
    fn driver(case: Case, die: bool) cuda.Driver {
        pending = false;
        allocated = 0;
        freed = 0;
        clears = 0;
        early = 0;
        cpu_allocs = 0;
        cpu_frees = 0;
        written = @splat(false);
        live = @splat(false);
        fail_stream = case == .wait or die;
        fail_context = die;
        fatal = die;
        fail_clear = case == .clear;
        fail_cpu = case == .cpu;
        var d: cuda.Driver = undefined;
        d.api.cuMemAlloc_v2 = alloc;
        d.api.cuMemFree_v2 = free;
        d.api.cuMemsetD8Async = clear;
        d.api.cuStreamSynchronize = wait;
        d.api.cuCtxSynchronize = context;
        return d;
    }
};
fn run(case: Case, die: bool, cpu: ?std.mem.Allocator) !void {
    var d = F.driver(case, die);
    var ctx: cuda.Context = undefined;
    ctx.d = &d;
    var e: Engine = undefined;
    e.c = std.mem.zeroes(@import("config.zig").Config);
    e.c.layers = 1;
    e.c.kinds[0] = .attention;
    e.c.hidden = 8;
    e.c.kv_heads = 1;
    e.c.head_dim = 8;
    e.k.triton = null;
    e.head = null;
    e.max_len = 1024;
    e.ctx = &ctx;
    e.stream = .{ .d = &d, .handle = @ptrFromInt(1) };
    e.gpa = cpu orelse F.allocator();
    e.bound = &e.own;
    e.slots = if (case == .slots) std.math.maxInt(u64) else 0;
    const slots = e.slots;
    if (case == .clear) {
        const result = state.Seq.initPaged(&d, e.c, &.{}, e.stream, 0, 2);
        try std.testing.expectError(error.NotReady, result);
    } else {
        const p = Pages.init(e.gpa, &e, 2, 2) catch |err| {
            const want: anyerror = if (case == .slots) error.TooManySequences else if (case == .cpu or cpu != null) error.OutOfMemory else error.NotReady;
            try std.testing.expectEqual(want, err);
            try std.testing.expectEqual(@as(usize, 0), F.early);
            try std.testing.expectEqual(F.allocated, F.freed);
            try std.testing.expectEqual(slots, e.slots);
            if (cpu == null) try std.testing.expectEqual(F.cpu_allocs, F.cpu_frees);
            if (cpu != null) return err;
            return;
        };
        if (case == .close) {
            try p.lanes[0].seq.arena.?.buf.fill8(0, e.stream.handle);
            F.fail_stream = true;
        }
        p.deinit();
        if (case != .success and case != .close) return error.ExpectedConstructorFailure;
    }
    try std.testing.expectEqual(@as(usize, 0), F.early);
    try std.testing.expectEqual(F.allocated, F.freed);
    if (cpu == null) try std.testing.expectEqual(F.cpu_allocs, F.cpu_frees);
    try std.testing.expectEqual(slots, e.slots);
}
test "kv safety: Seq.initPaged drains an accepted clear before its inner arena error cleanup" {
    try run(.clear, false, null);
}
test "kv safety: Pages.init drains inner slot-check and outer CPU/wait failure cleanup" {
    for ([_]Case{ .slots, .cpu, .wait, .close, .success }) |case| try run(case, false, null);
}
fn allocationTrial(gpa: std.mem.Allocator) !void {
    try run(.success, false, gpa);
}
test "kv safety: every CPU allocation failure in the actual paged constructor balances owners and slots" {
    try std.testing.checkAllAllocationFailures(a, allocationTrial, .{});
}
const Child = struct {
    extern "c" fn fork() c_int;
};
test "kv safety: paged constructor fatal completion failure precedes every destination or parent release" {
    if (@import("builtin").os.tag == .windows) return error.SkipZigTest; // the child process comes from fork
    for ([_]Case{ .clear, .slots, .cpu, .wait }) |case| {
        const pid = Child.fork();
        if (pid < 0) return error.ForkFailed;
        if (pid == 0) {
            run(case, true, null) catch std.c._exit(72);
            std.c._exit(9);
        }
        var status: c_int = 0;
        _ = std.c.waitpid(pid, &status, 0);
        try std.testing.expectEqual(@as(c_int, 0), status & 0x7f);
        try std.testing.expectEqual(@as(c_int, 1), (status >> 8) & 0xff);
    }
}
