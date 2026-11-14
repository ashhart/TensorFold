//! TF_GEMMA4_V_ONLY's refusals fire in open before any kernel, weight, stream or copy: driver and allocator counted.
const std = @import("std");
const cuda = @import("cuda");
const native = @import("cuda_native.zig");

const t = std.testing;
const abi = cuda.abi;
const env = @import("sys").env;

/// Counts allocations over a fresh arena, so a grown buffer takes the same path on every run; frees pass through.
const Counted = struct {
    child: std.mem.Allocator,
    n: usize = 0,

    fn allocator(c: *Counted) std.mem.Allocator {
        return .{ .ptr = c, .vtable = &vt };
    }

    const vt: std.mem.Allocator.VTable = .{ .alloc = alloc, .resize = resize, .remap = remap, .free = free };

    fn of(p: *anyopaque) *Counted {
        return @ptrCast(@alignCast(p));
    }
    fn alloc(p: *anyopaque, len: usize, a: std.mem.Alignment, ra: usize) ?[*]u8 {
        of(p).n += 1;
        return of(p).child.rawAlloc(len, a, ra);
    }
    fn resize(p: *anyopaque, m: []u8, a: std.mem.Alignment, len: usize, ra: usize) bool {
        return of(p).child.rawResize(m, a, len, ra);
    }
    fn remap(p: *anyopaque, m: []u8, a: std.mem.Alignment, len: usize, ra: usize) ?[*]u8 {
        return of(p).child.rawRemap(m, a, len, ra);
    }
    fn free(p: *anyopaque, m: []u8, a: std.mem.Alignment, ra: usize) void {
        of(p).child.rawFree(m, a, ra);
    }
};

/// A driver whose module loads, streams, allocations and copies only count (and fail); anything else is unset.
const Device = struct {
    var calls: usize = 0;
    const no: abi.Result = abi.error_not_ready;
    fn load(_: *abi.Module, _: ?*const anyopaque) callconv(.c) abi.Result {
        calls += 1;
        return no;
    }
    fn stream(_: *abi.Stream, _: c_uint) callconv(.c) abi.Result {
        calls += 1;
        return no;
    }
    fn alloc(_: *abi.DevicePtr, _: usize) callconv(.c) abi.Result {
        calls += 1;
        return no;
    }
    fn copy(_: abi.DevicePtr, _: ?*const anyopaque, _: usize) callconv(.c) abi.Result {
        calls += 1;
        return no;
    }
    fn copyAsync(_: abi.DevicePtr, _: ?*const anyopaque, _: usize, _: abi.Stream) callconv(.c) abi.Result {
        calls += 1;
        return no;
    }
    fn driver() cuda.Driver {
        calls = 0;
        var d: cuda.Driver = .{ .lib = undefined, .api = undefined };
        d.api.cuModuleLoadData = load;
        d.api.cuStreamCreate = stream;
        d.api.cuMemAlloc_v2 = alloc;
        d.api.cuMemcpyHtoD_v2 = copy;
        d.api.cuMemcpyHtoDAsync_v2 = copyAsync;
        return d;
    }
};

const Case = struct {
    config: []const u8,
    v_only: [:0]const u8,
    share: bool = false,
    o: native.Options,
    want: anyerror,
};

const c31 = @embedFile("fixtures/config.json");
const e4b = @embedFile("fixtures/config_e4b.json");
const c12b = @embedFile("fixtures/config_12b.json");
const off: native.Options = .{ .context = 4096, .drafts = false };
const drafter: native.Options = .{ .context = 4096, .drafts = true, .drafter = "assistant" };
const linear: native.Options = .{ .context = 4096, .drafts = false, .kv_pages = false };

const cases = [_]Case{
    .{ .config = c31, .v_only = "1", .o = drafter, .want = error.GemmaVOnlyDrafter },
    .{ .config = c31, .v_only = "1", .share = true, .o = off, .want = error.GemmaVOnlyKvShare },
    .{ .config = c31, .v_only = "1", .o = linear, .want = error.GemmaVOnlyUnpaged },
    .{ .config = e4b, .v_only = "1", .o = off, .want = error.GemmaVOnlyKeysNotValues },
    .{ .config = c12b, .v_only = "1", .o = off, .want = error.GemmaVOnlyOneKeyHead },
    .{ .config = c31, .v_only = "on", .o = off, .want = error.BadGemmaVOnly },
};

/// open's error and allocations under TF_GEMMA4_V_ONLY=v (and TF_KV_SHARE=1 if share), over `config` alone.
fn opened(dir: []const u8, v: [:0]const u8, share: bool, o: native.Options) !struct { err: anyerror, allocs: usize } {
    defer _ = env.unsetenv("TF_GEMMA4_V_ONLY");
    defer _ = env.unsetenv("TF_KV_SHARE");
    try t.expectEqual(@as(c_int, 0), env.setenv("TF_GEMMA4_V_ONLY", v, 1));
    if (share) try t.expectEqual(@as(c_int, 0), env.setenv("TF_KV_SHARE", "1", 1));
    var d = Device.driver();
    const ctx: cuda.Context = .{ .d = &d, .device = 0, .handle = null };
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    var c: Counted = .{ .child = arena.allocator() };
    const got = native.open(c.allocator(), t.io, &ctx, dir, null, o);
    if (got) |_| return error.OpenedUnderTest else |e| return .{ .err = e, .allocs = c.n };
}

/// The allocations reading `dir`'s config takes, counted as `opened` counts them.
fn readAllocs(dir: []const u8) !usize {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    var c: Counted = .{ .child = arena.allocator() };
    _ = try @import("cuda_config.zig").Config.read(c.allocator(), t.io, dir);
    return c.n;
}

test "TF_GEMMA4_V_ONLY's refusals fire before open allocates anything but the config's text or touches the device" {
    for (cases) |c| {
        var tmp = t.tmpDir(.{});
        defer tmp.cleanup();
        try tmp.dir.writeFile(t.io, .{ .sub_path = "config.json", .data = c.config });
        const dir = try tmp.dir.realPathFileAlloc(t.io, ".", t.allocator);
        defer t.allocator.free(dir);
        const read = try readAllocs(dir);
        const got = try opened(dir, c.v_only, c.share, c.o);
        try t.expectEqual(c.want, got.err);
        try t.expectEqual(@as(usize, 0), Device.calls); // no module, stream, allocation or copy
        // the config's text is all it read: no Owned, no kernels, no weights (a bad value reads not even that)
        try t.expectEqual(if (c.want == error.BadGemmaVOnly) 0 else read, got.allocs);
        try t.expect(native.explain(null, got.err) != null);
    }
    if (cuda.kernels.available) return;
    // red control: the 31B with the mode on goes past the gate, to Owned and the kernels this build lacks
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(t.io, .{ .sub_path = "config.json", .data = c31 });
    const dir = try tmp.dir.realPathFileAlloc(t.io, ".", t.allocator);
    defer t.allocator.free(dir);
    const read = try readAllocs(dir);
    const on = try opened(dir, "1", false, off);
    try t.expectEqual(error.BuiltWithoutKernels, on.err);
    try t.expect(on.allocs > read);
}
